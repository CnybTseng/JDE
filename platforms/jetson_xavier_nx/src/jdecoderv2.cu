#include <float.h>
#include <cuda_runtime_api.h>

#include "jdecoderv2.h"

namespace nvinfer1 {

__device__ void softmax_inplace(float* X, size_t N)
{
    float max = -FLT_MAX;
    for (size_t i = 0; i < N; ++i) {
        int w = X[i] > max;
        max = w * X[i] + (1 - w) * max;
    }
    
    float sum = 0.f;
    for (size_t i = 0; i < N; ++i) {
        X[i] = expf(X[i] - max);
        sum += X[i];
    }
    
    for (size_t i = 0; i < N; ++i) {
        X[i] /= sum;
    }
}

//! \brief JDE decoder forward kernel function.
//! \param in Input tensor with NHWC memory layout.
//!  The encoded information in tensor channel dimension are x,y,w,h,
//!  [class probabilities],[embedding vector].
//!
//! \param out Decoding output. The output is a N by 134 matrix, and N is the
//!  maximum number of decoding outputs. By default, N is 1000.
//!
//! \param inWidth
//! \param inHeight
//! \param inSize
//! \param outWidth
//! \param outHeight
//! \param outStride
//! \param numClass
//! \param anchor
//!
__global__ void forward_kernel(const float* const in, float* out, int inWidth, int inHeight,
    int inSize, int outWidth, int outHeight, int outStride, int numClass,
    const float anchor[jdec::numAnchor * jdec::anchorDim])
{
    // Thread identifier.
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > inSize) {
        return;
    }
    
    int mapSize = inWidth * inHeight;       // feature map size
    int batchIdx = tid / mapSize;           // batch index
    int inIdx = tid - batchIdx * mapSize;   // index in feature map
    const float* pin = in + tid * jdec::decInputDim;
    float* pout = out + batchIdx * outStride;
    const int netStride = jdec::netInWidth / inWidth;
    
    for (int i = 0; i < jdec::numAnchor; ++i) {
        // Calculate class probabilities with Softmax activation.
        const float* pc = pin + jdec::boxDim;
        softmax_inplace(const_cast<float*>(pc), numClass);
        
        // Find the class with maximum probability.
        int category = 0;
        float score = -FLT_MAX;
        for (int c = 0; c < numClass; ++c) {
            int w = pc[c] > score;
            score = w * pc[c] + (1 - w) * score;
            category = w * c + (1 - w) * category;
        }
        
        // Calculate box parameters.
        int y = inIdx / inWidth;
        int x = inIdx - y * inWidth;
        float* pb = const_cast<float*>(pin);
        float bx = pb[0] * anchor[i << 1] + x * netStride;
        float by = pb[1] * anchor[(i << 1) + 1] + y * netStride;
        float bw = anchor[i << 1] * expf(pb[2]);
        float bh = anchor[(i << 1) + 1] * expf(pb[3]);
        
        // TODO: Normalize embedding vector.
    }
}

JDecoderV2::JDecoderV2()
{
    mNumThread = 256;
    mNumClass = jdec::numClass;
    
    mJDecKernel.push_back(jdec::jdk1);
    mJDecKernel.push_back(jdec::jdk2);
    mJDecKernel.push_back(jdec::jdk3);
    
    mNumKernel = mJDecKernel.size();
    
    mGpuAnchor.resize(mNumKernel);
    for (size_t i = 0; i < mGpuAnchor.size(); ++i) {
        cudaMalloc(&mGpuAnchor[i], sizeof(mJDecKernel[i].anchor));
        cudaMemcpy(mGpuAnchor[i], mJDecKernel[i].anchor, sizeof(mJDecKernel[i].anchor), cudaMemcpyHostToDevice);
    }
}

JDecoderV2::~JDecoderV2()
{
    for (size_t i = 0; i < mGpuAnchor.size(); ++i) {
        cudaFree(mGpuAnchor[i]);
    }
}

void JDecoderV2::forward(const float* const* in, float* out, cudaStream_t stream, int batchSize)
{
    // The first element of buffer is the number of valid boxes.
    // So we only need to set the first element as zero.
    size_t outStride = jdec::maxNumOutputBox * jdec::decOutputDim + 1;
    for (int i = 0; i < batchSize; ++i) {
        cudaMemset(out + i * outStride, 0, sizeof(float));
    }

    for (size_t j = 0; j < mJDecKernel.size(); ++j) {
        int inSize = mJDecKernel[j].inWidth * mJDecKernel[j].inHeight * batchSize;
        if (inSize < mNumThread) {
            mNumThread = inSize;
        }
        
        int numBlocks = (inSize + mNumThread - 1) / mNumThread;
        int blockSize = mNumThread;
        forward_kernel<<<numBlocks, blockSize>>>(in[j], out, mJDecKernel[j].inWidth, mJDecKernel[j].inHeight,
            inSize, jdec::decOutputDim, jdec::maxNumOutputBox, outStride, mNumClass, (float*)mGpuAnchor[j]);
    }
}

}   // namespace nvinfer1