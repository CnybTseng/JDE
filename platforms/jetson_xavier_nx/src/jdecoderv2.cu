#include <float.h>
#include <stdio.h>
#include <helper_math.h>
#include <cuda_runtime_api.h>

#include "jdecoderv2.h"

namespace nvinfer1 {

__device__ void softmax_inplace(float* X, size_t N)
{
    float max = -FLT_MAX;
    for (size_t i = 0; i < N; ++i) {
        // int w = X[i] > max;
        // max = w * X[i] + (1 - w) * max;
        if (X[i] > max) max = X[i];
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

__device__ void normalize(float *X, float* Y, size_t N, float eps=1e-12)
{
    int i = 0;
    float ssum = .0f;
    float2* X2 = reinterpret_cast<float2*>(X);
    for (int j = 0; j < N / 2; i += 2, ++j) {
        ssum += dot(X2[j], X2[j]);
    }
    
    for (; i < N; ++i) {
        ssum += X[i] * X[i];
    }
    
    float s = ssum > eps ? 1.f / ssum : 1.f / eps;
    
    i = 0;
    float2* Y2 = reinterpret_cast<float2*>(Y);
    for (int j = 0; j < N / 2; i += 2, ++j) {
        Y2[j] = s * X2[j];
    }
    
    for (; i < N; ++i) {
        Y[i] = s * X[i];
    }
}

//! \brief JDE decoder forward kernel function.
//! \param in Input tensor with NHWC memory layout.
//!  The encoded information in tensor channel dimension are x,y,w,h,
//!  [class probabilities],[embedding vector].
//!
//! \param out Decoding output. The output is a N by 134 matrix, and N is the
//!  maximum number of decoding outputs. By default, N is 1024.
//!
//! \param inWidth Input tensor width.
//! \param inHeight Input tensor height.
//! \param inSize Number of input tensor elements. The inSize value is N*H*W.
//! \param outWidth Output matrix width. The outWidth is the sum of jdec::boxDim,
//!  jdec::classDim, and jdec::embeDim.
//! \param outHeight Output matrix Height. The outHeight is jdec::maxNumOutputBox.
//! \param numClass Number of classes.
//! \param anchor Anchor boxes array.
//!
__global__ void forward_kernel(const float* const in, float* out, int inWidth, int inHeight,
    int inSize, int outWidth, int outHeight, int numClass,
    const float anchor[jdec::numAnchor * jdec::anchorDim],
    const float confThresh)
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
    int outStride = outWidth * outHeight + 2;
    float* pout = out + batchIdx * outStride;
    const int netStride = jdec::netInWidth / inWidth;

    for (int i = 0; i < jdec::numAnchor; ++i) {
        // Calculate class probabilities with Softmax activation.
        const float* pa = pin + i * (jdec::boxDim + jdec::classDim);
        const float* pc = pa + jdec::boxDim;
        softmax_inplace(const_cast<float*>(pc), numClass);
        
        // Find the class with maximum probability.
        int category = 0;
        float score = -FLT_MAX;
        for (int c = 0; c < numClass; ++c) {
            // int w = pc[c] > score;
            // score = w * pc[c] + (1 - w) * score;
            // category = w * c + (1 - w) * category;
            if (pc[c] > score) {score = pc[c]; category = c;}
        }
        
        if (0 == category || score < confThresh) {
            continue;
        }

        int count = static_cast<int>(atomicAdd(pout, 1));
        if (count > jdec::maxNumOutputBox) {
            return;
        }

        float* pOutRow = pout + count * outWidth + 2;
        
        pOutRow[0] = static_cast<float>(category);
        pOutRow[1] = score;
        
        // Calculate box parameters.
        int y = inIdx / inWidth;
        int x = inIdx - y * inWidth;
        float* pb = const_cast<float*>(pa);
        float bx = pb[0] * anchor[i << 1] + x * netStride;
        float by = pb[1] * anchor[(i << 1) + 1] + y * netStride;
        float bw = anchor[i << 1] * expf(pb[2]);
        float bh = anchor[(i << 1) + 1] * expf(pb[3]);
        
        pOutRow[2] = by - bh * 0.5f;    // top
        pOutRow[3] = bx - bw * 0.5f;    // left
        pOutRow[4] = by + bh * 0.5f;    // bottom
        pOutRow[5] = bx + bw * 0.5f;    // right
        
        // Normalize embedding vector.
        float* pe = const_cast<float*>(pin + (jdec::boxDim + jdec::classDim) * jdec::numAnchor);
        normalize(pe, pOutRow + jdec::boxDim + jdec::classDim, jdec::embeDim);
    }
}

JDecoderV2::JDecoderV2()
{
    mNumThread = 64;
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
    
    mConfThresh = 0.5f;
}

JDecoderV2::~JDecoderV2()
{
    for (size_t i = 0; i < mGpuAnchor.size(); ++i) {
        cudaFree(mGpuAnchor[i]);
    }
}

void JDecoderV2::forward_test(const float* const* in, float* out, cudaStream_t stream, int batchSize)
{
    forward(in, out, stream, batchSize);
}

void JDecoderV2::forward(const float* const* in, float* out, cudaStream_t stream, int batchSize)
{
    // The first element of buffer is the number of valid boxes.
    // So we only need to set the first element as zero.
    size_t outStride = jdec::maxNumOutputBox * jdec::decOutputDim + 2;
    for (int i = 0; i < batchSize; ++i) {
        cudaMemset(out + i * outStride, 0, sizeof(float2));
    }

    for (size_t j = 0; j < mJDecKernel.size(); ++j) {
        int numThread = mNumThread;
        int inSize = mJDecKernel[j].inWidth * mJDecKernel[j].inHeight * batchSize;
        if (inSize < numThread) {
            numThread = inSize;
        }
        
        int numBlocks = (inSize + numThread - 1) / numThread;
        int blockSize = numThread;
        forward_kernel<<<numBlocks, blockSize>>>(in[j], out, mJDecKernel[j].inWidth, mJDecKernel[j].inHeight,
            inSize, jdec::decOutputDim, jdec::maxNumOutputBox, mNumClass, (float*)mGpuAnchor[j], mConfThresh);
    }
}

}   // namespace nvinfer1