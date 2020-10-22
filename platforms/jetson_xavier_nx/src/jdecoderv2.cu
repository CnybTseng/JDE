#include <float.h>
#include <stdio.h>
#include <assert.h>
#include <helper_math.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include "utils.h"
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

template <typename T>
static void write(char*& buffer, const T& data)
{
    *reinterpret_cast<T*>(buffer) = data;
    buffer += sizeof(data);
}

template <typename T>
static T read(const char*& buffer)
{
    T data = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return data;
}

JDecoderPlugin::JDecoderPlugin()
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

JDecoderPlugin::JDecoderPlugin(const void* data, size_t size)
{
    const char* p = reinterpret_cast<const char*>(data);
    const char* b = p;
    mNumThread = read<int32_t>(p);
    mNumClass = read<int32_t>(p);
    mNumKernel = read<int32_t>(p);
    mConfThresh = read<float>(p);
    mJDecKernel.resize(mNumKernel);
    size_t kernelSize = mNumKernel * sizeof(jdec::JDecKernel);
    memcpy(mJDecKernel.data(), p, kernelSize);
    p += kernelSize;
    assert(p == b + size);
    
    mGpuAnchor.resize(mNumKernel);
    for (size_t i = 0; i < mGpuAnchor.size(); ++i) {
        cudaMalloc(&mGpuAnchor[i], sizeof(mJDecKernel[i].anchor));
        cudaMemcpy(mGpuAnchor[i], mJDecKernel[i].anchor, sizeof(mJDecKernel[i].anchor), cudaMemcpyHostToDevice);
    }
}

JDecoderPlugin::~JDecoderPlugin()
{
    for (size_t i = 0; i < mGpuAnchor.size(); ++i) {
        cudaFree(mGpuAnchor[i]);
    }
}

const char* JDecoderPlugin::getPluginType() const
{
    return "JDecoderPlugin";
}

const char* JDecoderPlugin::getPluginVersion() const
{
    return "100";
}

int32_t JDecoderPlugin::getNbOutputs() const
{
    return 1;
}

Dims JDecoderPlugin::getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims)
{
    assert(0 == index);
    assert(3 == nbInputDims);
    int numel = numel_after_align(jdec::maxNumOutputBox * jdec::decOutputDim + 1, sizeof(float), 8);
    return Dims3(numel, 1, 1);
}

int32_t JDecoderPlugin::initialize()
{
    return 0;
}

void JDecoderPlugin::terminate()
{
}

size_t JDecoderPlugin::getWorkspaceSize(int32_t maxBatchSize) const
{
    return 0;
}

int32_t JDecoderPlugin::enqueue(int32_t batchSize, const void* const* inputs, void** outputs, void* workspace,
    cudaStream_t stream)
{
    forward((const float* const*)inputs, (float*)outputs[0], stream, batchSize);
    return 0;
}

size_t JDecoderPlugin::getSerializationSize() const
{
    return sizeof(mNumThread) + sizeof(mNumClass) + sizeof(mNumKernel) +
        sizeof(mConfThresh) + mNumKernel * sizeof(jdec::JDecKernel);
}

void JDecoderPlugin::serialize(void* buffer) const
{
    char* p = reinterpret_cast<char*>(buffer);
    char* b = p;
    write<int32_t>(p, mNumThread);
    write<int32_t>(p, mNumClass);
    write<int32_t>(p, mNumKernel);
    write<float>(p, mConfThresh);
    size_t kernelSize = mNumKernel * sizeof(jdec::JDecKernel);
    memcpy(p, mJDecKernel.data(), kernelSize);
    p += kernelSize;
    assert(p == b + getSerializationSize());
}

void JDecoderPlugin::destroy()
{
    delete this;
}

void JDecoderPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* JDecoderPlugin::getPluginNamespace() const
{
    return mPluginNamespace.c_str();
}

nvinfer1::DataType JDecoderPlugin::getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes,
    int32_t nbInputs) const
{
    assert(0 == index);
    assert(3 == nbInputs);
    return inputTypes[0];
}

bool JDecoderPlugin::isOutputBroadcastAcrossBatch(int32_t outputIndex, const bool* inputIsBroadcasted,
    int32_t nbInputs) const
{
    return false;
}

bool JDecoderPlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const
{
    return false;
}

void JDecoderPlugin::attachToContext(cudnnContext*, cublasContext*, IGpuAllocator*)
{
}

void JDecoderPlugin::detachFromContext()
{
}

IPluginV2Ext* JDecoderPlugin::clone() const
{
    JDecoderPlugin* p = new JDecoderPlugin();
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

void JDecoderPlugin::configurePlugin(const PluginTensorDesc* in, int32_t nbInput, const PluginTensorDesc* out,
    int32_t nbOutput)
{
}

bool JDecoderPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs,
    int32_t nbOutputs) const
{
    int device;
    auto ret = cudaGetDevice(&device);
    if (0 != ret) {
        fprintf(stderr, "cudaGetDevice fail\n");
        abort();
    }

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int sm_version = props.major << 8 | props.minor;

    return (DataType::kFLOAT == inOut[pos].type || (DataType::kHALF == inOut[pos].type && sm_version >= 0x600))
        && PluginFormat::kLINEAR == inOut[pos].format;
}

void JDecoderPlugin::forward_test(const float* const* in, float* out, cudaStream_t stream, int batchSize)
{
    forward(in, out, stream, batchSize);
}

void JDecoderPlugin::forward(const float* const* in, float* out, cudaStream_t stream, int batchSize)
{
    // The first element of buffer is the number of valid boxes.
    // So we only need to set the first element as zero.
    size_t outStride = numel_after_align(jdec::maxNumOutputBox * jdec::decOutputDim + 1, sizeof(float), 8);
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
        forward_kernel<<<numBlocks, blockSize, 0, stream>>>(in[j], out, mJDecKernel[j].inWidth, mJDecKernel[j].inHeight,
            inSize, jdec::decOutputDim, jdec::maxNumOutputBox, mNumClass, (float*)mGpuAnchor[j], mConfThresh);
    }
    cudaStreamSynchronize(stream);
}

JDecoderPluginCreator::JDecoderPluginCreator()
{
    mPluginAttributes.clear();
    mPluginFieldCollection.nbFields = mPluginAttributes.size();
    mPluginFieldCollection.fields = mPluginAttributes.data();
}

const char* JDecoderPluginCreator::getPluginName() const
{
    return "JDecoderPlugin";
}

const char* JDecoderPluginCreator::getPluginVersion() const
{
    return "100";
}

const PluginFieldCollection* JDecoderPluginCreator::getFieldNames()
{
    return &mPluginFieldCollection;
}

IPluginV2IOExt* JDecoderPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    JDecoderPlugin* p = new JDecoderPlugin();
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

IPluginV2IOExt* JDecoderPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    JDecoderPlugin* p = new JDecoderPlugin(serialData, serialLength);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

void JDecoderPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mPluginNamespace = libNamespace;
}

const char* JDecoderPluginCreator::getPluginNamespace() const
{
    return mPluginNamespace.c_str();
}

}   // namespace nvinfer1