#ifndef JDECODERV2_H_
#define JDECODERV2_H_

#include <vector>
#include <string>

#include <NvInfer.h>

namespace nvinfer1 {

namespace jdec {

static constexpr int numClass        = 2;       //! including background
static constexpr int numAnchor       = 4;       //! number of anchors per head
static constexpr int anchorDim       = 2;       //! width, height
static constexpr int netInWidth      = 576;     //! network input width
static constexpr int netInHeight     = 320;     //! network input height
static constexpr int boxDim          = 4;       //! left, top, right, bottom
static constexpr int classDim        = 2;       //! class identifier, class score
static constexpr int embeDim         = 128;     //! embedding vector dimension
static constexpr int maxNumOutputBox = 1024;    //! maximum number of output boxes per head

struct JDecKernel
{
    int32_t inWidth;
    int32_t inHeight;
    float   anchor[numAnchor * anchorDim];
};

static constexpr JDecKernel jdk1 = {
    netInWidth  >> 5,
    netInHeight >> 5,
    {85, 255, 120, 360, 170, 420, 340, 320}
};

static constexpr JDecKernel jdk2 = {
    netInWidth  >> 4,
    netInHeight >> 4,
    {21, 64, 30, 90, 43, 128, 60, 180}
};

static constexpr JDecKernel jdk3 = {
    netInWidth  >> 3,
    netInHeight >> 3,
    {6, 16, 8, 23, 11, 32, 16, 45}
};

//! JDE decoder input dimension
static constexpr int decInputDim = numAnchor * (boxDim + numClass) + embeDim;

//! JDE decoder output dimension
static constexpr int decOutputDim = boxDim + classDim + embeDim;

}   // namespace jdec

class JDecoderPlugin : public IPluginV2IOExt
{
public:
    JDecoderPlugin();
    JDecoderPlugin(const void* data, size_t size);
    ~JDecoderPlugin();
    
    //! Inherit from IPluginV2
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int32_t getNbOutputs() const override;
    Dims getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims) override;
    int32_t initialize() override;
    void terminate() override;
    size_t getWorkspaceSize(int32_t maxBatchSize) const override;
    int32_t enqueue(int32_t batchSize, const void* const* inputs, void** outputs, void* workspace,
        cudaStream_t stream) override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void destroy() override;
    void setPluginNamespace(const char* pluginNamespace) override;
    const char* getPluginNamespace() const override;
    
    //! Inherit from IPluginV2Ext
    nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes,
        int32_t nbInputs) const override;
    bool isOutputBroadcastAcrossBatch(int32_t outputIndex, const bool* inputIsBroadcasted,
        int32_t nbInputs) const override;
    bool canBroadcastInputAcrossBatch(int32_t inputIndex) const override;
    void attachToContext(cudnnContext*, cublasContext*, IGpuAllocator*) override;
    void detachFromContext() override;
    IPluginV2Ext* clone() const override;
    
    //! Inherit from IPluginV2IOExt
    void configurePlugin(const PluginTensorDesc* in, int32_t nbInput, const PluginTensorDesc* out,
        int32_t nbOutput) override;
    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs,
        int32_t nbOutputs) const override;
    
    void forward_test(const float* const* in, float* out, cudaStream_t stream, int batchSize=1);
private:
    //! \brief JDE output decoder forward.
    //!
    //! \param in Decoder inputs, including 3 tensors with dimensions 1x10x18x152,
    //!  1x20x36x152, and 1x40x72x152 respectively. And the memory layout of input
    //!  tensors is NHWC.
    //!
    //! \param out Decoder output. The output is a N by 134 matrix, and N is the
    //!  maximum number of decoding outputs. By default, N is 1024.
    //!
    //! \param stream
    //!
    //! \param batchSize
    //!
    void forward(const float* const* in, float* out, cudaStream_t stream, int batchSize=1);
    
    //! These parameters need to be serialized.
    int32_t mNumThread;
    int32_t mNumClass;
    int32_t mNumKernel;
    float   mConfThresh;
    std::vector<jdec::JDecKernel> mJDecKernel;
    
    //!  These parameters will not be serialized.
    std::vector<void*> mGpuAnchor;
    std::string mPluginNamespace;
};

class JDecoderPluginCreator : public IPluginCreator
{
public:
    JDecoderPluginCreator();
    ~JDecoderPluginCreator() {}
    const char* getPluginName() const override;
    const char* getPluginVersion() const override;
    const PluginFieldCollection* getFieldNames() override;
    IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;
    IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
    void setPluginNamespace(const char* libNamespace) override;
    const char* getPluginNamespace() const override;
private:
    std::string mPluginNamespace;
    std::vector<PluginField> mPluginAttributes;
    PluginFieldCollection mPluginFieldCollection;
};

REGISTER_TENSORRT_PLUGIN(JDecoderPluginCreator);

}   // namespace nvinfer1

#endif