#ifndef BUFFER_H_
#define BUFFER_H_

#include <vector>
#include <cassert>
#include <numeric>
#include <iostream>

#include "logger.h"

#define CHECK(status)                                           \
    do                                                          \
    {                                                           \
        auto ret = (status);                                    \
        if (ret != 0)                                           \
        {                                                       \
            std::cout << "Cuda failure: " << ret << std::endl;  \
            abort();                                            \
        }                                                       \
    } while (0)

namespace mot {

template <typename A, typename B>
inline A div_up(A x, B n)
{
    return (x + n - 1) / n;
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
//!          size is the amount of memory in bytes to allocate.
//!          The boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer address. It must work with nullptr input.
//!
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
public:
    //!
    //! \brief Construct an empty buffer.
    //!
    GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
        : mSize(0)
        , mCapacity(0)
        , mType(type)
        , mBuffer(nullptr)
    {
    }

    //!
    //! \brief Construct a buffer with the specified allocation size in bytes.
    //!
    GenericBuffer(size_t size, nvinfer1::DataType type)
        : mSize(size)
        , mCapacity(size)
        , mType(type)
    {
        if (!allocFn(&mBuffer, this->nbBytes()))
        {
            throw std::bad_alloc();
        }
    }

    GenericBuffer(GenericBuffer&& buf)
        : mSize(buf.mSize)
        , mCapacity(buf.mCapacity)
        , mType(buf.mType)
        , mBuffer(buf.mBuffer)
    {
        buf.mSize = 0;
        buf.mCapacity = 0;
        buf.mType = nvinfer1::DataType::kFLOAT;
        buf.mBuffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf)
        {
            freeFn(mBuffer);
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            mBuffer = buf.mBuffer;
            // Reset buf.
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    void* data()
    {
        return mBuffer;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    const void* data() const
    {
        return mBuffer;
    }

    //!
    //! \brief Returns the size (in number of elements) of the buffer.
    //!
    size_t size() const
    {
        return mSize;
    }

    //!
    //! \brief Returns the size (in bytes) of the buffer.
    //!
    size_t nbBytes() const
    {
        return this->size() * getElementSize(mType);
    }

    //!
    //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
    //!
    void resize(size_t newSize)
    {
        mSize = newSize;
        if (mCapacity < newSize)
        {
            freeFn(mBuffer);
            if (!allocFn(&mBuffer, this->nbBytes()))
            {
                throw std::bad_alloc{};
            }
            mCapacity = newSize;
        }
    }

    //!
    //! \brief Overload of resize that accepts Dims
    //!
    void resize(const nvinfer1::Dims& dims)
    {
        return this->resize(volume(dims));
    }

    ~GenericBuffer()
    {
        freeFn(mBuffer);
    }

private:
    size_t mSize{0}, mCapacity{0};
    nvinfer1::DataType mType;
    void* mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

class DeviceAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        return cudaMalloc(ptr, size) == cudaSuccess;
    }
};

class DeviceFree
{
public:
    void operator()(void* ptr) const
    {
        cudaFree(ptr);
    }
};

class HostAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }
};

class HostFree
{
public:
    void operator()(void* ptr) const
    {
        free(ptr);
    }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

class ManagedBuffer
{
public:
    DeviceBuffer device_buffer;
    HostBuffer host_buffer;
};

class BufferManager
{
public:
    BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int batch_size=0,
        const nvinfer1::IExecutionContext* context=nullptr)
        : _engine(engine)
        , _batch_size(batch_size)
    {
        // assert(engine->hasImplicitBatchDimension() || _batch_size == 0);
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            auto dims = context ? context->getBindingDimensions(i) : _engine->getBindingDimensions(i);
            size_t vol = context || !_batch_size ? 1 : static_cast<size_t>(_batch_size);
            nvinfer1::DataType type = _engine->getBindingDataType(i);
            // 获取buffer向量化的维度索引
            int vec_dim = _engine->getBindingVectorizedDim(i);
            if (-1 != vec_dim) {
                // 获取向量中的元素个数
                int scalars_per_vec = _engine->getBindingComponentsPerElement(i);
                dims.d[vec_dim] = div_up(dims.d[vec_dim], scalars_per_vec);
                vol *= scalars_per_vec;
            }
            vol *= volume(dims);
            std::unique_ptr<ManagedBuffer> man_buffer{new ManagedBuffer()};
            man_buffer->device_buffer = DeviceBuffer(vol, type);
            man_buffer->host_buffer = HostBuffer(vol, type);
            _device_bindings.emplace_back(man_buffer->device_buffer.data());
            _managed_buffers.emplace_back(std::move(man_buffer));
        }
    }
    
    std::vector<void*>& get_device_bindings()
    {
        return _device_bindings;
    }
    
    const std::vector<void*>& get_device_bindings() const
    {
        return _device_bindings;
    }
    
    void copy_input_to_device()
    {
        memcpyBuffers(true, false, false);
    }
    
    void copy_output_to_host()
    {
        memcpyBuffers(false, true, false);
    }
private:
    void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream = 0)
    {
        for (int i = 0; i < _engine->getNbBindings(); i++)
        {
            void* dstPtr
                = deviceToHost ? _managed_buffers[i]->host_buffer.data() : _managed_buffers[i]->device_buffer.data();
            const void* srcPtr
                = deviceToHost ? _managed_buffers[i]->device_buffer.data() : _managed_buffers[i]->host_buffer.data();
            const size_t byteSize = _managed_buffers[i]->host_buffer.nbBytes();
            const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
            if ((copyInput && _engine->bindingIsInput(i)) || (!copyInput && !_engine->bindingIsInput(i)))
            {
                if (async)
                    CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
                else
                    CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
            }
        }
    }

    std::shared_ptr<nvinfer1::ICudaEngine> _engine;
    int _batch_size;
    std::vector<std::unique_ptr<ManagedBuffer>> _managed_buffers;
    std::vector<void*> _device_bindings;
};

}   // namespace mot

#endif