#ifndef UTILS_H_
#define UTILS_H_

#include <map>
#include <vector>
#include <iomanip>
#include <ostream>
#include <algorithm>

#include <NvInfer.h>

#define PROFILE 1
#if PROFILE
#define PROFILE_JDE 0
#define PROFILE_DECODER 0
#define PROFILE_TRACKER 0
#else
#define PROFILE_JDE 0
#define PROFILE_DECODER 0
#define PROFILE_TRACKER 0    
#endif

#define USE_DECODERV2 1             //! Use GPU-optimized decoder
#if USE_DECODERV2
#define INTEGRATES_DECODER 1        //! And integrates decoder into engine
#else
#define INTEGRATES_DECODER 0    
#endif

#define SAFETY_FREE(mem)    \
do {                        \
    if (mem) {              \
        free(mem);          \
        mem = nullptr;      \
    }                       \
} while (0)

#define numel_after_align(n, elsize, align) \
    ((((n) * (elsize) + (align) - 1) / (align)) * (align) / (elsize))

namespace mot {

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) {
        if (obj) {
            obj->destroy();
        }
    }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, InferDeleter>;

struct SimpleProfiler : public nvinfer1::IProfiler
{
    struct Record
    {
        float time{0};
        int count{0};
    };

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        mProfile[layerName].count++;
        mProfile[layerName].time += ms;
        if (std::find(mLayerNames.begin(), mLayerNames.end(), layerName) == mLayerNames.end())
        {
            mLayerNames.push_back(layerName);
        }
    }

    SimpleProfiler(const char* name, const std::vector<SimpleProfiler>& srcProfilers = std::vector<SimpleProfiler>())
        : mName(name)
    {
        for (const auto& srcProfiler : srcProfilers)
        {
            for (const auto& rec : srcProfiler.mProfile)
            {
                auto it = mProfile.find(rec.first);
                if (it == mProfile.end())
                {
                    mProfile.insert(rec);
                }
                else
                {
                    it->second.time += rec.second.time;
                    it->second.count += rec.second.count;
                }
            }
        }
    }

    friend std::ostream& operator<<(std::ostream& out, const SimpleProfiler& value)
    {
        out << "========== " << value.mName << " profile ==========" << std::endl;
        float totalTime = 0;
        std::string layerNameStr = "TensorRT layer name";
        int maxLayerNameLength = std::max(static_cast<int>(layerNameStr.size()), 70);
        for (const auto& elem : value.mProfile)
        {
            totalTime += elem.second.time;
            maxLayerNameLength = std::max(maxLayerNameLength, static_cast<int>(elem.first.size()));
        }

        auto old_settings = out.flags();
        auto old_precision = out.precision();
        // Output header
        {
            out << std::setw(maxLayerNameLength) << layerNameStr << " ";
            out << std::setw(12) << "Runtime, "
                << "%"
                << " ";
            out << std::setw(12) << "Invocations"
                << " ";
            out << std::setw(12) << "Runtime, ms" << std::endl;
        }
        for (size_t i = 0; i < value.mLayerNames.size(); i++)
        {
            const std::string layerName = value.mLayerNames[i];
            auto elem = value.mProfile.at(layerName);
            out << std::setw(maxLayerNameLength) << layerName << " ";
            out << std::setw(12) << std::fixed << std::setprecision(1) << (elem.time * 100.0F / totalTime) << "%"
                << " ";
            out << std::setw(12) << elem.count << " ";
            out << std::setw(12) << std::fixed << std::setprecision(2) << elem.time << std::endl;
        }
        out.flags(old_settings);
        out.precision(old_precision);
        out << "========== " << value.mName << " total runtime = " << totalTime << " ms ==========" << std::endl;

        return out;
    }

private:
    std::string mName;
    std::vector<std::string> mLayerNames;
    std::map<std::string, Record> mProfile;
};

extern SimpleProfiler profiler;

// 兆字节转字节
constexpr long long int operator""_MiB(long long unsigned int val)
{
    return val * (1 << 20);
}

// 重载cout打印nvinfer1::Dims型变量
std::ostream& operator<<(std::ostream& os, nvinfer1::Dims dims);

}   // namespace mot

#endif