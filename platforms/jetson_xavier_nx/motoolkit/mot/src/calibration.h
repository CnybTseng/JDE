#ifndef CALIBRATION_H_
#define CALIBRATION_H_

#include <vector>
#include <string>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include "utils.h"

namespace mot {

//! \brief class Int8EntropyCalibrator2
//!
class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    //! \brief Int8EntropyCalibrator2 constructor.
    //! \param imDir     : The calibration image directory.
    //! \param dim       : The dimension of batch images in NCHW order.
    //! \param inBlobName: Input blob name.
    //! \param calTabName: Calibration table filename.
    //! \param readCache : Read cache or not.
    //!
    Int8EntropyCalibrator2(const char *imDir, DimsX dim,
        const char *inBlobName, const char *calTabName, bool readCache=true);
    ~Int8EntropyCalibrator2();
    
    //! Functions below are derived from base class.
    int32_t getBatchSize() const override;
    bool getBatch (void *bindings[], const char *names[], int32_t nbBindings) override;
    const void *readCalibrationCache (std::size_t &length) override;
    void writeCalibrationCache(const void *ptr, std::size_t length) override;
private:
    std::string mImDir;
    DimsX mDim;
    std::string mInBlobName;
    std::string mCalTabName;
    bool mReadCache;
    int imIndex;
    std::vector<std::string> imNameList;
    void *devInput;
    std::vector<char> calCache;
    cv::Mat preprocessImage(cv::Mat &in);
};

} // namespace mot

#endif // CALIBRATION_H_