#include <istream>
#include <ostream>
#include <fstream>
#include <iterator>
#include <cuda_runtime_api.h>
#include "calibration.h"

namespace mot {

Int8EntropyCalibrator2::Int8EntropyCalibrator2(const char *imDir, DimsX dim,
    const char *inBlobName, const char *calTabName, bool readCache) :
    mImDir(imDir),
    mDim(dim),
    mInBlobName(inBlobName),
    mCalTabName(calTabName),
    mReadCache(readCache),
    imIndex(0)
{
    cudaMalloc(&devInput, mDim.numel() * sizeof(float));
    read_file_list(imDir, imNameList);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{
    cudaFree(devInput);
}

int32_t Int8EntropyCalibrator2::getBatchSize() const
{
    return mDim.d[0];
}

bool Int8EntropyCalibrator2::getBatch (void *bindings[], const char *names[], int32_t nbBindings)
{
    if (imIndex + mDim.d[0] > static_cast<int>(imNameList.size())) {
        return false;
    }
    
    std::vector<cv::Mat> images(mDim.d[0]);
    for (int i = imIndex; i < imIndex + mDim.d[0]; ++i) {
        std::cout << "imread(" << mImDir + imNameList[i] << ")..." << std::endl;
        cv::Mat im = cv::imread(mImDir + imNameList[i]);
        if (im.empty()) {
            std::cerr << "imread(" << imNameList[i] << ") fail" << std::endl;
            return false;
        }
        images[i - imIndex] = preprocessImage(im);
        // cv::imwrite(imNameList[i], images[i - imIndex]);
    }
    
    imIndex += mDim.d[0];
    const double scalefactor = 1.0;
    const cv::Size size(mDim.d[3], mDim.d[2]);
    const cv::Scalar mean(0, 0, 0);
    const bool swapRB = false;
    const bool crop = false;
    cv::Mat blob = cv::dnn::blobFromImages(images, scalefactor, size, mean, swapRB, crop);
    
    cudaMemcpy(devInput, blob.ptr<float>(0), mDim.numel() * sizeof(float), cudaMemcpyHostToDevice);
    assert(!strcmp(names[0], mInBlobName.c_str()));
    bindings[0] = devInput;
    return true;
}

const void *Int8EntropyCalibrator2::readCalibrationCache (std::size_t &length)
{
    std::cout << "readCalibrationCache: " << mCalTabName << std::endl;
    calCache.clear();
    std::ifstream ifs(mCalTabName, std::ios::binary);
    ifs >> std::noskipws;
    if (mReadCache && ifs.good()) {
        std::copy(std::istream_iterator<char>(ifs), std::istream_iterator<char>(), std::back_inserter(calCache));
    }
    length = calCache.size();
    return length ? calCache.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void *ptr, std::size_t length)
{
    std::cout << "writeCalibrationCache: " << mCalTabName << ", " << length << std::endl;
    std::ofstream ofs(mCalTabName, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(ptr), length);
}

cv::Mat Int8EntropyCalibrator2::preprocessImage(cv::Mat &in)
{
    int vw = 0; //! valid width.
    int vh = 0; //! valid height.
    float sx = in.cols / static_cast<float>(mDim.d[3]);
    float sy = in.rows / static_cast<float>(mDim.d[2]);
    if (sx > sy) {  // The input image is fatter.
        vw = mDim.d[3];
        vh = static_cast<int>(in.rows / sx);
    } else {
        vh = mDim.d[2];
        vw = static_cast<int>(in.cols / sy);
    }
    
    int dx = (mDim.d[3] - vw) >> 1;
    int dy = (mDim.d[2] - vh) >> 1;
    cv::Mat out(mDim.d[2], mDim.d[3], CV_8UC3, cv::Scalar(128, 128, 128));
    cv::resize(in, out(cv::Rect(dx, dy, vw, vh)), cv::Size(vw, vh), 0, 0, cv::INTER_AREA);
    return out;
}

} // namespace mot