#ifndef NMS_H_
#define NMS_H_

#include <memory>

#define DETECTION_DIM   5
#define BLOCK_SIZE      64

namespace mot {

class NMS
{
public:
    //! \brief Get NMS instance.
    //!
    static NMS* instance();
    
    //! \brief Initialize NMS module.
    //! \param maxNumDet Maximum number of detections.
    //! \return True if initialize success, False otherwise.
    //!
    bool init(int maxNumDet);

    //! \brief Nonmaximum suppression with CUDA optimization.
    //! \param dets Detection array with [top,left,bottom,right,score] format elements.
    //!  dets should be stored in device memory.
    //! \brief numDet Number of detections.
    //! \param keeps The index array of detections need to be kept.
    //! \param numKeep The number of detections need to be kept.
    //! \param iouThresh Overlap threshold. The default value of iouThresh is 0.4.
    //!
    void nms(float* dets, int numDet, int* keeps, int* numKeep, float iouThresh=0.4f);
    
    //! \brief Destroy NMS module.
    //!
    void free();
private:
    static NMS* me;
    int mMaxNumDet;
    const int mBlockSize;
    std::shared_ptr<unsigned long long> mMaskCpu;
    std::shared_ptr<unsigned long long> mDiscard;
    unsigned long long* mMaskGpu;

    NMS() : mBlockSize(BLOCK_SIZE) {}
    ~NMS() {}
};

}   // namespace mot

#endif  // NMS_H_