#include <stdio.h>
#include <cuda_runtime_api.h>

#include "nms.h"

#define divide_up(m, n) ((m) / (n) + ((m) % (n) > 0))

namespace mot {

__device__ inline float iou(const float* const boxa, const float* const boxb)
{
    float miny = max(boxa[0], boxb[0]);
    float minx = max(boxa[1], boxb[1]);
    float maxy = min(boxa[2], boxb[2]);
    float maxx = min(boxa[3], boxb[3]);
    float w = max(maxx - minx, .0f);
    float h = max(maxy - miny, .0f);
    float inter = w * h;
    float areaa = (boxa[3] - boxa[1]) * (boxa[2] - boxa[0]);
    float areab = (boxb[3] - boxb[1]) * (boxb[2] - boxb[0]);
    return inter / (areaa + areab - inter);
}

__global__ void nms_kernel(const float* dets, const int numDet, unsigned long long* mask, float iouThresh)
{
    // These are IOU matrix block sizes instead of CUDA block sizes.
    const int iouBlockSizeX = min(numDet - blockIdx.x * blockDim.x, blockDim.x);
    const int iouBlockSizeY = min(numDet - blockIdx.y * blockDim.x, blockDim.x);
    
    // Shared memory store top, left, bottom, right, and score.
    __shared__ float blockDets[BLOCK_SIZE * DETECTION_DIM];

    if (threadIdx.x < iouBlockSizeX) {
        blockDets[DETECTION_DIM * threadIdx.x]     = dets[DETECTION_DIM * (blockIdx.x * blockDim.x + threadIdx.x)];
        blockDets[DETECTION_DIM * threadIdx.x + 1] = dets[DETECTION_DIM * (blockIdx.x * blockDim.x + threadIdx.x) + 1];
        blockDets[DETECTION_DIM * threadIdx.x + 2] = dets[DETECTION_DIM * (blockIdx.x * blockDim.x + threadIdx.x) + 2];
        blockDets[DETECTION_DIM * threadIdx.x + 3] = dets[DETECTION_DIM * (blockIdx.x * blockDim.x + threadIdx.x) + 3];
        blockDets[DETECTION_DIM * threadIdx.x + 4] = dets[DETECTION_DIM * (blockIdx.x * blockDim.x + threadIdx.x) + 4];
    }
    
    __syncthreads();
    
    if (threadIdx.x < iouBlockSizeY) {
        int start = 0;
        // Diagonal blocks only process up-triangle elements.
        if (blockIdx.x == blockIdx.y) {
            start = threadIdx.x + 1;
        }
        
        const int detIdx = blockIdx.y * blockDim.x + threadIdx.x;
        const float* det = dets + detIdx * DETECTION_DIM;
        unsigned long long m = 0;
        
        for (int i = start; i < iouBlockSizeX; ++i) {
            if (iou(det, blockDets + i * DETECTION_DIM) > iouThresh) {
                m |= (1ULL << i);
            }
        }
        
        const int numBlocksX = divide_up(numDet, blockDim.x);
        mask[detIdx * numBlocksX + blockIdx.x] = m;
    }
}

NMS* NMS::me = nullptr;

NMS* NMS::instance()
{
    if (!me) {
        me = new NMS();
    }
    return me;
}

bool NMS::init(int maxNumDet)
{
    mMaxNumDet = maxNumDet;    
    const int maxNumBlocksX = divide_up(maxNumDet, mBlockSize);
    
    mMaskCpu = std::shared_ptr<unsigned long long>(new unsigned long long[maxNumDet * maxNumBlocksX]);
    if (!mMaskCpu) {
        fprintf(stderr, "make shared pointer fail\n");
        return false;
    }
    
    mDiscard = std::shared_ptr<unsigned long long>(new unsigned long long[maxNumBlocksX]);
    if (!mDiscard) {
        fprintf(stderr, "make shared pointer fail\n");
        return false;
    }
    
    cudaError_t code = cudaMalloc((void**)&mMaskGpu, maxNumDet * maxNumBlocksX * sizeof(unsigned long long));
    if (cudaSuccess != code) {
        fprintf(stderr, "cudaMalloc fail: %s\n", cudaGetErrorString(code));
        return false;
    }
    
    return true;
}

//!
//! Nonmaximum suppression with CUDA optimization.
//!
void NMS::nms(float* dets, int numDet, int* keeps, int* numKeep, float iouThresh)
{
    int numBlocksX = divide_up(numDet, mBlockSize);
    dim3 numBlocks(numBlocksX, numBlocksX);
    dim3 blockSize(mBlockSize);
    
    nms_kernel<<<numBlocks, blockSize>>>(dets, numDet, mMaskGpu, iouThresh);
    
    cudaMemcpy(mMaskCpu.get(), mMaskGpu, numDet * numBlocksX * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost);
    memset(mDiscard.get(), 0, numBlocksX * sizeof(unsigned long long));
    
    int count = 0;
    for (int i = 0; i < numDet; ++i) {
        int blockID  = i / mBlockSize;
        int threadID = i % mBlockSize;
        if (!(mDiscard.get()[blockID] & (1ULL << threadID))) {
            keeps[count++] = i;
            // Following blocks (detections) with lower score.
            unsigned long long* pMask = mMaskCpu.get() + i * numBlocksX;
            for (int j = blockID; j < numBlocksX; ++j) {
                mDiscard.get()[j] |= pMask[j];
            }
        }
    }
    
    *numKeep = count;
}

void NMS::free()
{
    if (mMaskGpu) {
        cudaFree(mMaskGpu);
    }
}

}   // namespace mot