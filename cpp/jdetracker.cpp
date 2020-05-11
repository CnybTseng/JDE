#include <algorithm>

#include "jdetracker.h"

namespace mot {

JDETracker::JDETracker()
{
    
}

JDETracker::~JDETracker()
{
    
}

static void cdist(float *XA, int ma, float *XB, int mb, int n, float **Y)
{
    for (int i = 0; i < ma; ++i)
    {
        float *yi = Y + i * ma;
        float *xa = XA + i * n;
        for (int j = 0; j < mb; ++j)
        {
            float dist = 0;
            float *xb = XB + j * n;
            for (int k = 0; k < n; ++k)
            {
                dist += (xa[k] - xb[k]) * (xa[k] - xb[k]);
            }
            *(yi + j) = sqrt(dist);
        }
    }
}

int JDETracker::update(float *dets, int dim, int cnt)
{
    return 0;
}

}   // namespace mot