#ifndef LAPJV_H
#define LAPJV_H

#include <float.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "ndarrayobject.h"

namespace mot {

class LAPJV
{
public:
    LAPJV();
    ~LAPJV();
    bool solve(const float *cost, int rows, int cols, float *opt, int *x,
        int *y, bool extend_cost=false, float cost_limit=FLT_MAX);
private:
    bool inited = false;
    PyObject *module = 0;
    PyObject *dict = 0;
    PyObject *lapjv = 0;
};

}   // namespace mot

#endif  // LAPJV_H