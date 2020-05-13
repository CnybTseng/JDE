#ifndef LAPJV_H
#define LAPJV_H

#include <float.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"

namespace mot {

class LAPJV
{
private:
    static LAPJV *me;
public:
    static LAPJV *instance(void) {
        if (!me) me = new LAPJV();
        return me;
    };
    bool init();
    bool solve(const float *cost, int rows, int cols, float *opt, int *x,
        int *y, bool extend_cost=false, float cost_limit=FLT_MAX);
    void free();
private:
    PyObject *module = 0;
    PyObject *dict = 0;
    PyObject *lapjv = 0;
    LAPJV(void) {};
    ~LAPJV(void) {};
};

}   // namespace mot

#endif  // LAPJV_H