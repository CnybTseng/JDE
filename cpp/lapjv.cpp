#include <stdio.h>
#include <stdlib.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "ndarrayobject.h"

#include "lapjv.h"
#include "jdeutils.h"

namespace mot {

LAPJV::LAPJV()
{    
    Py_Initialize();
    check_error_goto(!Py_IsInitialized(), fail, "Py_Initialize fail!\n");
    
    module = PyImport_ImportModule("lap");
    check_error_goto(!module, fail, "PyImport_Import fail!\n");
    
    dict = PyModule_GetDict(module);
    check_error_goto(!dict, fail, "PyModule_GetDict fail!\n");
    
    lapjv = PyDict_GetItemString(dict, "lapjv");
    check_error_goto(!lapjv || !PyCallable_Check(lapjv), fail, "PyDict_GetItemString fail!\n");
    
    fail:
    inited = false;
}

LAPJV::~LAPJV()
{
    Py_Finalize();
}

bool LAPJV::solve(const float *cost, int rows, int cols, float *opt, int *x,
        int *y, bool extend_cost, float cost_limit)
{
    npy_intp dim[2] = {rows, cols};
    PyObject *costt = 0;
    PyObject *args = 0;
    int ret = 0;
    PyObject *extend_costt = extend_cost ? Py_True : Py_False;
    PyObject *result = 0;
            
    costt = PyArray_SimpleNewFromData(2, dim, NPY_FLOAT, const_cast<float *>(cost));
    check_error_goto(!costt, fail, "PyArray_SimpleNewFromData fail!\n");
    
    args = PyTuple_New(3);
    check_error_goto(!args, fail, "PyTuple_New fail!\n");
    
    ret = PyTuple_SetItem(args, 0, costt);
    check_error_goto(ret, fail, "PyTuple_SetItem fail!\n");
    
    ret = PyTuple_SetItem(args, 1, extend_costt);
    check_error_goto(ret, fail, "PyTuple_SetItem fail!\n");
    
    ret = PyTuple_SetItem(args, 2, Py_BuildValue("f", cost_limit));
    check_error_goto(ret, fail, "PyTuple_SetItem fail!\n");
    
    result = PyObject_CallObject(lapjv, args);
    check_error_goto(!result, fail, "PyObject_CallObject fail!\n");
    
    return true;
    fail: return false;
}

}   // namespace mot

// https://blog.csdn.net/foreverhehe716/article/details/82841567
// https://stackoverflow.com/questions/44122750/py-initialize-undefined-reference
// https://www.coder.work/article/1231533

int main(int argc, char *argv[])
{
    Py_Initialize();
    if (!Py_IsInitialized())
    {
        fprintf(stderr, "Py_Initialize fail!\n");
        return -1;
    }
    
    // PyRun_SimpleString("from lap import lapjv");
    // PyRun_SimpleString("help(lapjv)");

    PyObject *module = PyImport_ImportModule("lap");
    if (!module)
    {
        fprintf(stderr, "PyImport_Import fail!\n");
        return -1;
    }
    
    PyObject *dict = PyModule_GetDict(module);
    if (!dict)
    {
        fprintf(stderr, "PyModule_GetDict fail!\n");
        return -1;
    }
    
    PyObject *lapjv = PyDict_GetItemString(dict, "lapjv");
    if (!lapjv || !PyCallable_Check(lapjv))
    {
        fprintf(stderr, "PyDict_GetItemString fail!\n");
        return -1;
    }
    
    _import_array();
    float costt[12] = {1, 5, 6, 2, 4, 5, 8, 0, 1, 1, 6, 5};
    
    npy_intp dim[2] = {3, 4};
    PyObject *cost  = PyArray_SimpleNewFromData(2, dim, NPY_FLOAT, costt);
    if (!cost)
    {
        fprintf(stderr, "PyArray_SimpleNewFromData fail!\n");
        return -1;
    }
    
    PyObject *args = PyTuple_New(3);
    if (!args)
    {
        fprintf(stderr, "PyTuple_New fail!\n");
        return -1;
    }
    
    if (PyTuple_SetItem(args, 0, cost))
    {
        fprintf(stderr, "PyTuple_SetItem fail 0!\n");
        return -1;
    }
    
    PyObject *extend_cost = Py_True;
    if (PyTuple_SetItem(args, 1, extend_cost))
    {
        fprintf(stderr, "PyTuple_SetItem fail 1!\n");
        return -1;
    }
    
    float cost_limit = 0.7f;
    if (PyTuple_SetItem(args, 2, Py_BuildValue("f", cost_limit)))
    {
        fprintf(stderr, "PyTuple_SetItem fail 2!\n");
        return -1;
    }
    
    PyObject *ret = PyObject_CallObject(lapjv, args);
    if (!ret)
    {
        fprintf(stderr, "PyObject_CallObject fail!\n");
        return -1;
    }

    int size = PyTuple_Size(ret);
    fprintf(stderr, "lapjv return size %d\n", size);
    
    PyObject *opt_ = PyTuple_GetItem(ret, 0);
    double opt = PyFloat_AsDouble(opt_);
    fprintf(stderr, "opt is %lf\n", opt);
    
    PyArrayObject *x = (PyArrayObject *)PyTuple_GetItem(ret, 1);
    int *px = (int *)PyArray_DATA(x);
    int nx = PyArray_DIM(x, 0);
    fprintf(stderr, "x is ");
    for (int i = 0; i < nx; ++i)
    {
        fprintf(stderr, "%d ", px[i]);
    }
    
    fprintf(stderr, "\n");
    PyArrayObject *y = (PyArrayObject *)PyTuple_GetItem(ret, 2);
    int *py = (int *)PyArray_DATA(y);
    int ny = PyArray_DIM(y, 0);
    fprintf(stderr, "y is ");
    for (int i = 0; i < ny; ++i)
    {
        fprintf(stderr, "%d ", py[i]);
    }
    
    fprintf(stderr, "\n");
    
    Py_Finalize();
    
    return 0;
}