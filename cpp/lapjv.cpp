#include <stdio.h>
#include <stdlib.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "ndarrayobject.h"

#include "lapjv.h"
#include "jdeutils.h"

namespace mot {

LAPJV* LAPJV::me = 0;

bool LAPJV::init(void)
{
    Py_Initialize();
    check_error_goto(!Py_IsInitialized(), fail, "Py_Initialize fail!\n");
    
    _import_array();
    module = PyImport_ImportModule("lap");
    check_error_goto(!module, fail, "PyImport_Import fail!\n");
    
    dict = PyModule_GetDict(module);
    check_error_goto(!dict, fail, "PyModule_GetDict fail!\n");
    
    lapjv = PyDict_GetItemString(dict, "lapjv");
    check_error_goto(!lapjv || !PyCallable_Check(lapjv), fail, "PyDict_GetItemString fail!\n");
    return true;
    
    fail:    
    free();
    return false;
}

#define lapjv_solve_clr_ret(ret)    \
do {                                \
    Py_CLEAR(args);                 \
    Py_CLEAR(rest);                 \
    return ret;                     \
} while (0)

bool LAPJV::solve(const float *cost, int rows, int cols, float *opt, int *x,
        int *y, bool extend_cost, float cost_limit)
{
    npy_intp dim[2] = {rows, cols};
    PyObject *args = 0;             // new reference
    PyObject *arg1 = 0;             // reference is stealed by args
    PyObject *arg2 = 0;             // reference is stealed by args
    PyObject *arg3 = 0;             // reference is stealed by args
    PyObject *rest = 0;             // new reference
    PyObject *robj = 0;             // borrow reference from rest
    PyArrayObject *rarr = 0;        // borrow reference from rest
    int nx = 0;
    int ny = 0;
    
    arg1 = PyArray_SimpleNewFromData(2, dim, NPY_FLOAT, const_cast<float *>(cost));
    check_error_ret(!arg1, false, "PyArray_SimpleNewFromData fail!\n");
        
    arg2 = extend_cost ? Py_True : Py_False;
        
    arg3 = Py_BuildValue("f", cost_limit);
    check_error_ret(!arg3, false, "Py_BuildValue fail!\n");
    
    args = Py_BuildValue("OOO", arg1, arg2, arg3);
    check_error_ret(!args, false, "PyTuple_New fail!\n");
    
    rest = PyObject_CallObject(lapjv, args);
    check_error_goto(!rest, fail, "PyObject_CallObject fail!\n");
    
    robj = PyTuple_GetItem(rest, 0);
    check_error_goto(!robj, fail, "PyTuple_GetItem 0 fail!\n");
    *opt = static_cast<float>(PyFloat_AsDouble(robj));
    
    rarr = (PyArrayObject *)PyTuple_GetItem(rest, 1);    
    check_error_goto(!rarr, fail, "PyTuple_GetItem 1 fail!\n");
    nx = PyArray_DIM(rarr, 0);
    memcpy(x, (int *)PyArray_DATA(rarr), nx * sizeof(int));
    
    rarr = (PyArrayObject *)PyTuple_GetItem(rest, 2);
    check_error_goto(!rarr, fail, "PyTuple_GetItem 2 fail!\n");
    ny = PyArray_DIM(rarr, 0);
    memcpy(y, (int *)PyArray_DATA(rarr), ny * sizeof(int));
    lapjv_solve_clr_ret(true);
    
    fail:
    lapjv_solve_clr_ret(false);
}

#undef lapjv_solve_clr_ret

void LAPJV::free(void)
{
    Py_CLEAR(lapjv);
    Py_CLEAR(dict);
    Py_CLEAR(module);
    Py_Finalize();
}

}   // namespace mot