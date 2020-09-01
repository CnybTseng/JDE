#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstdlib>
#include <algorithm>

#ifdef PYTHON_LAPJV

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "ndarrayobject.h"

#else   // PYTHON_LAPJV

#define LARGE 1000000

#if !defined TRUE
#define TRUE 1
#endif
#if !defined FALSE
#define FALSE 0
#endif

#define NEW(x, t, n) if ((x = (t *)malloc(sizeof(t) * (n))) == 0) { return -1; }
#define FREE(x) if (x != 0) { free(x); x = 0; }
#define SWAP_INDICES(a, b) { int_t _temp_index = a; a = b; b = _temp_index; }

#define ASSERT(cond)
#define PRINTF(fmt, ...)
#define PRINT_COST_ARRAY(a, n)
#define PRINT_INDEX_ARRAY(a, n)

#define matrix_set_roi(data, x1, y1, x2, y2, val)   \
do {                                                \
    for (size_t i = y1; i < y2; ++i) {              \
        for (size_t j = x1; j < x2; ++j) {          \
            data[i][j] = val;                       \
        }                                           \
    }                                               \
} while (0)

typedef signed int int_t;
typedef unsigned int uint_t;
typedef double cost_t;
typedef char boolean;
typedef enum fp_t { FP_1 = 1, FP_2 = 2, FP_DYNAMIC = 3 } fp_t;

#endif  // PYTHON_LAPJV

#include "lapjv.h"

namespace mot {

LAPJV* LAPJV::me = 0;

LAPJV *LAPJV::instance(void)
{
    if (!me)
        me = new LAPJV();
    return me;
}

#ifdef PYTHON_LAPJV

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

#else   // PYTHON_LAPJV

/** Column-reduction and reduction transfer for a dense cost matrix.
 */
static int_t _ccrrt_dense(const uint_t n, cost_t *cost[],
                     int_t *free_rows, int_t *x, int_t *y, cost_t *v)
{
    int_t n_free_rows;
    boolean *unique;

    for (uint_t i = 0; i < n; i++) {
        x[i] = -1;
        v[i] = LARGE;
        y[i] = 0;
    }
    for (uint_t i = 0; i < n; i++) {
        for (uint_t j = 0; j < n; j++) {
            const cost_t c = cost[i][j];
            if (c < v[j]) {
                v[j] = c;
                y[j] = i;
            }
            PRINTF("i=%d, j=%d, c[i,j]=%f, v[j]=%f y[j]=%d\n", i, j, c, v[j], y[j]);
        }
    }
    PRINT_COST_ARRAY(v, n);
    PRINT_INDEX_ARRAY(y, n);
    NEW(unique, boolean, n);
    memset(unique, TRUE, n);
    {
        int_t j = n;
        do {
            j--;
            const int_t i = y[j];
            if (x[i] < 0) {
                x[i] = j;
            } else {
                unique[i] = FALSE;
                y[j] = -1;
            }
        } while (j > 0);
    }
    n_free_rows = 0;
    for (uint_t i = 0; i < n; i++) {
        if (x[i] < 0) {
            free_rows[n_free_rows++] = i;
        } else if (unique[i]) {
            const int_t j = x[i];
            cost_t min = LARGE;
            for (uint_t j2 = 0; j2 < n; j2++) {
                if (j2 == (uint_t)j) {
                    continue;
                }
                const cost_t c = cost[i][j2] - v[j2];
                if (c < min) {
                    min = c;
                }
            }
            PRINTF("v[%d] = %f - %f\n", j, v[j], min);
            v[j] -= min;
        }
    }
    FREE(unique);
    return n_free_rows;
}


/** Augmenting row reduction for a dense cost matrix.
 */
static int_t _carr_dense(
    const uint_t n, cost_t *cost[],
    const uint_t n_free_rows,
    int_t *free_rows, int_t *x, int_t *y, cost_t *v)
{
    uint_t current = 0;
    int_t new_free_rows = 0;
    uint_t rr_cnt = 0;
    PRINT_INDEX_ARRAY(x, n);
    PRINT_INDEX_ARRAY(y, n);
    PRINT_COST_ARRAY(v, n);
    PRINT_INDEX_ARRAY(free_rows, n_free_rows);
    while (current < n_free_rows) {
        int_t i0;
        int_t j1, j2;
        cost_t v1, v2, v1_new;
        boolean v1_lowers;

        rr_cnt++;
        PRINTF("current = %d rr_cnt = %d\n", current, rr_cnt);
        const int_t free_i = free_rows[current++];
        j1 = 0;
        v1 = cost[free_i][0] - v[0];
        j2 = -1;
        v2 = LARGE;
        for (uint_t j = 1; j < n; j++) {
            PRINTF("%d = %f %d = %f\n", j1, v1, j2, v2);
            const cost_t c = cost[free_i][j] - v[j];
            if (c < v2) {
                if (c >= v1) {
                    v2 = c;
                    j2 = j;
                } else {
                    v2 = v1;
                    v1 = c;
                    j2 = j1;
                    j1 = j;
                }
            }
        }
        i0 = y[j1];
        v1_new = v[j1] - (v2 - v1);
        v1_lowers = v1_new < v[j1];
        PRINTF("%d %d 1=%d,%f 2=%d,%f v1'=%f(%d,%g) \n", free_i, i0, j1, v1, j2, v2, v1_new, v1_lowers, v[j1] - v1_new);
        if (rr_cnt < current * n) {
            if (v1_lowers) {
                v[j1] = v1_new;
            } else if (i0 >= 0 && j2 >= 0) {
                j1 = j2;
                i0 = y[j2];
            }
            if (i0 >= 0) {
                if (v1_lowers) {
                    free_rows[--current] = i0;
                } else {
                    free_rows[new_free_rows++] = i0;
                }
            }
        } else {
            PRINTF("rr_cnt=%d >= %d (current=%d * n=%d)\n", rr_cnt, current * n, current, n);
            if (i0 >= 0) {
                free_rows[new_free_rows++] = i0;
            }
        }
        x[free_i] = j1;
        y[j1] = free_i;
    }
    return new_free_rows;
}


/** Find columns with minimum d[j] and put them on the SCAN list.
 */
static uint_t _find_dense(const uint_t n, uint_t lo, cost_t *d, int_t *cols, int_t *y)
{
    uint_t hi = lo + 1;
    cost_t mind = d[cols[lo]];
    for (uint_t k = hi; k < n; k++) {
        int_t j = cols[k];
        if (d[j] <= mind) {
            if (d[j] < mind) {
                hi = lo;
                mind = d[j];
            }
            cols[k] = cols[hi];
            cols[hi++] = j;
        }
    }
    return hi;
}


// Scan all columns in TODO starting from arbitrary column in SCAN
// and try to decrease d of the TODO columns using the SCAN column.
static int_t _scan_dense(const uint_t n, cost_t *cost[],
                    uint_t *plo, uint_t*phi,
                    cost_t *d, int_t *cols, int_t *pred,
                    int_t *y, cost_t *v)
{
    uint_t lo = *plo;
    uint_t hi = *phi;
    cost_t h, cred_ij;

    while (lo != hi) {
        int_t j = cols[lo++];
        const int_t i = y[j];
        const cost_t mind = d[j];
        h = cost[i][j] - v[j] - mind;
        PRINTF("i=%d j=%d h=%f\n", i, j, h);
        // For all columns in TODO
        for (uint_t k = hi; k < n; k++) {
            j = cols[k];
            cred_ij = cost[i][j] - v[j] - h;
            if (cred_ij < d[j]) {
                d[j] = cred_ij;
                pred[j] = i;
                if (cred_ij == mind) {
                    if (y[j] < 0) {
                        return j;
                    }
                    cols[k] = cols[hi];
                    cols[hi++] = j;
                }
            }
        }
    }
    *plo = lo;
    *phi = hi;
    return -1;
}


/** Single iteration of modified Dijkstra shortest path algorithm as explained in the JV paper.
 *
 * This is a dense matrix version.
 *
 * \return The closest free column index.
 */
static int_t find_path_dense(
    const uint_t n, cost_t *cost[],
    const int_t start_i,
    int_t *y, cost_t *v,
    int_t *pred)
{
    uint_t lo = 0, hi = 0;
    int_t final_j = -1;
    uint_t n_ready = 0;
    int_t *cols;
    cost_t *d;

    NEW(cols, int_t, n);
    NEW(d, cost_t, n);

    for (uint_t i = 0; i < n; i++) {
        cols[i] = i;
        pred[i] = start_i;
        d[i] = cost[start_i][i] - v[i];
    }
    PRINT_COST_ARRAY(d, n);
    while (final_j == -1) {
        // No columns left on the SCAN list.
        if (lo == hi) {
            PRINTF("%d..%d -> find\n", lo, hi);
            n_ready = lo;
            hi = _find_dense(n, lo, d, cols, y);
            PRINTF("check %d..%d\n", lo, hi);
            PRINT_INDEX_ARRAY(cols, n);
            for (uint_t k = lo; k < hi; k++) {
                const int_t j = cols[k];
                if (y[j] < 0) {
                    final_j = j;
                }
            }
        }
        if (final_j == -1) {
            PRINTF("%d..%d -> scan\n", lo, hi);
            final_j = _scan_dense(
                    n, cost, &lo, &hi, d, cols, pred, y, v);
            PRINT_COST_ARRAY(d, n);
            PRINT_INDEX_ARRAY(cols, n);
            PRINT_INDEX_ARRAY(pred, n);
        }
    }

    PRINTF("found final_j=%d\n", final_j);
    PRINT_INDEX_ARRAY(cols, n);
    {
        const cost_t mind = d[cols[lo]];
        for (uint_t k = 0; k < n_ready; k++) {
            const int_t j = cols[k];
            v[j] += d[j] - mind;
        }
    }

    FREE(cols);
    FREE(d);

    return final_j;
}


/** Augment for a dense cost matrix.
 */
static int_t _ca_dense(
    const uint_t n, cost_t *cost[],
    const uint_t n_free_rows,
    int_t *free_rows, int_t *x, int_t *y, cost_t *v)
{
    int_t *pred;

    NEW(pred, int_t, n);

    for (int_t *pfree_i = free_rows; pfree_i < free_rows + n_free_rows; pfree_i++) {
        int_t i = -1, j;
        uint_t k = 0;

        PRINTF("looking at free_i=%d\n", *pfree_i);
        j = find_path_dense(n, cost, *pfree_i, y, v, pred);
        ASSERT(j >= 0);
        ASSERT(j < n);
        while (i != *pfree_i) {
            PRINTF("augment %d\n", j);
            PRINT_INDEX_ARRAY(pred, n);
            i = pred[j];
            PRINTF("y[%d]=%d -> %d\n", j, y[j], i);
            y[j] = i;
            PRINT_INDEX_ARRAY(x, n);
            SWAP_INDICES(j, x[i]);
            k++;
            if (k >= n) {
                ASSERT(FALSE);
            }
        }
    }
    FREE(pred);
    return 0;
}


/** Solve dense sparse LAP.
 */
static int lapjv_internal(
    const uint_t n, cost_t *cost[],
    int_t *x, int_t *y)
{
    int ret;
    int_t *free_rows;
    cost_t *v;

    NEW(free_rows, int_t, n);
    NEW(v, cost_t, n);
    ret = _ccrrt_dense(n, cost, free_rows, x, y, v);
    int i = 0;
    while (ret > 0 && i < 2) {
        ret = _carr_dense(n, cost, ret, free_rows, x, y, v);
        i++;
    }
    if (ret > 0) {
        ret = _ca_dense(n, cost, ret, free_rows, x, y, v);
    }
    FREE(v);
    FREE(free_rows);
    return ret;
}

bool LAPJV::init(void)
{
    return true;
}

bool LAPJV::solve(const float *cost, int rows, int cols, float *opt, int *x,
    int *y, bool extend_cost, float cost_limit)
{
    int dim = 0;
    double **extd_cost = NULL;
    int size = std::max(rows, cols);
    if (rows != cols) {
        if (false == extend_cost) {
            fprintf(stderr, "warning: set extend_cost as true!\n");
            return false;
        }
        
        float maxele = -999999;
        for (size_t i = 0; i < rows * cols; ++i) {
            if (cost[i] > maxele) {
                maxele = cost[i];
            }
        }
        
        if (cost_limit < FLT_MAX) {
            dim = size << 1;
            extd_cost = (double **)calloc(dim, sizeof(double *));
            for (size_t i = 0; i < dim; ++i) {
                extd_cost[i] = (double *)calloc(dim, sizeof(double));
            }
            
            if (size == rows) {
                matrix_set_roi(extd_cost, cols, 0, size, size, maxele + cost_limit + 1);
            } else {
                matrix_set_roi(extd_cost, 0, rows, size, size, maxele + cost_limit + 1);
            }
            
            matrix_set_roi(extd_cost, size, 0, dim, size, cost_limit);
            matrix_set_roi(extd_cost, 0, size, dim, dim, cost_limit);
        } else {
            dim = size;
            extd_cost = (double **)calloc(dim, sizeof(double *));
            for (size_t i = 0; i < dim; ++i) {
                extd_cost[i] = (double *)calloc(dim, sizeof(double));
            }
            
            if (size == rows) {
                matrix_set_roi(extd_cost, cols, 0, dim, dim, maxele + 1);
            } else {
                matrix_set_roi(extd_cost, 0, rows, dim, dim, maxele + 1);
            }
        }
    } else {
        if (cost_limit < FLT_MAX) {
            dim = size << 1;
        } else {
            dim = size;
        }
        
        extd_cost = (double **)calloc(dim, sizeof(double *));
        for (size_t i = 0; i < dim; ++i) {
            extd_cost[i] = (double *)calloc(dim, sizeof(double));
        }
        
        if (cost_limit < FLT_MAX) {
            matrix_set_roi(extd_cost, cols, 0, dim, rows, cost_limit);
            matrix_set_roi(extd_cost, 0, rows, dim, dim, cost_limit);
        }
    }
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            extd_cost[i][j] = cost[i * cols + j];
        }
    }
    
    int *extd_x = (int *)calloc(dim, sizeof(int));
    int *extd_y = (int *)calloc(dim, sizeof(int));
    
    int ret = lapjv_internal(dim, extd_cost, extd_x, extd_y);
    
    *opt = 0;
    if (cost_limit < FLT_MAX || extend_cost) {
        for (size_t i = 0; i < dim; ++i) {
            if (extd_x[i] >= cols) {
                extd_x[i] = -1;
            }
        }
        for (size_t i = 0; i < dim; ++i) {
            if (extd_y[i] >= rows) {
                extd_y[i] = -1;
            }
        }
        for (size_t i = 0; i < rows; ++i) {
            x[i] = extd_x[i];
        }
        for (size_t i = 0; i < cols; ++i) {
            y[i] = extd_y[i];
        }
        
        for (size_t i = 0; i < rows; ++i) {
            if (-1 != x[i]) {
                *opt += cost[i * cols + x[i]];
            }
        }
    } else {
        for (size_t i = 0; i < rows; ++i) {
            x[i] = extd_x[i];
        }
        for (size_t i = 0; i < cols; ++i) {
            y[i] = extd_y[i];
        }
        
        for (size_t i = 0; i < rows; ++i) {
            *opt += cost[i * cols + x[i]];
        }
    }
    
    for (size_t i = 0; i < dim; ++i) {
        std::free(extd_cost[i]);
        extd_cost[i] = NULL;
    }
    
    std::free(extd_cost);
    extd_cost = NULL;
    
    std::free(extd_x);
    extd_x = NULL;
    
    std::free(extd_y);
    extd_y = NULL;
    
    if (0 != ret) {
        if (-1 == ret) {
            fprintf(stderr, "out of memory\n");
        } else {
            fprintf(stderr, "unknown error (lapjv_internal return %d)\n", ret);
        }
        return false;
    }
        
    return true;
}

void LAPJV::free(void)
{
}

#endif  // PYTHON_LAPJV

}   // namespace mot