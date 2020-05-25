#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "lapjv.h"
#include "jdeutils.h"

using namespace mot;

int main(int argc, char *argv[])
{
    int nloops = 1;
    float cost[962];
    float opt;
    int x[32];
    int y[32];
    
    if (argc > 1)
        nloops = atoi(argv[1]);
    
    bool ret = LAPJV::instance()->init();
    check_error_ret(!ret, 0, "solver init fail!\n");
    
    srand(time(NULL));
    for (int t = 0; t < nloops; ++t)
    {
        int rows = rand() % 30 + 1;
        int cols = rand() % 30 + 1;
        for (int i = 0; i < rows * cols; ++i)
            cost[i] = (float)rand() / RAND_MAX;
        
        ret = LAPJV::instance()->solve(cost, rows, cols, &opt, x, y, true, 0.7f);
        if (!ret)
            break;
        
        if (nloops > 1)
        {
            fprintf(stderr, "%d\n", t);
            continue;
        }
        
        fprintf(stderr, "cost matrix size:%dx%d\n", rows, cols);
        fprintf(stderr, "opt: %f\n", opt);
        fprintf(stderr, "x:\n");
        for (int i = 0; i < rows; ++i)
            fprintf(stderr, "%d ", x[i]);
        
        fprintf(stderr, "\ny:\n");
        for (int i = 0; i < cols; ++i)
            fprintf(stderr, "%d ", y[i]);
        
        fprintf(stderr, "\n");
        
        FILE *fp = fopen("cost.bin", "wb");
        fwrite(cost, rows * cols, sizeof(float), fp);
        fclose(fp);
    }
    
    LAPJV::instance()->free();   
    return 0;
}