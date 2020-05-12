#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// building command:
// /opt/rh/devtoolset-6/root/usr/bin/g++ -o cdist cdist.cpp

void cdist(float *XA, int ma, float *XB, int mb, int n, float *Y)
{
    for (int i = 0; i < ma; ++i)
    {
        float *yi = Y + i * mb;
        float *xa = XA + i * n;
        for (int j = 0; j < mb; ++j)
        {
            float dist = 0;
            float *xb = XB + j * n;
            for (int k = 0; k < n; ++k)
            {
                dist += (xa[k] - xb[k]) * (xa[k] - xb[k]);
            }
            *(yi + j) = sqrtf(dist);
        }
    }
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    const int n = 512;
    const int ma = rand() % 10;
    const int mb = rand() % 10;
    
    fprintf(stderr, "ma is %d, mb is %d, n is %d\n", ma, mb, n);
    float *XA = 0;
    if (ma > 0)
        XA = (float *)calloc(ma * n, sizeof(float));
    
    float *XB = 0;
    if (mb > 0)
        XB = (float *)calloc(mb * n, sizeof(float));
    
    float *Y = 0;
    if (ma > 0 && mb > 0)
        Y = (float *)calloc(ma * mb, sizeof(float));
    
    for (int i = 0; i < ma * n; ++i)
        XA[i] = (float)rand() / RAND_MAX;
    
    FILE *fa = fopen("XA.bin", "wb");
    fwrite(XA, sizeof(float), ma * n, fa);
    fclose(fa);
    
    for (int i = 0; i < mb * n; ++i)
        XB[i] = (float)rand() / RAND_MAX;
    
    FILE *fb = fopen("XB.bin", "wb");
    fwrite(XB, sizeof(float), mb * n, fb);
    fclose(fb);
    
    cdist(XA, ma, XB, mb, n, Y);
    
    FILE *fy = fopen("Y.bin", "wb");    
    fwrite(Y, sizeof(float), ma * mb, fy);
    fclose(fy);
    
    if (XA)
        free(XA);
    
    if (XB)
        free(XB);
    
    if (Y)
        free(Y);
    
    return 0;
}