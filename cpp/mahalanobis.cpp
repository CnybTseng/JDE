#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <opencv2/opencv.hpp>

// building command
// /opt/rh/devtoolset-6/root/usr/bin/g++ -o mahalanobis mahalanobis.cpp -lopencv_core

int main(int argc, char *argv[])
{
    cv::RNG rng(time(NULL));
    
    cv::Mat x(1, 10, CV_32FC1);
    rng.fill(x, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(5));
    
    cv::Mat mean(1, 10, CV_32FC1);
    rng.fill(mean, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(5));
    
    cv::Mat covariance(10, 10, CV_32FC1);
    rng.fill(covariance, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(5));
    
    double dist = cv::Mahalanobis(x, mean, covariance);
    fprintf(stderr, "dist is %.10f\n", dist);
    
    FILE *fp = fopen("u.bin", "wb");
    fwrite((float *)x.data, sizeof(float), x.cols, fp);
    fclose(fp);
    
    fp = fopen("v.bin", "wb");
    fwrite((float *)mean.data, sizeof(float), mean.cols, fp);
    fclose(fp);
    
    fp = fopen("VI.bin", "wb");
    fwrite((float *)covariance.data, sizeof(float), covariance.rows * covariance.cols, fp);
    fclose(fp);
    
    return 0;
}