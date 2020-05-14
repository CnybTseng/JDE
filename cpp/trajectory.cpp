#include "trajectory.h"

namespace mot {

void TKalmanFilter::init(const cv::Mat &measurement)
{
    measurement.copyTo(cv::KalmanFilter::statePost(cv::Rect(0, 0, 1, 4)));
    cv::KalmanFilter::statePost(cv::Rect(0, 4, 1, 4)).setTo(0);
    cv::KalmanFilter::statePost.copyTo(cv::KalmanFilter::statePre);

    float varpos = 2 *  std_weight_position * measurement.at<float>(3);
    varpos *= varpos;
    float varvel = 10 * std_weight_velocity * measurement.at<float>(3);
    varvel *= varvel;
    
    cv::KalmanFilter::errorCovPost.setTo(0);
    cv::KalmanFilter::errorCovPost.at<float>(0, 0) = varpos;
    cv::KalmanFilter::errorCovPost.at<float>(1, 1) = varpos;
    cv::KalmanFilter::errorCovPost.at<float>(2, 2) = 1e-4f;
    cv::KalmanFilter::errorCovPost.at<float>(3, 3) = varpos;
    cv::KalmanFilter::errorCovPost.at<float>(4, 4) = varvel;
    cv::KalmanFilter::errorCovPost.at<float>(5, 5) = varvel;
    cv::KalmanFilter::errorCovPost.at<float>(6, 6) = 1e-10f;
    cv::KalmanFilter::errorCovPost.at<float>(7, 7) = varvel;  
    cv::KalmanFilter::errorCovPost.copyTo(cv::KalmanFilter::errorCovPre);
}

const cv::Mat &TKalmanFilter::predict()
{
    float varpos = std_weight_position * cv::KalmanFilter::statePre.at<float>(3);
    varpos *= varpos;
    float varvel = std_weight_velocity * cv::KalmanFilter::statePre.at<float>(3);
    varvel *= varvel;
    
    cv::KalmanFilter::processNoiseCov.setTo(0);
    cv::KalmanFilter::processNoiseCov.at<float>(0, 0) = varpos;
    cv::KalmanFilter::processNoiseCov.at<float>(1, 1) = varpos;
    cv::KalmanFilter::processNoiseCov.at<float>(2, 2) = 1e-4f;
    cv::KalmanFilter::processNoiseCov.at<float>(3, 3) = varpos;
    cv::KalmanFilter::processNoiseCov.at<float>(4, 4) = varvel;
    cv::KalmanFilter::processNoiseCov.at<float>(5, 5) = varvel;
    cv::KalmanFilter::processNoiseCov.at<float>(6, 6) = 1e-10f;
    cv::KalmanFilter::processNoiseCov.at<float>(7, 7) = varvel;
    
    return cv::KalmanFilter::predict();
}

const cv::Mat &TKalmanFilter::correct(const cv::Mat &measurement)
{
    float varpos = std_weight_position * measurement.at<float>(3);
    varpos *= varpos;
    
    const float h = measurement.at<float>(3);
    cv::KalmanFilter::measurementNoiseCov.setTo(0);
    cv::KalmanFilter::measurementNoiseCov.at<float>(0, 0) = varpos;
    cv::KalmanFilter::measurementNoiseCov.at<float>(1, 1) = varpos;
    cv::KalmanFilter::measurementNoiseCov.at<float>(2, 2) = 1e-2f;
    cv::KalmanFilter::measurementNoiseCov.at<float>(3, 3) = varpos;
    
    return cv::KalmanFilter::correct(measurement);
}

void TKalmanFilter::project(cv::Mat &mean, cv::Mat &covariance)
{   
    cv::Mat &statePost = cv::KalmanFilter::statePost;
    cv::Mat &errorCovPost = cv::KalmanFilter::errorCovPost;
    cv::Mat &measurementMatrix = cv::KalmanFilter::measurementMatrix;
    
    float varpos = std_weight_position * statePost.at<float>(3);
    varpos *= varpos;
    
    cv::Mat measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F);
    measurementNoiseCov.at<float>(0, 0) = varpos;
    measurementNoiseCov.at<float>(1, 1) = varpos;
    measurementNoiseCov.at<float>(2, 2) = 1e-2f;
    measurementNoiseCov.at<float>(3, 3) = varpos;
        
    mean = measurementMatrix.dot(statePost);
    cv::Mat temp = measurementMatrix * errorCovPost;
    gemm(temp, measurementMatrix, 1, measurementNoiseCov, 1, covariance, cv::GEMM_2_T);
}

void TKalmanFilter::gating_distance(std::vector<cv::Mat> &measurements, std::vector<float> &dists)
{
    cv::Mat mean;
    cv::Mat covariance;
    project(mean, covariance);
    
    cv::Mat icovariance;
    cv::invert(covariance, icovariance);
    
    dists.clear();
    dists.resize(measurements.size());
    
    for (int i = 0; i < measurements.size(); ++i)
    {
        cv::Mat &x = measurements[i];
        float dist = static_cast<float>(cv::Mahalanobis(x, mean, icovariance));
        dists[i] = dist * dist;
    }
}

int Trajectory::count = 0;

const cv::Mat &Trajectory::predict(void)
{
    if (state != Tracked)
        cv::KalmanFilter::statePost.at<float>(7) = 0;
    return KalmanFilter::predict();
}

void Trajectory::update(Trajectory &traj, int timestamp_, bool update_embedding_)
{
    timestamp = timestamp_;
    ++length;
    ltrb = traj.ltrb;
    xyah = traj.xyah;    
    TKalmanFilter::correct(cv::Mat(traj.xyah));    
    state = Tracked;
    is_activated = true;
    score = traj.score;   
    if (update_embedding_)
        update_embedding(traj.current_embedding);
}

void Trajectory::activate(int timestamp_)
{
    id = next_id();    
    TKalmanFilter::init(cv::Mat(xyah));    
    length = 0;
    state = Tracked;
    timestamp = timestamp_;
    starttime = timestamp_;
}

void Trajectory::reactivate(Trajectory &traj, int timestamp_, bool newid)
{
    TKalmanFilter::correct(cv::Mat(traj.xyah));
    update_embedding(traj.current_embedding);
    length = 0;
    state = Tracked;
    is_activated = true;
    timestamp = timestamp_;
    if (newid)
        id = next_id();
}

void Trajectory::update_embedding(const cv::Mat &embedding)
{
    current_embedding = embedding / cv::norm(embedding);
    if (smooth_embedding.empty())
        smooth_embedding = current_embedding;
    else
        smooth_embedding = eta * smooth_embedding + (1 - eta) * current_embedding;
    smooth_embedding = smooth_embedding / cv::norm(smooth_embedding);
}

}   // namespace mot

#ifdef TEST_TRAJECTORY_MODULE

#include <time.h>
#include <iostream>

int main(void)
{
    mot::TKalmanFilter tkf;
    
    cv::RNG rng(time(NULL));
    cv::Mat measurement(4, 1, CV_32F);
    rng.fill(measurement, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(5));
    std::cout << measurement << std::endl;
    
    tkf.init(measurement);    
    tkf.predict();

    rng.fill(measurement, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(5));
    cv::Mat state = tkf.correct(measurement);
    std::cout << state << std::endl;
    
    return 0;
}

#endif // TEST_TRAJECTORY_MODULE