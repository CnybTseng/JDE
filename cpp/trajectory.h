#ifndef TRAJECTORY_H
#define TRAJECTORY_H

#include <vector>
#include <opencv2/opencv.hpp>

namespace mot {

typedef enum
{
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3
} TrajectoryState;

class Trajectory;
typedef std::vector<Trajectory> TrajectoryPool;
typedef std::vector<Trajectory>::iterator TrajectoryPoolIterator;
typedef std::vector<Trajectory *>TrajectoryPtrPool;
typedef std::vector<Trajectory *>::iterator TrajectoryPtrPoolIterator;

class TKalmanFilter : public cv::KalmanFilter
{
public:
    TKalmanFilter(void);
    virtual ~TKalmanFilter(void) {}
    void init(const cv::Mat &measurement);
    const cv::Mat &predict();
    const cv::Mat &correct(const cv::Mat &measurement);
    void project(cv::Mat &mean, cv::Mat &covariance) const;
    void gating_distance(std::vector<cv::Mat> &measurements, std::vector<float> &dists);
private:
    float std_weight_position;
    float std_weight_velocity;
};

inline TKalmanFilter::TKalmanFilter(void) : cv::KalmanFilter(8, 4)
{
    cv::KalmanFilter::transitionMatrix = cv::Mat::eye(8, 8, CV_32F);
    for (int i = 0; i < 4; ++i)
        cv::KalmanFilter::transitionMatrix.at<float>(i, i + 4) = 1;
    cv::KalmanFilter::measurementMatrix = cv::Mat::eye(4, 8, CV_32F);
    std_weight_position = 1/20.f;
    std_weight_velocity = 1/160.f;
}

class Trajectory : public TKalmanFilter
{
private:
    static int count;
public:
    Trajectory();
    Trajectory(cv::Vec4f &ltrb, float score, const cv::Mat &embedding);
    Trajectory(const Trajectory &other);
    Trajectory &operator=(const Trajectory &rhs);
    virtual ~Trajectory(void) {};
    static int next_id();
    const cv::Mat & predict(void);
    void update(Trajectory &traj, int timestamp, bool update_embedding=true);
    void activate(int timestamp);
    void reactivate(Trajectory &traj, int timestamp, bool newid=false);
    void mark_lost(void);
    void mark_removed(void);
    friend TrajectoryPool operator+(const TrajectoryPool &a, const TrajectoryPool &b);
    friend TrajectoryPool operator+(const TrajectoryPool &a, const TrajectoryPtrPool &b);
    friend TrajectoryPool &operator+=(TrajectoryPool &a, const TrajectoryPtrPool &b);
    friend TrajectoryPool operator-(const TrajectoryPool &a, const TrajectoryPool &b);
    friend TrajectoryPool &operator-=(TrajectoryPool &a, const TrajectoryPool &b);
    friend TrajectoryPtrPool operator+(const TrajectoryPtrPool &a, const TrajectoryPtrPool &b);
    friend TrajectoryPtrPool operator+(const TrajectoryPtrPool &a, TrajectoryPool &b);
    friend TrajectoryPtrPool operator-(const TrajectoryPtrPool &a, const TrajectoryPtrPool &b);
    friend cv::Mat embedding_distance(const TrajectoryPool &a, const TrajectoryPool &b);
    friend cv::Mat embedding_distance(const TrajectoryPtrPool &a, const TrajectoryPtrPool &b);
    friend cv::Mat embedding_distance(const TrajectoryPtrPool &a, const TrajectoryPool &b);
    friend cv::Mat mahalanobis_distance(const TrajectoryPool &a, const TrajectoryPool &b);
    friend cv::Mat mahalanobis_distance(const TrajectoryPtrPool &a, const TrajectoryPtrPool &b);
    friend cv::Mat mahalanobis_distance(const TrajectoryPtrPool &a, const TrajectoryPool &b);
    friend cv::Mat iou_distance(const TrajectoryPool &a, const TrajectoryPool &b);
    friend cv::Mat iou_distance(const TrajectoryPtrPool &a, const TrajectoryPtrPool &b);
    friend cv::Mat iou_distance(const TrajectoryPtrPool &a, const TrajectoryPool &b);
public:
    TrajectoryState state;
private:
    cv::Vec4f ltrb;
    cv::Vec4f xyah;
    float score;
    cv::Mat current_embedding;
    cv::Mat smooth_embedding;
    int id;
public:
    bool is_activated;
private:
    float eta;
public:
    int timestamp;
private:
    int length;
public:
    int starttime;
private:   
    void update_embedding(const cv::Mat &embedding);
};

inline cv::Vec4f ltrb2xyah(cv::Vec4f &ltrb)
{
    cv::Vec4f xyah;
    xyah[0] = (ltrb[0] + ltrb[2]) * 0.5f;
    xyah[1] = (ltrb[1] + ltrb[3]) * 0.5f;
    xyah[3] =  ltrb[3] - ltrb[1];
    xyah[2] = (ltrb[2] - ltrb[0]) / xyah[3];
    return xyah;
}

inline Trajectory::Trajectory() :
    state(New), ltrb(cv::Vec4f()), score(0), smooth_embedding(cv::Mat()), id(0),
    is_activated(false), eta(0.9), timestamp(0), length(0), starttime(0)
{
}

inline Trajectory::Trajectory(cv::Vec4f &ltrb_, float score_, const cv::Mat &embedding) :
    state(New), ltrb(ltrb_), score(score_), smooth_embedding(cv::Mat()), id(0),
    is_activated(false), eta(0.9), timestamp(0), length(0), starttime(0)
{
    xyah = ltrb2xyah(ltrb);
    update_embedding(embedding);
}

inline Trajectory::Trajectory(const Trajectory &other):
    state(other.state), ltrb(other.ltrb), xyah(other.xyah), score(other.score),
    id(other.id), is_activated(other.is_activated), eta(other.eta), timestamp(other.timestamp),
    length(other.length), starttime(other.starttime)
{
    other.current_embedding.copyTo(current_embedding);
    other.smooth_embedding.copyTo(smooth_embedding);
}

inline Trajectory &Trajectory::operator=(const Trajectory &rhs)
{
    this->state = rhs.state;
    this->ltrb = rhs.ltrb;
    this->xyah = rhs.xyah;
    this->score = rhs.score;
    rhs.current_embedding.copyTo(this->current_embedding);
    rhs.smooth_embedding.copyTo(this->smooth_embedding);
    this->id = rhs.id;
    this->is_activated = rhs.is_activated;
    this->eta = rhs.eta;
    this->timestamp = rhs.timestamp;
    this->length = rhs.length;
    this->starttime = rhs.starttime;    
    return *this;
}

inline int Trajectory::next_id()
{
    ++count;
    return count;
}

inline void Trajectory::mark_lost(void)
{
    state = Lost;
}

inline void Trajectory::mark_removed(void)
{
    state = Removed;
}

}   // namespace mot

#endif  // TRAJECTORY_H