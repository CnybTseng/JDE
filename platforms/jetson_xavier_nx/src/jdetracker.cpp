#include <map>
#include <omp.h>
#include <chrono>
#include <stdio.h>
#include <limits.h>
#include <algorithm>

#include "utils.h"
#include "lapjv.h"
#include "jdeutils.h"
#include "jdetracker.h"

#define MAXIMUM_CANDIDATES 1000
#define mat2vec4f(m) cv::Vec4f(*m.ptr<float>(0,0), *m.ptr<float>(0,1), *m.ptr<float>(0,2), *m.ptr<float>(0,3))

namespace mot {

static std::map<int, float> chi2inv95 = {
    {1,  3.841459f},
    {2,  5.991465f},
    {3,  7.814728f},
    {4,  9.487729f},
    {5, 11.070498f},
    {6, 12.591587f},
    {7, 14.067140f},
    {8, 15.507313f},
    {9, 16.918978f}
};

JDETracker *JDETracker::me = NULL;

JDETracker *JDETracker::instance(void)
{
    if (!me)
        me = new JDETracker;
    return me;
}

JDETracker::JDETracker(void) : timestamp(0), max_lost_time(30), lambda(0.98f)
{    
}

bool JDETracker::init(void)
{
    bool ret = LAPJV::instance()->init();
    check_error_ret(!ret, false, "solver init fail!\n");
    candidates.reserve(MAXIMUM_CANDIDATES);
    return true;
}

bool JDETracker::update(const cv::Mat &dets, std::vector<Track> &tracks)
{
#if PROFILE_TRACKER
    auto create_candi = std::chrono::high_resolution_clock::now();
#endif
    ++timestamp;
    candidates.resize(dets.rows);
#if PROFILE_TRACKER
    profiler.reportLayerTime("create_candi", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - create_candi).count());
    auto set_candi = std::chrono::high_resolution_clock::now();
#endif
    // About 1.9x speed up using OpenMP and other optimization methods.
#pragma omp parallel for num_threads(2)
    for (int i = 0; i < dets.rows; ++i)
    {
        float score = *dets.ptr<float>(i, 1);
        const cv::Mat &ltrb_ = dets(cv::Rect(2, i, 4, 1));
        cv::Vec4f ltrb = mat2vec4f(ltrb_);
        const cv::Mat &embedding = dets(cv::Rect(6, i, dets.cols - 6, 1));
        candidates[i] = mot::Trajectory(ltrb, score, embedding);
    }
#if PROFILE_TRACKER
    profiler.reportLayerTime("set_candi", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - set_candi).count());
    auto init_trajpool = std::chrono::high_resolution_clock::now();
#endif
    TrajectoryPtrPool tracked_trajectories;
    TrajectoryPtrPool unconfirmed_trajectories;
    for (size_t i = 0; i < this->tracked_trajectories.size(); ++i)
    {
        if (this->tracked_trajectories[i].is_activated)
            tracked_trajectories.push_back(&this->tracked_trajectories[i]);
        else
            unconfirmed_trajectories.push_back(&this->tracked_trajectories[i]);
    }

    TrajectoryPtrPool trajectory_pool = tracked_trajectories + this->lost_trajectories;
#if PROFILE_TRACKER
    profiler.reportLayerTime("init_trajpool", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - init_trajpool).count());
    auto start_predict = std::chrono::high_resolution_clock::now();
#endif
#pragma omp parallel for num_threads(2)
    for (size_t i = 0; i < trajectory_pool.size(); ++i)
        trajectory_pool[i]->predict();
#if PROFILE_TRACKER
    profiler.reportLayerTime("predict", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_predict).count());
    auto start_motion = std::chrono::high_resolution_clock::now();
#endif   
    Match matches;
    std::vector<int> mismatch_row;
    std::vector<int> mismatch_col;
    cv::Mat cost = motion_distance(trajectory_pool, candidates);
#if PROFILE_TRACKER
    profiler.reportLayerTime("motion_distance", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_motion).count());
    auto start_lap = std::chrono::high_resolution_clock::now();
#endif 
    linear_assignment(cost, 0.7f, matches, mismatch_row, mismatch_col);
 #if PROFILE_TRACKER
    profiler.reportLayerTime("linear_assignment", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_lap).count());
#endif   
    MatchIterator miter;
    TrajectoryPtrPool activated_trajectories;
    TrajectoryPtrPool retrieved_trajectories;    
    for (miter = matches.begin(); miter != matches.end(); miter++)
    {
        Trajectory *pt = trajectory_pool[miter->first];
        Trajectory &ct = candidates[miter->second];
        if (pt->state == Tracked)
        {
            pt->update(ct, timestamp);
            activated_trajectories.push_back(pt);
        }
        else
        {
            pt->reactivate(ct, timestamp);
            retrieved_trajectories.push_back(pt);
        }
    }
#if PROFILE_TRACKER
    profiler.reportLayerTime("motion", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_motion).count());
    auto start_iou = std::chrono::high_resolution_clock::now();
#endif    
    TrajectoryPtrPool next_candidates(mismatch_col.size());
    for (size_t i = 0; i < mismatch_col.size(); ++i)
        next_candidates[i] = &candidates[mismatch_col[i]];
    
    TrajectoryPtrPool next_trajectory_pool;
    for (size_t i = 0; i < mismatch_row.size(); ++i)
    {
        int j = mismatch_row[i];
        if (trajectory_pool[j]->state == Tracked)
            next_trajectory_pool.push_back(trajectory_pool[j]);
    }
#if PROFILE_TRACKER
    auto start_iou_dist = std::chrono::high_resolution_clock::now();
#endif
    cost = iou_distance(next_trajectory_pool, next_candidates);
#if PROFILE_TRACKER
    profiler.reportLayerTime("iou_distance", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_iou_dist).count());
#endif 
    linear_assignment(cost, 0.5f, matches, mismatch_row, mismatch_col);
    
    for (miter = matches.begin(); miter != matches.end(); miter++)
    {
        Trajectory *pt = next_trajectory_pool[miter->first];
        Trajectory *ct = next_candidates[miter->second];
        if (pt->state == Tracked)
        {
            pt->update(*ct, timestamp);
            activated_trajectories.push_back(pt);
        }
        else
        {
            pt->reactivate(*ct, timestamp);
            retrieved_trajectories.push_back(pt);
        }
    }
#if PROFILE_TRACKER
    profiler.reportLayerTime("iou", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_iou).count());
    auto start_lost_iou = std::chrono::high_resolution_clock::now();
#endif    
    TrajectoryPtrPool lost_trajectories;
    for (size_t i = 0; i < mismatch_row.size(); ++i)
    {
        Trajectory *pt = next_trajectory_pool[mismatch_row[i]];
        if (pt->state != Lost)
        {
            pt->mark_lost();
            lost_trajectories.push_back(pt);
        }
    }
    
    TrajectoryPtrPool nnext_candidates(mismatch_col.size());
    for (size_t i = 0; i < mismatch_col.size(); ++i)
        nnext_candidates[i] = next_candidates[mismatch_col[i]];
    
    cost = iou_distance(unconfirmed_trajectories, nnext_candidates);
    linear_assignment(cost, 0.7f, matches, mismatch_row, mismatch_col);
    
    for (miter = matches.begin(); miter != matches.end(); miter++)
    {
        unconfirmed_trajectories[miter->first]->update(*nnext_candidates[miter->second], timestamp);
        activated_trajectories.push_back(unconfirmed_trajectories[miter->first]);
    }
#if PROFILE_TRACKER
    profiler.reportLayerTime("lost iou", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_lost_iou).count());
    auto start_left_work = std::chrono::high_resolution_clock::now();
#endif     
    TrajectoryPtrPool removed_trajectories;
    for (size_t i = 0; i < mismatch_row.size(); ++i)
    {
        unconfirmed_trajectories[mismatch_row[i]]->mark_removed();
        removed_trajectories.push_back(unconfirmed_trajectories[mismatch_row[i]]);
    }
    
    for (size_t i = 0; i < mismatch_col.size(); ++i)
    {
        nnext_candidates[mismatch_col[i]]->activate(timestamp);
        activated_trajectories.push_back(nnext_candidates[mismatch_col[i]]);
    }
    
    for (size_t i = 0; i < this->lost_trajectories.size(); ++i)
    {
        Trajectory &lt = this->lost_trajectories[i];
        if (timestamp - lt.timestamp > max_lost_time)
        {
            lt.mark_removed();
            removed_trajectories.push_back(&lt);
        }
    }
    
    TrajectoryPoolIterator piter;
    for (piter = this->tracked_trajectories.begin(); piter != this->tracked_trajectories.end(); )
    {
        if (piter->state != Tracked)
            piter = this->tracked_trajectories.erase(piter);
        else
            ++piter;
    }
    
    this->tracked_trajectories += activated_trajectories;
    this->tracked_trajectories += retrieved_trajectories;
    this->lost_trajectories -= this->tracked_trajectories;
    this->lost_trajectories += lost_trajectories;
    this->lost_trajectories -= this->removed_trajectories;
    this->removed_trajectories += removed_trajectories;
    remove_duplicate_trajectory(this->tracked_trajectories, this->lost_trajectories);
    
    tracks.clear();
    for (size_t i = 0; i < this->tracked_trajectories.size(); ++i)
    {
        if (this->tracked_trajectories[i].is_activated)
        {
            Track track = {
                .id = this->tracked_trajectories[i].id,
                .ltrb = this->tracked_trajectories[i].ltrb};
            tracks.push_back(track);
        }
    }
#if PROFILE_TRACKER
    profiler.reportLayerTime("left work", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_left_work).count());
#endif     
    return 0;
}

void JDETracker::free(void)
{
    LAPJV::instance()->free();
}

cv::Mat JDETracker::motion_distance(const TrajectoryPtrPool &a, const TrajectoryPool &b)
{
    if (0 == a.size() || 0 == b.size())
        return cv::Mat(a.size(), b.size(), CV_32F);
#if PROFILE_TRACKER
    auto start_embd = std::chrono::high_resolution_clock::now();
#endif   
    cv::Mat edists = embedding_distance(a, b);
#if PROFILE_TRACKER
    profiler.reportLayerTime("embedding_distance", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_embd).count());
    auto start_maha = std::chrono::high_resolution_clock::now();
#endif
    cv::Mat mdists = mahalanobis_distance(a, b);
#if PROFILE_TRACKER
    profiler.reportLayerTime("mahalanobis_distance", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_maha).count());
    auto start_linear = std::chrono::high_resolution_clock::now();
#endif
    cv::Mat fdists = lambda * edists + (1 - lambda) * mdists;
#if PROFILE_TRACKER
    profiler.reportLayerTime("linear combine", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_linear).count());
    auto start_gate = std::chrono::high_resolution_clock::now();
#endif    
    const float gate_thresh = chi2inv95[4];
    for (int i = 0; i < fdists.rows; ++i)
    {
        for (int j = 0; j < fdists.cols; ++j)
        {
            if (*mdists.ptr<float>(i, j) > gate_thresh)
                *fdists.ptr<float>(i, j) = FLT_MAX;
        }
    }
#if PROFILE_TRACKER
    profiler.reportLayerTime("gate", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_gate).count());
#endif    
    return fdists;
}

void JDETracker::linear_assignment(const cv::Mat &cost, float cost_limit, Match &matches,
    std::vector<int> &mismatch_row, std::vector<int> &mismatch_col)
{
    matches.clear();
    mismatch_row.clear();    
    mismatch_col.clear();
    
    if (cost.empty())
    {
        for (int i = 0; i < cost.rows; ++i)
            mismatch_row.push_back(i);
        for (int i = 0; i < cost.cols; ++i)
            mismatch_col.push_back(i);
        return;
    }
    
    float opt;
    cv::Mat x(cost.rows, 1, CV_32S);
    cv::Mat y(cost.cols, 1, CV_32S);
    
    bool ret = LAPJV::instance()->solve((float *)cost.data, cost.rows, cost.cols, &opt,
        (int *)x.data, (int *)y.data, true, cost_limit);
    if (!ret)
        return;
    
    for (int i = 0; i < x.rows; ++i)
    {
        int j = *x.ptr<int>(i);
        if (j >= 0)
            matches.insert({i, j});
        else
            mismatch_row.push_back(i);
    }
    
    for (int j = 0; j < y.rows; ++j)
    {
        int i = *y.ptr<int>(j);
        if (i < 0)
            mismatch_col.push_back(j);
    }
}

void JDETracker::remove_duplicate_trajectory(TrajectoryPool &a, TrajectoryPool &b, float iou_thresh)
{
    if (0 == a.size() || 0 == b.size())
        return;
    
    cv::Mat dist = iou_distance(a, b);
    cv::Mat mask = dist < iou_thresh;
    std::vector<cv::Point> idx;
    cv::findNonZero(mask, idx);
    
    std::vector<int> da;
    std::vector<int> db;
    for (size_t i = 0; i < idx.size(); ++i)
    {
        int ta = a[idx[i].y].timestamp - a[idx[i].y].starttime;
        int tb = b[idx[i].x].timestamp - b[idx[i].x].starttime;
        if (ta > tb)
            db.push_back(idx[i].x);
        else
            da.push_back(idx[i].y);
    }
    
    int id = 0;
    TrajectoryPoolIterator piter;
    for (piter = a.begin(); piter != a.end(); )
    {
        std::vector<int>::iterator iter = find(da.begin(), da.end(), id++);
        if (iter != da.end())
            piter = a.erase(piter);
        else
            ++piter;
    }
    
    id = 0;
    for (piter = b.begin(); piter != b.end(); )
    {
        std::vector<int>::iterator iter = find(db.begin(), db.end(), id++);
        if (iter != db.end())
            piter = b.erase(piter);
        else
            ++piter;
    }
}

}   // namespace mot