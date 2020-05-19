#include <map>
#include <limits.h>
#include <algorithm>

#include "lapjv.h"
#include "jdeutils.h"
#include "jdetracker.h"

#define mat2vec4f(m) cv::Vec4f(*m.ptr<float>(0), *m.ptr<float>(1), *m.ptr<float>(2), *m.ptr<float>(3))

std::map<int, float> chi2inv95 = {
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

namespace mot {

JDETracker *JDETracker::me = NULL;

bool JDETracker::init(void)
{
    bool ret = LAPJV::instance()->init();
    check_error_ret(!ret, false, "solver init fail!\n");
    return true;
}

bool JDETracker::update(const cv::Mat &dets)
{
    ++timestamp;
    TrajectoryPool candidates(dets.rows);
    for (int i = 0; i < dets.rows; ++i)
    {
        float score = *dets.ptr<float>(i, 1);
        const cv::Mat &ltrb_ = dets(cv::Rect(2, i, 4, 1));
        cv::Vec4f ltrb = mat2vec4f(ltrb_);
        const cv::Mat &embedding = dets(cv::Rect(6, i, dets.cols - 6, 1));
        candidates[i] = mot::Trajectory(ltrb, score, embedding);
    }
    
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
    for (size_t i = 0; i < trajectory_pool.size(); ++i)
        trajectory_pool[i]->predict();
        
    Match matches;
    std::vector<int> mismatch_row;
    std::vector<int> mismatch_col;
    cv::Mat cost = motion_distance(trajectory_pool, candidates);
    linear_assignment(cost, 0.7f, matches, mismatch_row, mismatch_col);
    
    MatchIterator miter;
    TrajectoryPtrPool activated_trajectories;
    TrajectoryPtrPool retrieved_trajectories;    
    for (miter = matches.begin(); miter != matches.end(); miter++)
    {
        Trajectory &pt = *trajectory_pool[miter->first];
        Trajectory &ct = candidates[miter->second];
        if (pt.state == Tracked)
        {
            pt.update(ct, timestamp);
            activated_trajectories.push_back(&pt);
        }
        else
        {
            pt.reactivate(ct, timestamp);
            retrieved_trajectories.push_back(&pt);
        }
    }
    
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
    
    cost = iou_distance(next_trajectory_pool, next_candidates);
    linear_assignment(cost, 0.5f, matches, mismatch_row, mismatch_col);
    
    for (miter = matches.begin(); miter != matches.end(); miter++)
    {
        Trajectory &pt = *next_trajectory_pool[miter->first];
        Trajectory &ct = *next_candidates[miter->second];
        if (pt.state == Tracked)
        {
            pt.update(ct, timestamp);
            activated_trajectories.push_back(&pt);
        }
        else
        {
            pt.reactivate(ct, timestamp);
            retrieved_trajectories.push_back(&pt);
        }
    }
    
    TrajectoryPtrPool lost_trajectories;
    for (size_t i = 0; i < mismatch_row.size(); ++i)
    {
        Trajectory &pt = *next_trajectory_pool[mismatch_row[i]];
        if (pt.state != Lost)
        {
            pt.mark_lost();
            lost_trajectories.push_back(&pt);
        }
    }
    
    TrajectoryPtrPool nnext_candidates(mismatch_col.size());
    for (size_t i = 0; i < mismatch_col.size(); ++i)
        nnext_candidates[i] = next_candidates[mismatch_row[i]];
    
    cost = iou_distance(unconfirmed_trajectories, nnext_candidates);
    linear_assignment(cost, 0.7f, matches, mismatch_row, mismatch_col);
    
    for (miter = matches.begin(); miter != matches.end(); miter++)
    {
        unconfirmed_trajectories[miter->first]->update(*nnext_candidates[miter->second], timestamp);
        activated_trajectories.push_back(unconfirmed_trajectories[miter->first]);
    }
    
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
        
    return 0;
}

void JDETracker::free(void)
{
    LAPJV::instance()->free();
}

cv::Mat JDETracker::motion_distance(const TrajectoryPtrPool &a, const TrajectoryPool &b)
{
    cv::Mat edists = embedding_distance(a, b);
    cv::Mat mdists = mahalanobis_distance(a, b);
    cv::Mat fdists = lambda * edists + (1 - lambda) * mdists;
    
    const float gate_thresh = chi2inv95[4];
    for (int i = 0; i < fdists.rows; ++i)
    {
        for (int j = 0; j < fdists.cols; ++j)
        {
            if (*mdists.ptr<float>(i, j) > gate_thresh)
                *fdists.ptr<float>(i, j) = FLT_MAX;
        }
    }
    
    return fdists;
}

void JDETracker::linear_assignment(const cv::Mat &cost, float cost_limit, Match &matches,
    std::vector<int> &mismatch_row, std::vector<int> &mismatch_col)
{
    matches.clear();
    mismatch_row.clear();
    mismatch_col.clear();
    if (cost.empty())
        return;
    
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
        std::vector<int>::iterator iter = find(db.begin(), db.end(), id);
        if (iter != db.end())
            piter = b.erase(piter);
        else
            ++piter;
    }
}

}   // namespace mot

#ifdef TEST_JDETRACKER_MODULE

#include <iostream>

struct Test {
    int a;
    float b[2];
};

typedef std::vector<Test *> TestPtrVector;

Test &operator+=(Test &x, const Test &y)
{
    x.a += y.a;
    x.b[0] += y.b[0];
    x.b[1] += y.b[1];
    return x;
}

int main()
{
#if 0
    mot::JDETracker::instance()->init();
    mot::JDETracker::instance()->free();
    
    float arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    cv::Mat mat(4, 4, CV_32F, arr);
    std::cout << mat << std::endl;
    const cv::Mat &ltrb_ = mat(cv::Rect(0, 1, 4, 1));
    cv::Vec4f ltrb = mat2vec4f(ltrb_);
    std::cout << ltrb << std::endl;
    std::cout << *mat.ptr<float>(2, 2) << std::endl;
    
    cv::Mat test(0, 1, CV_32F);
    float *ptr = test.ptr<float>(0);
    std::cout << test.empty() << std::endl;
    std::cout << ptr << std::endl;
#endif

    Test tests[5];
    for (int i = 0; i < 5; ++i)
    {
        tests[i].a = i;
        tests[i].b[0] = i + 1000;
        tests[i].b[1] = i + 2000;
    }
    
    TestPtrVector testss;
    for (int i = 0; i < 5; ++i)
        testss.push_back(&tests[i]);
    
    for (size_t i = 0; i < testss.size(); ++i)
    {
        fprintf(stderr, "%d %f %f\n", testss[i]->a, testss[i]->b[0], testss[i]->b[1]);
        testss[i]->a = i + 9999;
    }
    
    for (int i = 0; i < 5; ++i)
    {
        fprintf(stderr, "%d %f %f\n", tests[i].a, tests[i].b[0], tests[i].b[1]);
    }
    
    tests[0] += tests[4];
    for (int i = 0; i < 5; ++i)
    {
        fprintf(stderr, "%d %f %f\n", tests[i].a, tests[i].b[0], tests[i].b[1]);
    }
    
    return 0;
}

#endif