#ifndef JDETRACKER_H
#define JDETRACKER_H

namespace mot {

class JDETracker
{
public:
    JDETracker();
    ~JDETracker();
    int update(float *dets, int dim, int cnt);
};

}   // namespace mot

#endif  // JDETRACKER_H