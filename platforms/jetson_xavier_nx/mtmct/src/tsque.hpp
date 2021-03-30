/********************************************************************
 *
 * @file tsque.h
 * @brief Thread safty queue.
 * @author CnybTseng
 * @version 1.0.0
 * @date March 10, 2021
 *
 *******************************************************************/

#ifndef _TSQUE_HPP
#define _TSQUE_HPP

#include <queue>
#include <mutex>
#include <condition_variable>

namespace algorithm {

/**
 * @brief class tsque. Thread safty queue.
 */
template <typename T>
class tsque
{
public:
    tsque() {}
    /**
     * @warning Must be implemented because std::mutex is not copyable.
     *  https://stackoverflow.com/questions/28311049/attempting-to-reference-a-deleted-function-when-using-a-mutex/28311106
     */
    tsque(const tsque &other)
    {
        std::lock_guard<std::mutex> lock(mtx);
        que = other.que;
    }
    /**
     * @warning Must be implemented because std::mutex is not copyable.
     */
    tsque &operator=(const tsque &rhs)
    {
        std::lock_guard<std::mutex> lock(mtx);
        que = rhs.que;
    }
    /**
     * @brief Push data in queue.
     * @param data The data to be pushed.
     */
    void push(const T &data)
    {
        std::lock_guard<std::mutex> lock(mtx);
        que.push(data);
        cv.notify_one();
    }
    /**
     * @brief Check if the queue is empty.
     * @return Returh true if the queue is empty, else return false.
     */
    bool empty() const
    {
        std::lock_guard<std::mutex> lock(mtx);
        return que.empty();
    }
    /**
     * @brief Non-block pop data from queue.
     * @param data The popped data.
     * @return Return true if pop success, else return false.
     */
    bool try_pop(T &data)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (que.empty()) {
            return false;
        }
        data = que.front();
        que.pop();
        return true;
    }
    /**
     * @brief Block pop data from queue.
     * @param data The popped data.
     */
    void wait_pop(T &data)
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]{return !que.empty();});    // wait until non-empty
        data = que.front();
        que.pop();
    }
private:
    mutable std::mutex mtx;
    std::condition_variable cv;
    std::queue<T> que;
};

}   // namespace algorithm

#endif  // _TSQUE_HPP