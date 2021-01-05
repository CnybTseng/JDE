#ifndef CONFIG_H_
#define CONFIG_H_

#include <map>
#include <vector>
#include <string>

namespace mot {

constexpr float conf_thresh = 0.5f;
constexpr float iou_thresh = 0.4f;
constexpr int   trajectory_len = 1000;
constexpr int   max_lost_time = 30;
extern std::vector<std::string> categories;
extern std::map<std::string, std::pair<int, int>> arch;
constexpr float motion_cost_limit = 1.0f;
constexpr float iou_cost_limit1 = 1.0f;
constexpr float iou_cost_limit2 = 1.0f;

}   // namespace mot

#endif