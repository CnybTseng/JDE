#include "config.h"

namespace mot {

std::vector<std::string> categories = {"person"};

std::map<std::string, std::pair<int, int>> arch = {
    {"stage2", std::pair<int, int>(58, 116)},
    {"stage3", std::pair<int, int>(116, 232)},
    {"stage4", std::pair<int, int>(232, 464)}
};

}   // namespace mot