#pragma once

#include "sgp4_model.hpp"
#include <string>
#include <vector>

namespace sgp4 {

struct TLEData {
    std::string satellite_name;
    std::string line1;
    std::string line2;
};

ElsetRec parse_tle(const std::string& line1, const std::string& line2);
ElsetRec parse_tle_with_name(const std::string& name, const std::string& line1, const std::string& line2);

}
