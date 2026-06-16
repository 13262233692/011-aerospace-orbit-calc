#pragma once

#include "sgp4_model.hpp"
#include "wgs84_converter.hpp"
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace sgp4 {

struct TimedPosition {
    double timestamp_epoch = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
};

struct BatchResult {
    std::string satellite_id;
    double epoch_jd = 0.0;
    std::vector<TimedPosition> positions;
};

BatchResult propagate_7days(const ElsetRec& rec, int step_seconds = 1);

}
