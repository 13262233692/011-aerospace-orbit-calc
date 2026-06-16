#pragma once

#include <vector>
#include <cstddef>

namespace sgp4 {

struct WGS84Position {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
};

struct ECIPosition {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
};

WGS84Position eci_to_wgs84(const ECIPosition& eci, double jd);

struct BatchECEFResult {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> vx;
    std::vector<double> vy;
    std::vector<double> vz;
};

BatchECEFResult eci_to_ecef_batch(
    const double* eci_x, const double* eci_y, const double* eci_z,
    const double* eci_vx, const double* eci_vy, const double* eci_vz,
    const double* jd_array,
    size_t n
);

}
