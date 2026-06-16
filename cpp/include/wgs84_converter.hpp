#pragma once

#include <vector>
#include <cstddef>
#include <memory>

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

class BatchECEFBuffer {
public:
    using Ptr = std::shared_ptr<BatchECEFBuffer>;

    BatchECEFBuffer() = default;

    explicit BatchECEFBuffer(size_t n)
        : size_(n), x_(new double[n]), y_(new double[n]), z_(new double[n]),
          vx_(new double[n]), vy_(new double[n]), vz_(new double[n]) {}

    ~BatchECEFBuffer() = default;

    size_t size() const { return size_; }

    double* x() { return x_.get(); }
    double* y() { return y_.get(); }
    double* z() { return z_.get(); }
    double* vx() { return vx_.get(); }
    double* vy() { return vy_.get(); }
    double* vz() { return vz_.get(); }

    const double* x() const { return x_.get(); }
    const double* y() const { return y_.get(); }
    const double* z() const { return z_.get(); }
    const double* vx() const { return vx_.get(); }
    const double* vy() const { return vy_.get(); }
    const double* vz() const { return vz_.get(); }

    void resize(size_t n) {
        size_ = n;
        x_.reset(new double[n]);
        y_.reset(new double[n]);
        z_.reset(new double[n]);
        vx_.reset(new double[n]);
        vy_.reset(new double[n]);
        vz_.reset(new double[n]);
    }

    void clear() {
        size_ = 0;
        x_.reset();
        y_.reset();
        z_.reset();
        vx_.reset();
        vy_.reset();
        vz_.reset();
    }

    bool empty() const { return size_ == 0; }

private:
    size_t size_ = 0;
    std::unique_ptr<double[]> x_;
    std::unique_ptr<double[]> y_;
    std::unique_ptr<double[]> z_;
    std::unique_ptr<double[]> vx_;
    std::unique_ptr<double[]> vy_;
    std::unique_ptr<double[]> vz_;
};

BatchECEFResult eci_to_ecef_batch(
    const double* eci_x, const double* eci_y, const double* eci_z,
    const double* eci_vx, const double* eci_vy, const double* eci_vz,
    const double* jd_array,
    size_t n
);

BatchECEFBuffer::Ptr eci_to_ecef_buffer(
    const double* eci_x, const double* eci_y, const double* eci_z,
    const double* eci_vx, const double* eci_vy, const double* eci_vz,
    const double* jd_array,
    size_t n
);

}

