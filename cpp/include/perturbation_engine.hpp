#pragma once

#define _USE_MATH_DEFINES
#include <atomic>
#include <vector>
#include <cstddef>
#include <memory>
#include <cstring>
#include <chrono>
#include <array>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace sgp4 {

struct PerturbationParams {
    double f10_7 = 150.0;
    double f10_7_avg = 150.0;
    double kp_index = 2.0;
    double kp_3h = 2.0;
    double atmospheric_density = 5.0e-12;
    double drag_coefficient = 2.2;
    double srp_coefficient = 1.0;
    double area_to_mass = 0.01;
    double storm_level = 0.0;
    double timestamp_jd = 0.0;
    double update_sequence = 0.0;
};

class LockFreePerturbationEngine {
public:
    static constexpr int PARAM_COUNT = 10;

    LockFreePerturbationEngine()
        : params0_{150.0, 150.0, 2.0, 2.0, 5.0e-12, 2.2, 1.0, 0.01, 0.0, 0.0},
          params1_{150.0, 150.0, 2.0, 2.0, 5.0e-12, 2.2, 1.0, 0.01, 0.0, 0.0} {
        active_buffer_.store(0, std::memory_order_release);
        sequence_.store(0, std::memory_order_release);
    }

    void update_param(int index, double value) {
        int current = active_buffer_.load(std::memory_order_acquire);
        int next = 1 - current;

        double* next_buf = (next == 0) ? params0_ : params1_;
        const double* cur_buf = (current == 0) ? params0_ : params1_;

        std::memcpy(next_buf, cur_buf, PARAM_COUNT * sizeof(double));
        next_buf[index] = value;

        uint64_t new_seq = sequence_.load(std::memory_order_acquire) + 1;
        next_buf[9] = static_cast<double>(new_seq);

        std::atomic_thread_fence(std::memory_order_release);
        active_buffer_.store(next, std::memory_order_release);
        sequence_.store(new_seq, std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    void update_all(const double* values) {
        int current = active_buffer_.load(std::memory_order_acquire);
        int next = 1 - current;

        double* next_buf = (next == 0) ? params0_ : params1_;
        std::memcpy(next_buf, values, PARAM_COUNT * sizeof(double));

        uint64_t new_seq = sequence_.load(std::memory_order_acquire) + 1;
        next_buf[9] = static_cast<double>(new_seq);

        std::atomic_thread_fence(std::memory_order_release);
        active_buffer_.store(next, std::memory_order_release);
        sequence_.store(new_seq, std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    double get_param(int index) const {
        int current = active_buffer_.load(std::memory_order_acquire);
        const double* buf = (current == 0) ? params0_ : params1_;
        std::atomic_thread_fence(std::memory_order_acquire);
        return buf[index];
    }

    void get_all(double* values) const {
        int current = active_buffer_.load(std::memory_order_acquire);
        const double* buf = (current == 0) ? params0_ : params1_;
        std::atomic_thread_fence(std::memory_order_acquire);
        std::memcpy(values, buf, PARAM_COUNT * sizeof(double));
    }

    uint64_t sequence() const {
        return sequence_.load(std::memory_order_acquire);
    }

    bool has_update(uint64_t last_seq) const {
        return sequence_.load(std::memory_order_acquire) > last_seq;
    }

    void apply_perturbation(double* x, double* y, double* z,
                            double* vx, double* vy, double* vz,
                            double jd, double dt) const {
        double params[PARAM_COUNT];
        get_all(params);

        double xv = *x, yv = *y, zv = *z;
        double vxv = *vx, vyv = *vy, vzv = *vz;

        double r = std::sqrt(xv*xv + yv*yv + zv*zv);
        if (r < 1e-10) return;

        double altitude = r - 6378.137;

        double rho = compute_atmospheric_density(altitude, params);

        double vx_ms = vxv * 1000.0;
        double vy_ms = vyv * 1000.0;
        double vz_ms = vzv * 1000.0;
        double v_rel_mag_ms = std::sqrt(vx_ms*vx_ms + vy_ms*vy_ms + vz_ms*vz_ms);

        double Cd = params[5];
        double Am = params[7];
        double Cr = params[6];

        double a_drag_mag_ms2 = 0.5 * Cd * Am * rho * v_rel_mag_ms * v_rel_mag_ms;
        double ax_ms = -a_drag_mag_ms2 * (vx_ms / v_rel_mag_ms);
        double ay_ms = -a_drag_mag_ms2 * (vy_ms / v_rel_mag_ms);
        double az_ms = -a_drag_mag_ms2 * (vz_ms / v_rel_mag_ms);

        double ax = ax_ms / 1000.0;
        double ay = ay_ms / 1000.0;
        double az = az_ms / 1000.0;

        double sun_theta = 2 * M_PI * (jd - 2451545.0) / 365.25;
        double sx_km = 149597870.7 * std::cos(sun_theta);
        double sy_km = 149597870.7 * std::sin(sun_theta);
        double sz_km = 0.0;

        double dx_km = sx_km - xv;
        double dy_km = sy_km - yv;
        double dz_km = sz_km - zv;
        double dmag_km = std::sqrt(dx_km*dx_km + dy_km*dy_km + dz_km*dz_km);

        if (dmag_km > 1e-10) {
            double cos_beta = -(xv*dx_km + yv*dy_km + zv*dz_km) / (r * dmag_km);
            cos_beta = cos_beta > 1.0 ? 1.0 : (cos_beta < -1.0 ? -1.0 : cos_beta);
            double beta = std::acos(cos_beta);
            double sun_ang = std::atan(696000.0 / (dmag_km * 1000.0));
            double earth_ang = std::atan(6378.137 / r);
            bool in_sunlight = beta < (earth_ang - sun_ang);
            if (in_sunlight) {
                double r_au = dmag_km / 149597870.7;
                double P = 4.56e-6 / (r_au * r_au);
                double f_srp = P * Cr * Am;
                ax += f_srp * (dx_km / dmag_km) / 1000.0;
                ay += f_srp * (dy_km / dmag_km) / 1000.0;
                az += f_srp * (dz_km / dmag_km) / 1000.0;
            }
        }

        double dvx = ax * dt;
        double dvy = ay * dt;
        double dvz = az * dt;

        *x = xv + 0.5 * ax * dt * dt;
        *y = yv + 0.5 * ay * dt * dt;
        *z = zv + 0.5 * az * dt * dt;
        *vx = vxv + dvx;
        *vy = vyv + dvy;
        *vz = vzv + dvz;
    }

    void apply_perturbation_batch(double* x, double* y, double* z,
                                   double* vx, double* vy, double* vz,
                                   const double* jd_array, double step_sec,
                                   size_t n) const {
        double params[PARAM_COUNT];
        get_all(params);

        double Cd = params[5];
        double Cr = params[6];
        double Am = params[7];

        for (size_t i = 0; i < n; ++i) {
            double xi = x[i], yi = y[i], zi = z[i];
            double vxi = vx[i], vyi = vy[i], vzi = vz[i];

            double r = std::sqrt(xi*xi + yi*yi + zi*zi);
            if (r < 1e-10) continue;

            double altitude = r - 6378.137;
            double rho = compute_atmospheric_density(altitude, params);

            double vxi_ms = vxi * 1000.0;
            double vyi_ms = vyi * 1000.0;
            double vzi_ms = vzi * 1000.0;
            double v_rel_mag_ms = std::sqrt(vxi_ms*vxi_ms + vyi_ms*vyi_ms + vzi_ms*vzi_ms);

            double a_drag_ms2 = 0.5 * Cd * Am * rho * v_rel_mag_ms * v_rel_mag_ms;
            double ax = -a_drag_ms2 * (vxi_ms / v_rel_mag_ms) / 1000.0;
            double ay = -a_drag_ms2 * (vyi_ms / v_rel_mag_ms) / 1000.0;
            double az = -a_drag_ms2 * (vzi_ms / v_rel_mag_ms) / 1000.0;

            double jd = jd_array[i];
            double sun_theta = 2 * M_PI * (jd - 2451545.0) / 365.25;
            double sx_km = 149597870.7 * std::cos(sun_theta);
            double sy_km = 149597870.7 * std::sin(sun_theta);
            double sz_km = 0.0;

            double dx_km = sx_km - xi, dy_km = sy_km - yi, dz_s_km = sz_km - zi;
            double dmag_km = std::sqrt(dx_km*dx_km + dy_km*dy_km + dz_s_km*dz_s_km);

            if (dmag_km > 1e-10) {
                double cos_beta = -(xi*dx_km + yi*dy_km + zi*dz_s_km) / (r * dmag_km);
                cos_beta = cos_beta > 1.0 ? 1.0 : (cos_beta < -1.0 ? -1.0 : cos_beta);
                double beta = std::acos(cos_beta);
                double sun_ang = std::atan(696000.0 / (dmag_km * 1000.0));
                double earth_ang = std::atan(6378.137 / r);
                bool in_sunlight = beta < (earth_ang - sun_ang);
                if (in_sunlight) {
                    double r_au = dmag_km / 149597870.7;
                    double P = 4.56e-6 / (r_au * r_au);
                    double f_srp = P * Cr * Am;
                    ax += f_srp * (dx_km / dmag_km) / 1000.0;
                    ay += f_srp * (dy_km / dmag_km) / 1000.0;
                    az += f_srp * (dz_s_km / dmag_km) / 1000.0;
                }
            }

            double dvx = ax * step_sec;
            double dvy = ay * step_sec;
            double dvz = az * step_sec;

            x[i] = xi + 0.5 * ax * step_sec * step_sec;
            y[i] = yi + 0.5 * ay * step_sec * step_sec;
            z[i] = zi + 0.5 * az * step_sec * step_sec;
            vx[i] = vxi + dvx;
            vy[i] = vyi + dvy;
            vz[i] = vzi + dvz;
        }
    }

private:
    static double compute_atmospheric_density(double altitude_km, const double* params) {
        double f10 = params[0];
        double f10_bar = params[1];
        double kp = params[2];
        double storm = params[8];

        double f10_eff = 0.5 * (f10 + f10_bar);
        if (f10_eff < 65.0) f10_eff = 65.0;

        double T_inf = 600.0 + 0.4 * (f10 - 70.0) + 0.025 * (f10_bar - 70.0) + 1.8 * kp;
        double H = 1.7 + 0.02 * T_inf;

        double f10_factor = (f10_eff / 150.0);
        if (f10_factor < 0.3) f10_factor = 0.3;
        if (f10_factor > 3.0) f10_factor = 3.0;

        double rho;
        if (altitude_km < 200.0) {
            double rho_200 = 2.5e-10 * f10_factor;
            double d = 200.0 - altitude_km;
            rho = rho_200 * std::exp(d / 30.0);
        } else if (altitude_km < 350.0) {
            double rho_200 = 2.5e-10 * f10_factor;
            double rho_350 = 3.5e-11 * f10_factor;
            double alpha = (altitude_km - 200.0) / 150.0;
            double log_rho = (1.0 - alpha) * std::log(rho_200) + alpha * std::log(rho_350);
            rho = std::exp(log_rho);
        } else if (altitude_km < 500.0) {
            double rho_350 = 3.5e-11 * f10_factor;
            double rho_500 = 2.2e-12 * f10_factor;
            double alpha = (altitude_km - 350.0) / 150.0;
            double log_rho = (1.0 - alpha) * std::log(rho_350) + alpha * std::log(rho_500);
            rho = std::exp(log_rho);
        } else if (altitude_km < 800.0) {
            double rho_500 = 2.2e-12 * f10_factor;
            double rho_800 = 3.0e-14 * f10_factor;
            double alpha = (altitude_km - 500.0) / 300.0;
            double log_rho = (1.0 - alpha) * std::log(rho_500) + alpha * std::log(rho_800);
            rho = std::exp(log_rho);
        } else {
            double rho_800 = 3.0e-14 * f10_factor;
            rho = rho_800 * std::exp(-(altitude_km - 800.0) / (H * 1.5));
        }

        double kp_factor = 1.0 + 0.3 * (kp - 2.0) * std::exp(-std::abs(altitude_km - 400.0) / 150.0);
        if (kp_factor < 0.4) kp_factor = 0.4;
        if (kp_factor > 20.0) kp_factor = 20.0;
        rho *= kp_factor;

        if (storm >= 0.5) {
            double storm_factor = 1.0 + 5.0 * (storm - 0.5) * std::exp(-std::abs(altitude_km - 400.0) / 200.0);
            if (storm_factor < 1.0) storm_factor = 1.0;
            if (storm_factor > 100.0) storm_factor = 100.0;
            rho *= storm_factor;
        }

        if (rho < 1e-22) rho = 1e-22;

        return rho;
    }

    double params0_[PARAM_COUNT];
    double params1_[PARAM_COUNT];
    std::atomic<int> active_buffer_;
    std::atomic<uint64_t> sequence_;
};

class BatchPerturbationEngine {
public:
    using Ptr = std::shared_ptr<BatchPerturbationEngine>;

    BatchPerturbationEngine(size_t num_satellites = 1)
        : num_satellites_(num_satellites) {
        engines_.reserve(num_satellites);
        for (size_t i = 0; i < num_satellites; ++i) {
            engines_.emplace_back(std::make_unique<LockFreePerturbationEngine>());
        }
    }

    LockFreePerturbationEngine& operator[](size_t index) {
        return *engines_[index];
    }

    const LockFreePerturbationEngine& operator[](size_t index) const {
        return *engines_[index];
    }

    size_t size() const { return engines_.size(); }

    void resize(size_t new_size) {
        while (engines_.size() < new_size) {
            engines_.emplace_back(std::make_unique<LockFreePerturbationEngine>());
        }
        num_satellites_ = new_size;
    }

    void update_global_env(double f10_7, double f10_7_avg, double kp, double storm_level) {
        double params[LockFreePerturbationEngine::PARAM_COUNT];
        for (auto& engine : engines_) {
            engine->get_all(params);
            params[0] = f10_7;
            params[1] = f10_7_avg;
            params[2] = kp;
            params[3] = kp;
            params[8] = storm_level;
            engine->update_all(params);
        }
    }

private:
    size_t num_satellites_;
    std::vector<std::unique_ptr<LockFreePerturbationEngine>> engines_;
};

}
