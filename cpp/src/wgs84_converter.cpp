#include "wgs84_converter.hpp"
#include <cmath>

namespace sgp4 {

static const double PI = 3.14159265358979323846;
static const double OMEGA_EARTH = 7.2921151467e-5;

static double greenwich_apparent_sidereal_time(double jd) {
    double tut1 = (jd - 2451545.0) / 36525.0;

    double gmst_sec = 67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * tut1 +
                      0.093104 * tut1 * tut1 - 6.2e-6 * tut1 * tut1 * tut1;
    gmst_sec = fmod(gmst_sec, 86400.0);
    if (gmst_sec < 0) gmst_sec += 86400.0;

    double gmst_rad = gmst_sec * PI / 43200.0;

    double t = tut1;
    double omega_deg = 125.04452 - 1934.136261 * t + 0.0020708 * t * t + t * t * t / 450000.0;
    double omega = omega_deg * PI / 180.0;
    double dpsi_deg = -17.2 * sin(omega) / 3600.0;
    double dpsi = dpsi_deg * PI / 180.0;

    double eps0_arcsec = 84381.448 - 46.8150 * t - 0.00059 * t * t + 0.001813 * t * t * t;
    double epsilon = (eps0_arcsec / 3600.0) * PI / 180.0;

    double gast = gmst_rad + dpsi * cos(epsilon);
    gast = fmod(gast, 2.0 * PI);
    if (gast < 0) gast += 2.0 * PI;

    return gast;
}

WGS84Position eci_to_wgs84(const ECIPosition& eci, double jd) {
    double theta = greenwich_apparent_sidereal_time(jd);
    double ct = cos(theta);
    double st = sin(theta);

    WGS84Position ecef;
    ecef.x = eci.x * ct + eci.y * st;
    ecef.y = -eci.x * st + eci.y * ct;
    ecef.z = eci.z;

    ecef.vx = eci.vx * ct + eci.vy * st + OMEGA_EARTH * ecef.y;
    ecef.vy = -eci.vx * st + eci.vy * ct - OMEGA_EARTH * ecef.x;
    ecef.vz = eci.vz;

    return ecef;
}

BatchECEFResult eci_to_ecef_batch(
    const double* eci_x, const double* eci_y, const double* eci_z,
    const double* eci_vx, const double* eci_vy, const double* eci_vz,
    const double* jd_array,
    size_t n
) {
    BatchECEFResult result;
    result.x.resize(n);
    result.y.resize(n);
    result.z.resize(n);
    result.vx.resize(n);
    result.vy.resize(n);
    result.vz.resize(n);

    for (size_t i = 0; i < n; ++i) {
        double theta = greenwich_apparent_sidereal_time(jd_array[i]);
        double ct = cos(theta);
        double st = sin(theta);

        double xi = eci_x[i], yi = eci_y[i], zi = eci_z[i];
        double vxi = eci_vx[i], vyi = eci_vy[i], vzi = eci_vz[i];

        result.x[i] = ct * xi + st * yi;
        result.y[i] = -st * xi + ct * yi;
        result.z[i] = zi;

        result.vx[i] = ct * vxi + st * vyi + OMEGA_EARTH * result.y[i];
        result.vy[i] = -st * vxi + ct * vyi - OMEGA_EARTH * result.x[i];
        result.vz[i] = vzi;
    }

    return result;
}

BatchECEFBuffer::Ptr eci_to_ecef_buffer(
    const double* eci_x, const double* eci_y, const double* eci_z,
    const double* eci_vx, const double* eci_vy, const double* eci_vz,
    const double* jd_array,
    size_t n
) {
    auto buf = std::make_shared<BatchECEFBuffer>(n);
    double* x = buf->x();
    double* y = buf->y();
    double* z = buf->z();
    double* vx = buf->vx();
    double* vy = buf->vy();
    double* vz = buf->vz();

    for (size_t i = 0; i < n; ++i) {
        double theta = greenwich_apparent_sidereal_time(jd_array[i]);
        double ct = cos(theta);
        double st = sin(theta);

        double xi = eci_x[i], yi = eci_y[i], zi = eci_z[i];
        double vxi = eci_vx[i], vyi = eci_vy[i], vzi = eci_vz[i];

        x[i] = ct * xi + st * yi;
        y[i] = -st * xi + ct * yi;
        z[i] = zi;

        vx[i] = ct * vxi + st * vyi + OMEGA_EARTH * y[i];
        vy[i] = -st * vxi + ct * vyi - OMEGA_EARTH * x[i];
        vz[i] = vzi;
    }

    return buf;
}

}
