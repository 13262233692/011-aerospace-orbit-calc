#include "batch_propagator.hpp"
#include "sgp4_propagator.hpp"
#include <cmath>

namespace sgp4 {

static double epoch_to_jd(double epochyr, double epochdays) {
    double year = epochyr;
    if (year < 57.0) {
        year += 2000.0;
    } else {
        year += 1900.0;
    }
    double jd_year_start = 367.0 * year - floor(7.0 * (year + floor((floor(10.0 + 9.0) / 12.0))) / 4.0) +
                           floor(275.0 * 1.0 / 9.0) + 1721013.5;
    jd_year_start = 367.0 * year - floor(7.0 * (year + 9.0 / 12.0) / 4.0) +
                    floor(275.0 * 1.0 / 9.0) + 1721013.5;

    return jd_year_start + epochdays;
}

BatchResult propagate_7days(const ElsetRec& rec, int step_seconds) {
    BatchResult result;
    result.satellite_id = rec.satname.empty() ?
                          std::to_string(static_cast<int64_t>(rec.satnum)) :
                          rec.satname;

    result.epoch_jd = epoch_to_jd(rec.epochyr, rec.epochdays);

    ElsetRec work_rec = rec;

    int total_seconds = 7 * 24 * 3600;
    int num_steps = total_seconds / step_seconds + 1;

    result.positions.reserve(num_steps);

    for (int t = 0; t <= total_seconds; t += step_seconds) {
        double tsince = static_cast<double>(t);

        PropagationResult prop = sgp4_propagate(work_rec, tsince / 60.0);

        ECIPosition eci;
        eci.x = prop.position.x;
        eci.y = prop.position.y;
        eci.z = prop.position.z;
        eci.vx = prop.velocity.x;
        eci.vy = prop.velocity.y;
        eci.vz = prop.velocity.z;

        double jd_current = result.epoch_jd + tsince / 86400.0;

        WGS84Position wgs84 = eci_to_wgs84(eci, jd_current);

        TimedPosition pos;
        pos.timestamp_epoch = jd_current;
        pos.x = wgs84.x;
        pos.y = wgs84.y;
        pos.z = wgs84.z;
        pos.vx = wgs84.vx;
        pos.vy = wgs84.vy;
        pos.vz = wgs84.vz;

        result.positions.push_back(pos);
    }

    return result;
}

}
