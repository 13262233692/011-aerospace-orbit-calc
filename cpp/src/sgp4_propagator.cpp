#include "sgp4_propagator.hpp"
#include "sgp4_model.hpp"
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace sgp4 {

static constexpr double PI = 3.14159265358979323846;
static constexpr double DEG2RAD = PI / 180.0;
static constexpr double TWOPI = 2.0 * PI;
static constexpr double MIN_PER_DAY = 1440.0;

static double trim_double(const std::string& str) {
    size_t start = str.find_first_not_of(' ');
    if (start == std::string::npos) return 0.0;
    size_t end = str.find_last_not_of(' ');
    std::string trimmed = str.substr(start, end - start + 1);

    if (trimmed.find('.') == std::string::npos && trimmed.find('e') == std::string::npos) {
        try {
            return static_cast<double>(std::stoll(trimmed));
        } catch (...) {
            return 0.0;
        }
    }
    try {
        return std::stod(trimmed);
    } catch (...) {
        return 0.0;
    }
}

static double parse_power_of_ten(const std::string& str) {
    if (str.empty()) return 0.0;

    size_t start = str.find_first_not_of(' ');
    if (start == std::string::npos) return 0.0;
    size_t end = str.find_last_not_of(' ');
    std::string trimmed = str.substr(start, end - start + 1);

    if (trimmed.empty()) return 0.0;

    int sign_val = 1;
    size_t idx = 0;
    if (trimmed[0] == '-') {
        sign_val = -1;
        idx = 1;
    } else if (trimmed[0] == '+') {
        idx = 1;
    }

    if (idx >= trimmed.size()) return 0.0;

    double mantissa = 0.0;
    if (idx < trimmed.size() && trimmed[idx] == '.') {
        idx++;
        double frac = 0.1;
        while (idx < trimmed.size() && trimmed[idx] >= '0' && trimmed[idx] <= '9') {
            mantissa += (trimmed[idx] - '0') * frac;
            frac *= 0.1;
            idx++;
        }
    }

    if (idx >= trimmed.size()) return sign_val * mantissa;

    int exp_sign = 1;
    if (trimmed[idx] == '-') {
        exp_sign = -1;
        idx++;
    } else if (trimmed[idx] == '+') {
        idx++;
    }

    int exponent = 0;
    while (idx < trimmed.size() && trimmed[idx] >= '0' && trimmed[idx] <= '9') {
        exponent = exponent * 10 + (trimmed[idx] - '0');
        idx++;
    }

    return sign_val * mantissa * pow(10.0, exp_sign * exponent);
}

ElsetRec parse_tle(const std::string& line1, const std::string& line2) {
    ElsetRec rec;

    if (line1.size() < 69 || line2.size() < 69) {
        rec.error = 1;
        return rec;
    }

    if (line1[0] != '1' || line2[0] != '2') {
        rec.error = 2;
        return rec;
    }

    rec.satnum = trim_double(line1.substr(2, 5));
    rec.epochyr = trim_double(line1.substr(18, 2));
    rec.epochdays = trim_double(line1.substr(20, 12));
    rec.ndot = trim_double(line1.substr(33, 10));
    rec.nddot = parse_power_of_ten(line1.substr(44, 8));
    rec.bstar = parse_power_of_ten(line1.substr(53, 8));

    rec.inclo = trim_double(line2.substr(8, 8));
    rec.nodeo = trim_double(line2.substr(16, 8));
    rec.ecco = trim_double(line2.substr(26, 7)) * 1.0e-7;
    rec.argpo = trim_double(line2.substr(34, 8));
    rec.mo = trim_double(line2.substr(43, 8));
    rec.no_kozai = trim_double(line2.substr(52, 11)) * TWOPI / MIN_PER_DAY;

    rec.error = 0;

    sgp4_init(rec);

    return rec;
}

ElsetRec parse_tle_with_name(const std::string& name, const std::string& line1, const std::string& line2) {
    ElsetRec rec = parse_tle(line1, line2);
    rec.satname = name;
    return rec;
}

}
