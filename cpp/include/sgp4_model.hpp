#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace sgp4 {

struct ElsetRec {
    double satnum = 0.0;
    double epochyr = 0.0;
    double epochdays = 0.0;
    double ndot = 0.0;
    double nddot = 0.0;
    double bstar = 0.0;
    double inclo = 0.0;
    double nodeo = 0.0;
    double ecco = 0.0;
    double argpo = 0.0;
    double mo = 0.0;
    double no_kozai = 0.0;

    int error = 0;
    std::string satname;

    double a = 0.0;
    double aldp = 0.0;
    double aycof = 0.0;
    double con41 = 0.0;
    double cc1 = 0.0;
    double cc3 = 0.0;
    double cc4 = 0.0;
    double cc5 = 0.0;
    double d2 = 0.0;
    double d3 = 0.0;
    double d4 = 0.0;
    double delmo = 0.0;
    double eta = 0.0;
    double argpdot = 0.0;
    double nodedot = 0.0;
    double omgcof = 0.0;
    double sinmao = 0.0;
    double t = 0.0;
    double t2cof = 0.0;
    double t3cof = 0.0;
    double t4cof = 0.0;
    double t5cof = 0.0;
    double x1mth2 = 0.0;
    double x7thm19 = 0.0;
    double mdot = 0.0;
    double xlcof = 0.0;

    int operationmode = 'a';
    int isimp = 0;
    int method = 0;

    double gsto = 0.0;
    double no_unkozai = 0.0;

    double irc3 = 0.0;
    double irc4 = 0.0;
    double itc = 0.0;
    double ihat = 0.0;
};

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct PropagationResult {
    Vec3 position;
    Vec3 velocity;
};

void sgp4_init(ElsetRec& rec);
PropagationResult sgp4_propagate(ElsetRec& rec, double tsince);

}
