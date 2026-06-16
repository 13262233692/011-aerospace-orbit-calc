#include "sgp4_model.hpp"
#include <cmath>

namespace sgp4 {

static constexpr double PI = 3.14159265358979323846;
static constexpr double TWOPI = 2.0 * PI;
static constexpr double DEG2RAD = PI / 180.0;
static constexpr double RE = 6378.135;
static constexpr double AE = 1.0;
static constexpr double XKE = 0.07436685316871385;
static constexpr double J2 = 1.082616e-3;
static constexpr double J3 = -2.53881e-6;
static constexpr double J4 = -1.65597e-6;
static constexpr double CK2 = 0.5 * J2;
static constexpr double CK4 = -0.375 * J4;
static const double A3OVK2 = -J3 / CK2;
static const double VKMPS = RE * XKE / 60.0;
static constexpr double QZMS2T = 1.880279159015270643865e-9;
static constexpr double S4 = 1.012229280190835e-2;
static constexpr double E6A = 1.0e-6;

static double fmod2p(double x) {
    double result = fmod(x, TWOPI);
    if (result < 0.0) result += TWOPI;
    return result;
}

void sgp4_init(ElsetRec& satrec) {
    double cosio = cos(satrec.inclo * DEG2RAD);
    double sinio = sin(satrec.inclo * DEG2RAD);
    double eosq = satrec.ecco * satrec.ecco;
    double betao2 = 1.0 - eosq;
    double betao = sqrt(betao2);
    double theta2 = cosio * cosio;

    double a1 = pow(XKE / satrec.no_kozai, 2.0 / 3.0);
    double del1 = 1.5 * CK2 * (3.0 * theta2 - 1.0) / (a1 * a1 * betao * betao2);
    double ao = a1 * (1.0 - del1 * (1.0 / 3.0 + del1 * (1.0 + 134.0 / 81.0 * del1)));
    double delo = 1.5 * CK2 * (3.0 * theta2 - 1.0) / (ao * ao * betao * betao2);
    double xnodp = satrec.no_kozai / (1.0 + delo);
    double aodp = ao / (1.0 - delo);

    satrec.a = aodp;
    satrec.no_unkozai = xnodp;

    double s4 = S4;
    double qzms24 = QZMS2T;
    double perige = (aodp * (1.0 - satrec.ecco) - AE) * RE;

    if (perige < 156.0) {
        s4 = perige - 78.0;
        if (perige <= 98.0) s4 = 20.0;
        qzms24 = pow((120.0 - s4) / RE, 4.0);
        s4 = s4 / RE + AE;
    }

    double tsi = 1.0 / (aodp - s4);
    satrec.eta = aodp * satrec.ecco * tsi;
    double etasq = satrec.eta * satrec.eta;
    double eeta = satrec.ecco * satrec.eta;
    double psisq = fabs(1.0 - etasq);
    double coef = qzms24 * pow(tsi, 4.0);
    double coef1 = coef / pow(psisq, 3.5);

    double c2 = coef1 * xnodp * (aodp * (1.0 + 1.5 * etasq + eeta * (4.0 + etasq)) +
                0.75 * CK2 * tsi / psisq * (-2.0 + 3.0 * theta2) *
                (8.0 + 3.0 * etasq * (8.0 + etasq)));

    satrec.cc1 = satrec.bstar * c2;
    satrec.cc4 = 2.0 * xnodp * coef1 * aodp * betao2 *
                 (satrec.eta * (2.0 + 0.5 * etasq) +
                  satrec.ecco * (0.5 + 2.0 * etasq));
    satrec.cc5 = 2.0 * coef1 * aodp * betao2 * (1.0 + 2.75 * etasq + eeta);

    double c3 = 0.0;
    if (satrec.ecco > E6A) {
        c3 = coef * tsi * A3OVK2 * xnodp * AE *
             sin(satrec.argpo * DEG2RAD) / satrec.ecco;
    }
    satrec.cc3 = c3;

    satrec.xlcof = 0.125 * A3OVK2 * sinio / (1.0 + cosio);
    satrec.aycof = 0.25 * A3OVK2 * sinio;
    satrec.x1mth2 = 1.0 - theta2;
    satrec.x7thm19 = 7.0 * theta2 - 1.0;

    double pinvsq = 1.0 / (aodp * aodp * betao2 * betao2);
    double argpm_per_min = -0.5 * J2 * (7.0 * theta2 - 1.0) * pinvsq * xnodp;
    double nodedm_per_min = -1.5 * J2 * cosio * pinvsq * xnodp;
    double mdot_per_min = xnodp + 0.75 * J2 * (3.0 * theta2 - 1.0) * pinvsq * xnodp;

    satrec.mdot = mdot_per_min;
    satrec.argpdot = argpm_per_min;
    satrec.nodedot = nodedm_per_min;

    satrec.omgcof = 0.0;
    if (satrec.ecco > E6A) {
        satrec.omgcof = -0.5 * A3OVK2 * sinio / (1.0 + cosio);
    }

    double xmcof = 0.0;
    if (satrec.ecco > E6A) {
        xmcof = -2.0 / 3.0 * A3OVK2 * sinio / (1.0 + cosio);
    }
    satrec.con41 = xmcof;

    satrec.t2cof = 1.5 * J2 * (3.0 * theta2 - 1.0) * pinvsq;
    satrec.t3cof = J2 * (13.0 * theta2 - 1.0) / (2.0 * aodp * aodp * betao2 * betao2);
    satrec.t4cof = 0.375 * J2 * (7.0 * theta2 - 1.0) * pinvsq;
    satrec.t5cof = 0.0625 * J2 * (3.0 * theta2 - 1.0) * pinvsq;

    satrec.isimp = 0;
    if ((aodp * (1.0 - satrec.ecco) - AE) * RE < 220.0) {
        satrec.isimp = 1;
    }

    double d2 = 0.0;
    double d3 = 0.0;
    double d4 = 0.0;
    if (satrec.isimp == 0) {
        d2 = 4.0 * aodp * tsi * satrec.cc1 / xnodp;
        d3 = 4.0 / 3.0 * aodp * tsi * tsi * satrec.cc1 * (17.0 * aodp + s4) / xnodp;
        d4 = 2.0 / 3.0 * aodp * tsi * tsi * tsi * satrec.cc1 *
             (221.0 * aodp + 31.0 * s4) * satrec.cc1 / xnodp;
    }
    satrec.d2 = d2;
    satrec.d3 = d3;
    satrec.d4 = d4;

    satrec.sinmao = sin(satrec.mo * DEG2RAD);
    satrec.delmo = pow(1.0 + satrec.eta * cos(satrec.mo * DEG2RAD), 3.0);
}

PropagationResult sgp4_propagate(ElsetRec& satrec, double tsince) {
    PropagationResult result{};

    double xmdf = satrec.mo * DEG2RAD + satrec.mdot * tsince;
    double argpdf = satrec.argpo * DEG2RAD + satrec.argpdot * tsince;
    double nodedf = satrec.nodeo * DEG2RAD + satrec.nodedot * tsince;

    double argpm = argpdf;
    double nodem = nodedf;
    double mm = xmdf;
    double em = satrec.ecco;
    double inclm = satrec.inclo * DEG2RAD;
    double nm = satrec.no_unkozai;

    if (satrec.isimp != 1) {
        double delm = satrec.xlcof * satrec.sinmao;
        double dls = satrec.omgcof * sin(satrec.argpo * DEG2RAD +
                     satrec.mdot * tsince - satrec.argpdot * tsince);
        mm = xmdf + delm + dls;
        argpm = argpdf - dls;
        nm = satrec.no_unkozai;
        double tsel = tsince;
        double dndt = satrec.cc1 * sin(satrec.sinmao) +
                      satrec.cc3 * sin(2.0 * satrec.sinmao);
        nm = nm + dndt;
        double t2 = tsel * tsel;
        mm = mm + nm * t2 * (satrec.t2cof + tsel * (satrec.t3cof +
             tsel * (satrec.t4cof + tsel * satrec.t5cof)));
        nm = nm + satrec.cc4 * tsel + satrec.cc5 * tsel * tsel;
    }

    mm = fmod2p(mm);
    double xlm = mm + argpm + nodem;
    nodem = fmod2p(nodem);

    double ep = em;
    double xincl = inclm;
    double xnode = nodem;
    double omega = argpm;
    double xn = nm;

    if (xn <= 0.0) xn = satrec.no_unkozai;

    double sinip = sin(xincl);
    double cosip = cos(xincl);

    double a = pow(XKE / xn, 2.0 / 3.0);
    double pl = a * (1.0 - ep * ep);
    double el = ep;

    double xnodep = xnode;
    double argpp = omega;
    double mp = xlm - argpp - xnodep;

    double axnl = el * cos(argpp);
    double aynl = el * sin(argpp) - satrec.aycof;
    double xl = mp + argpp + xnodep + satrec.xlcof * axnl;

    double u = xl - xnodep;
    double eo1 = mp + axnl * sin(u) - aynl * cos(u);

    double sineo1, coseo1;
    for (int i = 0; i < 10; i++) {
        sineo1 = sin(eo1);
        coseo1 = cos(eo1);
        double tem5 = 1.0 - coseo1 * axnl - sineo1 * aynl;
        if (fabs(tem5) < E6A) break;
        double eo2 = eo1 + (mp - axnl * sineo1 + aynl * coseo1 - eo1) / tem5;
        if (fabs(eo2 - eo1) <= E6A) { eo1 = eo2; break; }
        eo1 = eo2;
    }

    double sl = sin(eo1);
    double cl = cos(eo1);

    double capu = fmod2p(mp - axnl * sl + aynl * cl);
    double su = sin(capu);
    double cu = cos(capu);

    double su2 = su - aynl;
    double cu2 = cu - axnl;

    double p = cu2 * cu2 + su2 * su2;
    double r = a * (1.0 - cl * axnl - sl * aynl);

    if (p < E6A) p = E6A;
    double sqrtp = sqrt(p);

    double xnprime = XKE / pow(a, 1.5);
    double rdot = xnprime * (axnl * su2 - aynl * cu2) / p;
    double rvdot = xnprime * sqrtp / p;

    double sinu = su2 / sqrtp;
    double cosu = cu2 / sqrtp;

    double sin2u = 2.0 * sinu * cosu;
    double cos2u = 2.0 * cosu * cosu - 1.0;

    double rk = r + 1.5 * CK2 * sinip * sin2u / p;
    double uk = atan2(sinu, cosu) + 0.25 * CK2 * sinip * cos2u / p;
    double xnodek = xnodep - 1.5 * CK2 * cosip * sin2u / p;
    double xinck = xincl + 1.5 * CK2 * cosip * sinip * cos2u / p;

    double rdotk = rdot - 1.5 * CK2 * sinip * sin2u / p;
    double rvdotk = rvdot + 1.5 * CK2 * (cosip - 0.5 * sinip * sinip) * sin2u / p;

    double sinuk = sin(uk);
    double cosuk = cos(uk);
    double sinik = sin(xinck);
    double cosik = cos(xinck);
    double sinnok = sin(xnodek);
    double cosnok = cos(xnodek);

    double xmx = -sinnok * cosik;
    double xmy = cosnok * cosik;
    double ux = xmx * sinuk + cosnok * cosuk;
    double uy = xmy * sinuk + sinnok * cosuk;
    double uz = sinik * sinuk;

    double vx = xmx * cosuk - cosnok * sinuk;
    double vy = xmy * cosuk - sinnok * sinuk;
    double vz = sinik * cosuk;

    result.position.x = rk * ux * RE;
    result.position.y = rk * uy * RE;
    result.position.z = rk * uz * RE;

    result.velocity.x = (rdotk * ux + rvdotk * vx) * VKMPS;
    result.velocity.y = (rdotk * uy + rvdotk * vy) * VKMPS;
    result.velocity.z = (rdotk * uz + rvdotk * vz) * VKMPS;

    return result;
}

}
