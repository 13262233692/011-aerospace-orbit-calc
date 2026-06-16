from sgp4.api import Satrec, WGS72, jday
import math
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime, timezone, timedelta

OMEGA_EARTH = 7.2921151467e-5

_CPP_ACCEL_AVAILABLE = False
try:
    from orbit_calc._sgp4_binding import eci_to_ecef_batch_cpp
    _CPP_ACCEL_AVAILABLE = True
except ImportError:
    pass


class SGP4Propagator:
    def __init__(self, use_cpp_accel: bool = True):
        self.gravity_model = WGS72
        self.use_cpp_accel = use_cpp_accel and _CPP_ACCEL_AVAILABLE

    def parse_tle(self, line1: str, line2: str) -> Satrec:
        return Satrec.twoline2rv(line1, line2, self.gravity_model)

    def parse_tle_with_name(self, name: str, line1: str, line2: str) -> Tuple[str, Satrec]:
        sat = Satrec.twoline2rv(line1, line2, self.gravity_model)
        return (name, sat)

    def propagate_single(self, sat: Satrec, year: int, month: int,
                         day: int, hour: int, minute: int, second: float) -> dict:
        jd, fr = jday(year, month, day, hour, minute, second)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            return None
        return {
            "timestamp_jd": jd + fr,
            "eci_x": r[0], "eci_y": r[1], "eci_z": r[2],
            "eci_vx": v[0], "eci_vy": v[1], "eci_vz": v[2],
        }

    @staticmethod
    def eci_to_ecef(x_eci, y_eci, z_eci, vx_eci, vy_eci, vz_eci, jd: float) -> dict:
        theta_gast = _greenwich_apparent_sidereal_time(jd)
        cos_t = math.cos(theta_gast)
        sin_t = math.sin(theta_gast)

        x_ecef = cos_t * x_eci + sin_t * y_eci
        y_ecef = -sin_t * x_eci + cos_t * y_eci
        z_ecef = z_eci

        vx_ecef = cos_t * vx_eci + sin_t * vy_eci + OMEGA_EARTH * y_ecef
        vy_ecef = -sin_t * vx_eci + cos_t * vy_eci - OMEGA_EARTH * x_ecef
        vz_ecef = vz_eci

        return {
            "x": x_ecef, "y": y_ecef, "z": z_ecef,
            "vx": vx_ecef, "vy": vy_ecef, "vz": vz_ecef,
        }

    @staticmethod
    def eci_to_ecef_batch(r_eci: np.ndarray, v_eci: np.ndarray,
                          jd_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        theta = _greenwich_apparent_sidereal_time_array(jd_array)
        ct = np.cos(theta)
        st = np.sin(theta)

        r_ecef = np.empty_like(r_eci)
        v_ecef = np.empty_like(v_eci)

        r_ecef[:, 0] = ct * r_eci[:, 0] + st * r_eci[:, 1]
        r_ecef[:, 1] = -st * r_eci[:, 0] + ct * r_eci[:, 1]
        r_ecef[:, 2] = r_eci[:, 2]

        v_ecef[:, 0] = ct * v_eci[:, 0] + st * v_eci[:, 1] + OMEGA_EARTH * r_ecef[:, 1]
        v_ecef[:, 1] = -st * v_eci[:, 0] + ct * v_eci[:, 1] - OMEGA_EARTH * r_ecef[:, 0]
        v_ecef[:, 2] = v_eci[:, 2]

        return r_ecef, v_ecef

    @staticmethod
    def _eci_to_ecef_cpp(r_eci: np.ndarray, v_eci: np.ndarray,
                         jd_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cpp_result = eci_to_ecef_batch_cpp(
            r_eci[:, 0], r_eci[:, 1], r_eci[:, 2],
            v_eci[:, 0], v_eci[:, 1], v_eci[:, 2],
            jd_array
        )
        r_ecef = np.column_stack([
            np.asarray(cpp_result['x']),
            np.asarray(cpp_result['y']),
            np.asarray(cpp_result['z']),
        ])
        v_ecef = np.column_stack([
            np.asarray(cpp_result['vx']),
            np.asarray(cpp_result['vy']),
            np.asarray(cpp_result['vz']),
        ])
        return r_ecef, v_ecef

    def propagate_7days(self, sat: Satrec, epochyr: int, epochdays: float,
                        step_seconds: int = 1) -> List[dict]:
        if epochyr < 57:
            year = 2000 + epochyr
        else:
            year = 1900 + epochyr

        jd_epoch = self._year_day_to_jd(year, epochdays)

        total_seconds = 7 * 24 * 3600
        results = []

        for t in range(0, total_seconds + 1, step_seconds):
            jd_current = jd_epoch + t / 86400.0
            jd_int = int(jd_current)
            jd_frac = jd_current - jd_int

            e, r, v = sat.sgp4(jd_int, jd_frac)
            if e != 0:
                continue

            ecef = self.eci_to_ecef(
                r[0], r[1], r[2], v[0], v[1], v[2], jd_current
            )

            results.append({
                "timestamp_jd": jd_current,
                "x": ecef["x"], "y": ecef["y"], "z": ecef["z"],
                "vx": ecef["vx"], "vy": ecef["vy"], "vz": ecef["vz"],
            })

        return results

    def propagate_7days_fast(self, sat: Satrec, epochyr: int, epochdays: float,
                             step_seconds: int = 1) -> dict:
        if epochyr < 57:
            year = 2000 + epochyr
        else:
            year = 1900 + epochyr

        jd_epoch = self._year_day_to_jd(year, epochdays)

        total_seconds = 7 * 24 * 3600
        n = total_seconds // step_seconds + 1

        t_offsets = np.arange(n, dtype=np.float64) * step_seconds
        ts_jd_array = jd_epoch + t_offsets / 86400.0
        jd_array = np.floor(ts_jd_array)
        fr_array = ts_jd_array - jd_array

        e_array, r_eci, v_eci = sat.sgp4_array(jd_array, fr_array)

        valid_mask = e_array == 0
        r_eci_valid = r_eci[valid_mask]
        v_eci_valid = v_eci[valid_mask]
        ts_valid = ts_jd_array[valid_mask]

        if self.use_cpp_accel:
            r_ecef, v_ecef = self._eci_to_ecef_cpp(r_eci_valid, v_eci_valid, ts_valid)
        else:
            r_ecef, v_ecef = self.eci_to_ecef_batch(r_eci_valid, v_eci_valid, ts_valid)

        return {
            "timestamps_jd": ts_valid,
            "x": r_ecef[:, 0], "y": r_ecef[:, 1], "z": r_ecef[:, 2],
            "vx": v_ecef[:, 0], "vy": v_ecef[:, 1], "vz": v_ecef[:, 2],
        }

    def propagate_7days_eci(self, sat: Satrec, epochyr: int, epochdays: float,
                            step_seconds: int = 1) -> List[dict]:
        if epochyr < 57:
            year = 2000 + epochyr
        else:
            year = 1900 + epochyr

        jd_epoch = self._year_day_to_jd(year, epochdays)

        total_seconds = 7 * 24 * 3600
        results = []

        for t in range(0, total_seconds + 1, step_seconds):
            jd_current = jd_epoch + t / 86400.0
            jd_int = int(jd_current)
            jd_frac = jd_current - jd_int

            e, r, v = sat.sgp4(jd_int, jd_frac)
            if e != 0:
                continue

            results.append({
                "timestamp_jd": jd_current,
                "eci_x": r[0], "eci_y": r[1], "eci_z": r[2],
                "eci_vx": v[0], "eci_vy": v[1], "eci_vz": v[2],
            })

        return results

    @staticmethod
    def _year_day_to_jd(year: int, day_of_year: float) -> float:
        dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1)
        return SGP4Propagator._datetime_to_jd(dt)

    @staticmethod
    def _datetime_to_jd(dt: datetime) -> float:
        a = (14 - dt.month) // 12
        y = dt.year + 4800 - a
        m = dt.month + 12 * a - 3
        jd = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
        frac = (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0
        return jd + frac

    @staticmethod
    def jd_to_datetime(jd: float) -> datetime:
        jd0 = jd - 0.5
        z = int(jd0)
        f = jd0 - z

        if z < 2299161:
            a = z
        else:
            alpha = int((z - 1867216.25) / 36524.25)
            a = z + 1 + alpha - int(alpha / 4)

        b = a + 1524
        c = int((b - 122.1) / 365.25)
        d = int(365.25 * c)
        e = int((b - d) / 30.6001)

        day = b - d - int(30.6001 * e) + f
        month = e - 1 if e < 14 else e - 13
        year = c - 4716 if month > 2 else c - 4715

        day_int = int(day)
        frac_day = day - day_int
        hour = int(frac_day * 24)
        minute = int((frac_day * 24 - hour) * 60)
        second = ((frac_day * 24 - hour) * 60 - minute) * 60

        return datetime(year, month, day_int, hour, int(minute),
                        int(second), tzinfo=timezone.utc)


def _greenwich_apparent_sidereal_time(jd: float) -> float:
    t_ut1 = (jd - 2451545.0) / 36525.0

    gmst_sec = 67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * t_ut1 + \
               0.093104 * t_ut1 * t_ut1 - 6.2e-6 * t_ut1 * t_ut1 * t_ut1
    gmst_sec = gmst_sec % 86400.0
    if gmst_sec < 0:
        gmst_sec += 86400.0

    gmst_rad = gmst_sec * math.pi / 43200.0

    nutation = _nutation_longitude(jd)
    epsilon = _mean_obliquity(jd)

    gast = gmst_rad + nutation * math.cos(epsilon)
    gast = gast % (2.0 * math.pi)
    if gast < 0:
        gast += 2.0 * math.pi

    return gast


def _nutation_longitude(jd: float) -> float:
    t = (jd - 2451545.0) / 36525.0
    omega = math.radians(125.04452 - 1934.136261 * t + 0.0020708 * t * t + t * t * t / 450000.0)
    dpsi = -17.2 * math.sin(omega) / 3600.0
    return math.radians(dpsi)


def _mean_obliquity(jd: float) -> float:
    t = (jd - 2451545.0) / 36525.0
    eps0 = 84381.448 - 46.8150 * t - 0.00059 * t * t + 0.001813 * t * t * t
    return math.radians(eps0 / 3600.0)


def _greenwich_apparent_sidereal_time_array(jd_array: np.ndarray) -> np.ndarray:
    t_ut1 = (jd_array - 2451545.0) / 36525.0

    gmst_sec = 67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * t_ut1 + \
               0.093104 * t_ut1 * t_ut1 - 6.2e-6 * t_ut1 * t_ut1 * t_ut1
    gmst_sec = np.mod(gmst_sec, 86400.0)

    gmst_rad = gmst_sec * np.pi / 43200.0

    t = t_ut1
    omega = np.radians(125.04452 - 1934.136261 * t + 0.0020708 * t * t + t * t * t / 450000.0)
    dpsi = np.radians(-17.2 * np.sin(omega) / 3600.0)

    eps0 = 84381.448 - 46.8150 * t - 0.00059 * t * t + 0.001813 * t * t * t
    epsilon = np.radians(eps0 / 3600.0)

    gast = gmst_rad + dpsi * np.cos(epsilon)
    gast = np.mod(gast, 2.0 * np.pi)

    return gast
