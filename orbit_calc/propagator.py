from sgp4.api import Satrec, WGS72, jday
import math
import numpy as np
from typing import List, Tuple, Optional, Iterator, Generator
from datetime import datetime, timezone, timedelta
import gc
import logging

from orbit_calc.space_environment import (
    SpaceEnvironmentProvider, SpaceEnvironmentData
)
from orbit_calc.perturbations import (
    CombinedPerturbation, AtmosphericDensityModel,
    AtmosphericDragPerturbation, SolarRadiationPressurePerturbation
)

logger = logging.getLogger(__name__)

OMEGA_EARTH = 7.2921151467e-5

_CPP_ACCEL_AVAILABLE = False
_CPP_BUFFER_AVAILABLE = False
_CPP_PERTURBATION_AVAILABLE = False
try:
    from orbit_calc._sgp4_binding import eci_to_ecef_batch_cpp
    _CPP_ACCEL_AVAILABLE = True
except ImportError:
    pass

try:
    from orbit_calc._sgp4_binding import BatchECEFBuffer, eci_to_ecef_buffer
    _CPP_BUFFER_AVAILABLE = True
except ImportError:
    pass

try:
    from orbit_calc._sgp4_binding import (
        LockFreePerturbationEngine, BatchPerturbationEngine
    )
    _CPP_PERTURBATION_AVAILABLE = True
except ImportError:
    pass


class SGP4Propagator:
    def __init__(self, use_cpp_accel: bool = True, use_buffer: bool = True,
                 enable_perturbation: bool = True,
                 use_cpp_perturbation: bool = True,
                 auto_env_subscribe: bool = True,
                 area_to_mass: float = 0.01,
                 Cd: float = 2.2, Cr: float = 1.0):
        self.gravity_model = WGS72
        self.use_cpp_accel = use_cpp_accel and _CPP_ACCEL_AVAILABLE
        self.use_buffer = use_buffer and _CPP_BUFFER_AVAILABLE
        self.enable_perturbation = enable_perturbation
        self.use_cpp_perturbation = use_cpp_perturbation and _CPP_PERTURBATION_AVAILABLE
        self.area_to_mass = area_to_mass
        self.Cd = Cd
        self.Cr = Cr

        self._cpp_pert_engine: Optional[LockFreePerturbationEngine] = None
        self._py_pert_engine: Optional[CombinedPerturbation] = None
        self._last_env_seq = 0

        if enable_perturbation:
            if self.use_cpp_perturbation:
                self._cpp_pert_engine = LockFreePerturbationEngine()
            else:
                self._py_pert_engine = CombinedPerturbation()

            if auto_env_subscribe:
                try:
                    env_provider = SpaceEnvironmentProvider()
                    env_provider.add_listener(self._on_env_update)
                    self._env_provider = env_provider
                    self._on_env_update(env_provider.get_current())
                except Exception as e:
                    logger.warning(f"Space env subscription failed: {e}")
                    self._env_provider = None

    def _on_env_update(self, env: SpaceEnvironmentData):
        params = np.array([
            env.f10_7,
            env.f10_7_avg,
            env.kp,
            env.kp_3h,
            5.0e-12,
            self.Cd,
            self.Cr,
            self.area_to_mass,
            float(env.storm_level),
            0.0,
        ], dtype=np.float64)

        if self._cpp_pert_engine is not None:
            self._cpp_pert_engine.update_all(params)
        elif self._py_pert_engine is not None:
            from orbit_calc.space_environment import PerturbationParams
            p = PerturbationParams(
                timestamp=env.timestamp,
                atmospheric_density=5.0e-12,
                drag_coefficient=self.Cd,
                srp_coefficient=self.Cr,
                area_to_mass=self.area_to_mass,
                scale_factor=1.0 + env.storm_level * 0.1,
            )
            self._py_pert_engine.update_params(p)

        self._last_env_seq += 1

    def parse_tle(self, line1: str, line2: str) -> Satrec:
        return Satrec.twoline2rv(line1, line2, self.gravity_model)

    def parse_tle_with_name(self, name: str, line1: str, line2: str) -> Tuple[str, Satrec]:
        sat = Satrec.twoline2rv(line1, line2, self.gravity_model)
        return (name, sat)

    def apply_perturbation_correction(self, r_eci: np.ndarray, v_eci: np.ndarray,
                                       jd_array: np.ndarray,
                                       step_seconds: float = 1.0) -> None:
        if not self.enable_perturbation:
            return

        n = len(r_eci)
        if n == 0:
            return

        if self.use_cpp_perturbation and self._cpp_pert_engine is not None:
            n = len(r_eci)
            r_accum = np.zeros_like(r_eci)
            v_accum = np.zeros_like(v_eci)

            r_curr = r_eci[0].copy()
            v_curr = v_eci[0].copy()

            r_accum[0] = r_curr.copy()
            v_accum[0] = v_curr.copy()

            for i in range(1, n):
                jd_curr = jd_array[i - 1]
                x_s, y_s, z_s, vx_s, vy_s, vz_s = self._cpp_pert_engine.apply_perturbation_single(
                    r_curr[0], r_curr[1], r_curr[2],
                    v_curr[0], v_curr[1], v_curr[2],
                    jd_curr, step_seconds
                )
                r_curr[0] = x_s + (r_eci[i, 0] - r_eci[i - 1, 0])
                r_curr[1] = y_s + (r_eci[i, 1] - r_eci[i - 1, 1])
                r_curr[2] = z_s + (r_eci[i, 2] - r_eci[i - 1, 2])
                v_curr[0] = vx_s + (v_eci[i, 0] - v_eci[i - 1, 0])
                v_curr[1] = vy_s + (v_eci[i, 1] - v_eci[i - 1, 1])
                v_curr[2] = vz_s + (v_eci[i, 2] - v_eci[i - 1, 2])
                r_accum[i] = r_curr.copy()
                v_accum[i] = v_curr.copy()

            np.copyto(r_eci, r_accum)
            np.copyto(v_eci, v_accum)
        elif self._py_pert_engine is not None:
            env = self._env_provider.get_current() if self._env_provider else SpaceEnvironmentData(
                timestamp=datetime.now(timezone.utc)
            )
            for i in range(n):
                r = r_eci[i]
                v = v_eci[i]
                jd = jd_array[i]
                dv = self._py_pert_engine.compute_velocity_correction(
                    r, v, jd, env, step_seconds,
                    self.area_to_mass, self.Cd, self.Cr
                )
                r_eci[i] += v * step_seconds + 0.5 * dv * step_seconds
                v_eci[i] += dv
        elif self._cpp_pert_engine is not None:
            for i in range(n):
                r = r_eci[i]
                v = v_eci[i]
                jd = jd_array[i]
                new_vals = self._cpp_pert_engine.apply_perturbation_single(
                    r[0], r[1], r[2], v[0], v[1], v[2], jd, step_seconds
                )
                r_eci[i, 0] = new_vals[0]
                r_eci[i, 1] = new_vals[1]
                r_eci[i, 2] = new_vals[2]
                v_eci[i, 0] = new_vals[3]
                v_eci[i, 1] = new_vals[4]
                v_eci[i, 2] = new_vals[5]

    def set_perturbation_params(self, f10_7: float = 150.0, f10_7_avg: float = 150.0,
                                kp: float = 2.0, storm_level: float = 0.0,
                                area_to_mass: Optional[float] = None,
                                Cd: Optional[float] = None, Cr: Optional[float] = None):
        if area_to_mass is not None:
            self.area_to_mass = area_to_mass
        if Cd is not None:
            self.Cd = Cd
        if Cr is not None:
            self.Cr = Cr

        if self._cpp_pert_engine is not None:
            params = np.array([
                f10_7, f10_7_avg, kp, kp, 5.0e-12,
                self.Cd, self.Cr, self.area_to_mass,
                storm_level, 0.0,
            ], dtype=np.float64)
            self._cpp_pert_engine.update_all(params)

    def get_perturbation_params(self) -> np.ndarray:
        if self._cpp_pert_engine is not None:
            return self._cpp_pert_engine.get_all()
        elif self._py_pert_engine is not None:
            p = self._py_pert_engine.get_params()
            return p.as_array()
        return np.zeros(10)

    @property
    def perturbation_sequence(self) -> int:
        if self._cpp_pert_engine is not None:
            return int(self._cpp_pert_engine.sequence())
        return self._last_env_seq

    def has_perturbation_update(self, last_seq: int) -> bool:
        if self._cpp_pert_engine is not None:
            return self._cpp_pert_engine.has_update(last_seq)
        return self._last_env_seq > last_seq

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

        if self.enable_perturbation:
            self.apply_perturbation_correction(
                r_eci_valid, v_eci_valid, ts_valid, step_seconds
            )

        if self.use_buffer:
            buf = eci_to_ecef_buffer(
                r_eci_valid[:, 0], r_eci_valid[:, 1], r_eci_valid[:, 2],
                v_eci_valid[:, 0], v_eci_valid[:, 1], v_eci_valid[:, 2],
                ts_valid
            )
            data = buf.to_numpy_dict()
            return {
                "timestamps_jd": ts_valid,
                "x": np.asarray(data['x']),
                "y": np.asarray(data['y']),
                "z": np.asarray(data['z']),
                "vx": np.asarray(data['vx']),
                "vy": np.asarray(data['vy']),
                "vz": np.asarray(data['vz']),
                "_buffer": buf,
            }

        if self.use_cpp_accel:
            r_ecef, v_ecef = self._eci_to_ecef_cpp(r_eci_valid, v_eci_valid, ts_valid)
        else:
            r_ecef, v_ecef = self.eci_to_ecef_batch(r_eci_valid, v_eci_valid, ts_valid)

        return {
            "timestamps_jd": ts_valid,
            "x": r_ecef[:, 0], "y": r_ecef[:, 1], "z": r_ecef[:, 2],
            "vx": v_ecef[:, 0], "vy": v_ecef[:, 1], "vz": v_ecef[:, 2],
        }

    def propagate_chunked(self, sat: Satrec, epochyr: int, epochdays: float,
                          duration_days: int = 7, step_seconds: int = 1,
                          chunk_seconds: int = 86400) -> Iterator[dict]:
        if epochyr < 57:
            year = 2000 + epochyr
        else:
            year = 1900 + epochyr

        jd_epoch = self._year_day_to_jd(year, epochdays)

        total_seconds = duration_days * 24 * 3600

        for chunk_start in range(0, total_seconds + 1, chunk_seconds):
            chunk_end = min(chunk_start + chunk_seconds, total_seconds)
            chunk_n = (chunk_end - chunk_start) // step_seconds + 1

            t_offsets = chunk_start + np.arange(chunk_n, dtype=np.float64) * step_seconds
            ts_jd_chunk = jd_epoch + t_offsets / 86400.0
            jd_chunk = np.floor(ts_jd_chunk)
            fr_chunk = ts_jd_chunk - jd_chunk

            e_array, r_eci, v_eci = sat.sgp4_array(jd_chunk, fr_chunk)

            valid_mask = e_array == 0
            r_eci_valid = r_eci[valid_mask]
            v_eci_valid = v_eci[valid_mask]
            ts_valid = ts_jd_chunk[valid_mask]

            if self.enable_perturbation:
                self.apply_perturbation_correction(
                    r_eci_valid, v_eci_valid, ts_valid, step_seconds
                )

            if self.use_buffer and _CPP_BUFFER_AVAILABLE:
                buf = eci_to_ecef_buffer(
                    r_eci_valid[:, 0], r_eci_valid[:, 1], r_eci_valid[:, 2],
                    v_eci_valid[:, 0], v_eci_valid[:, 1], v_eci_valid[:, 2],
                    ts_valid
                )
                data = buf.to_numpy_dict()
                yield {
                    "timestamps_jd": ts_valid,
                    "x": np.asarray(data['x']),
                    "y": np.asarray(data['y']),
                    "z": np.asarray(data['z']),
                    "vx": np.asarray(data['vx']),
                    "vy": np.asarray(data['vy']),
                    "vz": np.asarray(data['vz']),
                    "_buffer": buf,
                }
            else:
                if self.use_cpp_accel:
                    r_ecef, v_ecef = self._eci_to_ecef_cpp(r_eci_valid, v_eci_valid, ts_valid)
                else:
                    r_ecef, v_ecef = self.eci_to_ecef_batch(r_eci_valid, v_eci_valid, ts_valid)

                yield {
                    "timestamps_jd": ts_valid,
                    "x": r_ecef[:, 0], "y": r_ecef[:, 1], "z": r_ecef[:, 2],
                    "vx": v_ecef[:, 0], "vy": v_ecef[:, 1], "vz": v_ecef[:, 2],
                }

    def propagate_constellation(self, sats: List[Tuple[str, Satrec]],
                                epochyr: int, epochdays: float,
                                duration_days: int = 7,
                                step_seconds: int = 1,
                                chunk_seconds: int = 86400,
                                gc_interval: int = 10) -> Iterator[Tuple[str, dict]]:
        count = 0
        for sat_name, sat in sats:
            for chunk in self.propagate_chunked(
                sat, epochyr, epochdays, duration_days, step_seconds, chunk_seconds
            ):
                yield (sat_name, chunk)

            count += 1
            if count % gc_interval == 0:
                gc.collect()

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
