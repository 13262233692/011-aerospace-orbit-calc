import math
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import threading

from orbit_calc.space_environment import SpaceEnvironmentData, PerturbationParams


R_EARTH = 6378.137
GM = 398600.4418
OMEGA_EARTH = 7.2921151467e-5
SOLAR_PRESSURE = 4.56e-6


class AtmosphericDensityModel:
    def __init__(self):
        self._thermosphere_coeffs = self._init_thermosphere_coeffs()

    @staticmethod
    def _init_thermosphere_coeffs() -> dict:
        return {
            'f10_scale': 1.0e-4,
            'kp_scale': 0.15,
            'exobase_altitude': 500.0,
            'exobase_temp_min': 600.0,
            'exobase_temp_max': 2000.0,
            'density_scale_h': 50.0,
            'geomagnetic_factor': 0.5,
            'semiannual_factor': 1.0,
        }

    def compute(self, altitude_km: float, latitude: float, longitude: float,
                local_solar_time: float, day_of_year: int,
                env: SpaceEnvironmentData) -> float:
        if altitude_km < 90:
            return self._lower_atmosphere(altitude_km)
        elif altitude_km < 500:
            return self._thermosphere(altitude_km, latitude, local_solar_time,
                                      day_of_year, env)
        else:
            return self._exosphere(altitude_km, env)

    @staticmethod
    def _lower_atmosphere(altitude_km: float) -> float:
        if altitude_km < 20:
            H = 8.5
            rho0 = 1.225e-3
        elif altitude_km < 50:
            H = 6.0
            rho0 = 3.89e-2
        elif altitude_km < 90:
            H = 15.0
            rho0 = 1.05e-3
        else:
            H = 20.0
            rho0 = 3.4e-6
        return rho0 * math.exp(-(altitude_km - 0) / H)

    def _thermosphere(self, altitude_km: float, latitude: float,
                      local_solar_time: float, day_of_year: int,
                      env: SpaceEnvironmentData) -> float:
        coeffs = self._thermosphere_coeffs

        f10 = env.f10_7
        f10_bar = env.f10_7_avg

        T_inf = coeffs['exobase_temp_min'] + \
                0.4 * (f10 - 70) + \
                0.025 * (f10_bar - 70) + \
                1.8 * env.kp + \
                30.0 * math.sin(local_solar_time * math.pi / 12 - 2.0)

        T_inf = min(coeffs['exobase_temp_max'], T_inf)
        T_inf = max(coeffs['exobase_temp_min'], T_inf)

        geomagnetic_factor = 1.0 + coeffs['geomagnetic_factor'] * env.kp / 9.0

        semiannual = 1.0 + 0.05 * math.sin(2 * math.pi * (day_of_year - 172) / 365)

        H = 1.7 + 0.02 * T_inf

        rho = 6.0e-10 * geomagnetic_factor * semiannual * \
              math.exp(-(altitude_km - 120) / H)

        if env.storm_level >= 2:
            storm_factor = 1.0 + 0.5 * env.storm_level * \
                           math.exp(-(altitude_km - 300) / 100)
            rho *= storm_factor

        return rho

    def _exosphere(self, altitude_km: float, env: SpaceEnvironmentData) -> float:
        f10 = env.f10_7
        f10_bar = env.f10_7_avg

        T_inf = 600 + 0.4 * (f10 - 70) + 0.025 * (f10_bar - 70) + 1.8 * env.kp
        H = 1.7 + 0.02 * T_inf

        base_density = 1.0e-15 * math.exp(-(altitude_km - 500) / H)

        storm_factor = 1.0 + 0.3 * env.kp * math.exp(-(altitude_km - 500) / 200)

        return base_density * storm_factor


class AtmosphericDragPerturbation:
    def __init__(self):
        self.density_model = AtmosphericDensityModel()

    def compute_acceleration(self, r_eci: np.ndarray, v_eci: np.ndarray,
                             altitude_km: float, latitude: float,
                             longitude: float, local_solar_time: float,
                             day_of_year: int,
                             env: SpaceEnvironmentData,
                             area_to_mass: float = 0.01,
                             Cd: float = 2.2) -> np.ndarray:
        rho = self.density_model.compute(
            altitude_km, latitude, longitude, local_solar_time, day_of_year, env
        )

        omega_vec = np.array([0, 0, OMEGA_EARTH])
        v_rel = v_eci - np.cross(omega_vec, r_eci)

        v_rel_mag = np.linalg.norm(v_rel)
        if v_rel_mag < 1e-10:
            return np.zeros(3)

        v_rel_unit = v_rel / v_rel_mag

        accel = -0.5 * Cd * area_to_mass * rho * v_rel_mag * v_rel

        return accel

    def compute_velocity_delta(self, r_eci: np.ndarray, v_eci: np.ndarray,
                               altitude_km: float, latitude: float,
                               longitude: float, local_solar_time: float,
                               day_of_year: int,
                               env: SpaceEnvironmentData,
                               dt: float,
                               area_to_mass: float = 0.01,
                               Cd: float = 2.2) -> np.ndarray:
        accel = self.compute_acceleration(
            r_eci, v_eci, altitude_km, latitude, longitude,
            local_solar_time, day_of_year, env, area_to_mass, Cd
        )
        return accel * dt


class SolarRadiationPressurePerturbation:
    def __init__(self):
        self._sun_cache = None
        self._sun_cache_time = None

    def _sun_position(self, jd: float) -> np.ndarray:
        if self._sun_cache is not None and self._sun_cache_time == jd:
            return self._sun_cache

        T = (jd - 2451545.0) / 36525.0

        L = 280.46645 + 36000.76983 * T + 0.0003032 * T * T
        g = 357.52910 + 35999.05030 * T - 0.0001559 * T * T - 0.00000048 * T * T * T
        L_rad = math.radians(L)
        g_rad = math.radians(g)

        sun_lon_rad = L_rad + math.radians(
            1.914600 * math.sin(g_rad) +
            0.019993 * math.sin(2 * g_rad) +
            0.000290 * math.sin(3 * g_rad)
        )

        R_AU = 1.000001018 * (1 - 0.016708617 * math.cos(g_rad) -
                               0.000139589 * math.cos(2 * g_rad))

        AU_km = 149597870.7
        R_km = R_AU * AU_km

        sun_pos_ecl = np.array([
            R_km * math.cos(sun_lon_rad),
            R_km * math.sin(sun_lon_rad),
            0.0
        ])

        eps = math.radians(23.0 + 26.0 / 60 + 21.448 / 3600 -
                            46.8150 / 3600 * T -
                            0.00059 / 3600 * T * T +
                            0.001813 / 3600 * T * T * T)

        sun_pos_eci = np.array([
            sun_pos_ecl[0],
            sun_pos_ecl[1] * math.cos(eps),
            sun_pos_ecl[1] * math.sin(eps),
        ])

        self._sun_cache = sun_pos_eci
        self._sun_cache_time = jd
        return sun_pos_eci

    def compute_acceleration(self, r_eci: np.ndarray, jd: float,
                             env: SpaceEnvironmentData,
                             area_to_mass: float = 0.01,
                             Cr: float = 1.0,
                             eclipsed: bool = False) -> np.ndarray:
        if eclipsed:
            return np.zeros(3)

        sun_pos = self._sun_position(jd)

        r_sun = sun_pos - r_eci
        r_sun_mag = np.linalg.norm(r_sun)
        if r_sun_mag < 1e-10:
            return np.zeros(3)

        r_sun_unit = r_sun / r_sun_mag

        AU_km = 149597870.7
        r_au = r_sun_mag / AU_km

        P = SOLAR_PRESSURE / (r_au * r_au)

        if env.storm_level >= 1:
            P *= 1.0 + 0.3 * env.storm_level

        accel = Cr * area_to_mass * P * r_sun_unit

        return accel

    def check_eclipse(self, r_eci: np.ndarray, jd: float) -> bool:
        sun_pos = self._sun_position(jd)

        r_sun = sun_pos - r_eci
        r_sun_mag = np.linalg.norm(r_sun)
        r_mag = np.linalg.norm(r_eci)

        cos_beta = -np.dot(r_eci, r_sun) / (r_mag * r_sun_mag)
        beta = math.acos(max(-1.0, min(1.0, cos_beta)))

        sun_angular = math.atan(696000.0 / r_sun_mag)
        earth_angular = math.atan(R_EARTH / r_mag)

        return beta < (earth_angular - sun_angular)


class CombinedPerturbation:
    def __init__(self):
        self.drag = AtmosphericDragPerturbation()
        self.srp = SolarRadiationPressurePerturbation()
        self._params = PerturbationParams(
            timestamp=__import__('datetime').datetime.now(__import__('datetime').timezone.utc)
        )
        self._params_lock = threading.Lock()

    def update_params(self, params: PerturbationParams):
        with self._params_lock:
            self._params = params

    def get_params(self) -> PerturbationParams:
        with self._params_lock:
            return self._params

    def compute_total_acceleration(self, r_eci: np.ndarray, v_eci: np.ndarray,
                                    jd: float, env: SpaceEnvironmentData,
                                    area_to_mass: float = 0.01,
                                    Cd: float = 2.2, Cr: float = 1.0) -> np.ndarray:
        r_mag = np.linalg.norm(r_eci)
        altitude_km = r_mag - R_EARTH

        latitude = math.degrees(math.asin(max(-1.0, min(1.0, r_eci[2] / r_mag))))
        longitude = math.degrees(math.atan2(r_eci[1], r_eci[0]))

        local_solar_time_hours = self._compute_local_solar_time(jd, longitude)

        jd0 = int(jd) - 1721425.5
        d = int(jd0)
        year = int((d - 15) / 365.25) + 1
        month = int((jd0 - (365 * year + int(year / 4) - int(year / 100) + int(year / 400))) / 30) + 1
        day = int(jd0 - (365 * year + int(year / 4) - int(year / 100) + int(year / 400)))
        day_of_year = int(jd0 - (365 * (year - 1) + int((year - 1) / 4) - int((year - 1) / 100) + int((year - 1) / 400)))

        eclipsed = self.srp.check_eclipse(r_eci, jd)

        drag_accel = self.drag.compute_acceleration(
            r_eci, v_eci, altitude_km, latitude, longitude,
            local_solar_time_hours, day_of_year, env, area_to_mass, Cd
        )

        srp_accel = self.srp.compute_acceleration(
            r_eci, jd, env, area_to_mass, Cr, eclipsed
        )

        return drag_accel + srp_accel

    def compute_velocity_correction(self, r_eci: np.ndarray, v_eci: np.ndarray,
                                    jd: float, env: SpaceEnvironmentData,
                                    dt: float,
                                    area_to_mass: float = 0.01,
                                    Cd: float = 2.2, Cr: float = 1.0) -> np.ndarray:
        accel = self.compute_total_acceleration(
            r_eci, v_eci, jd, env, area_to_mass, Cd, Cr
        )
        return accel * dt

    @staticmethod
    def _compute_local_solar_time(jd: float, longitude_deg: float) -> float:
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

        day = b - d - int(30.6001 * e)
        month = e - 1 if e < 14 else e - 13
        year = c - 4716 if month > 2 else c - 4715

        hour = int(f * 24)
        minute = int((f * 24 - hour) * 60)

        gst = 18.697374558 + 24.06570982441908 * (jd - 2451545.0)
        gst = gst % 24
        if gst < 0:
            gst += 24

        lst = (gst + longitude_deg / 15.0) % 24
        if lst < 0:
            lst += 24

        return lst


def compute_orbital_elements(r: np.ndarray, v: np.ndarray) -> dict:
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    h_vec = np.cross(r, v)
    h_mag = np.linalg.norm(h_vec)

    e_vec = (np.cross(v, h_vec) / GM) - (r / r_mag)
    e_mag = np.linalg.norm(e_vec)

    a = 1.0 / (2.0 / r_mag - v_mag * v_mag / GM)

    i = math.degrees(math.acos(h_vec[2] / h_mag))

    return {
        'a_km': a,
        'e': e_mag,
        'i_deg': i,
        'h_km2_s': h_mag,
        'r_km': r_mag,
        'v_km_s': v_mag,
        'altitude_km': r_mag - R_EARTH,
    }
