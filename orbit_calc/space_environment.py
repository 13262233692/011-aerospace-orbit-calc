import logging
import threading
import time
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from collections import deque

import requests
import numpy as np

logger = logging.getLogger(__name__)

NOAA_SWPC_BASE = "https://services.swpc.noaa.gov"
F107_URL = NOAA_SWPC_BASE + "/json/51007.json"
KP_URL = NOAA_SWPC_BASE + "/json/3100.json"
SOLAR_WIND_URL = NOAA_SWPC_BASE + "/products/summary/solar-wind-speed.json"

DEFAULT_POLL_INTERVAL = 300


@dataclass
class SpaceEnvironmentData:
    timestamp: datetime
    f10_7: float = 150.0
    f10_7_obs: float = 150.0
    f10_7_avg: float = 150.0
    kp: float = 2.0
    kp_3h: float = 2.0
    ap: float = 5.0
    solar_wind_speed: float = 400.0
    solar_wind_density: float = 5.0
    storm_level: int = 0

    def is_storm(self) -> bool:
        return self.kp >= 5.0 or self.storm_level >= 1


@dataclass
class PerturbationParams:
    timestamp: datetime
    atmospheric_density: float = 5.0e-12
    drag_coefficient: float = 2.2
    srp_coefficient: float = 1.0
    area_to_mass: float = 0.01
    scale_factor: float = 1.0

    def as_array(self) -> np.ndarray:
        return np.array([
            self.atmospheric_density,
            self.drag_coefficient,
            self.srp_coefficient,
            self.area_to_mass,
            self.scale_factor,
        ], dtype=np.float64)


class NOAASpaceWeatherClient:
    def __init__(self, poll_interval: int = DEFAULT_POLL_INTERVAL,
                 use_mock_data: bool = False,
                 on_update: Optional[Callable[[SpaceEnvironmentData], None]] = None):
        self.poll_interval = poll_interval
        self.use_mock_data = use_mock_data
        self.on_update = on_update
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._current_data = SpaceEnvironmentData(
            timestamp=datetime.now(timezone.utc)
        )
        self._history: deque = deque(maxlen=1000)
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'OrbitCalc/1.0 (LEO Satellite Trajectory System)'
        })
        self._retry_count = 0
        self._max_retries = 5

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("NOAA Space Weather client started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self._session.close()
        logger.info("NOAA Space Weather client stopped")

    def _run(self):
        while self._running:
            try:
                if self.use_mock_data:
                    self._update_mock()
                else:
                    self._update_real()
                self._retry_count = 0
            except Exception as e:
                self._retry_count += 1
                wait_time = min(60 * (2 ** min(self._retry_count, 4)), 600)
                logger.warning(f"NOAA fetch failed ({self._retry_count}x): {e}. Retry in {wait_time}s")
                time.sleep(wait_time)
                continue

            if self.on_update:
                try:
                    self.on_update(self._current_data)
                except Exception as e:
                    logger.error(f"Update callback error: {e}")

            time.sleep(self.poll_interval)

    def _update_real(self):
        data = SpaceEnvironmentData(timestamp=datetime.now(timezone.utc))

        try:
            resp = self._session.get(F107_URL, timeout=10)
            if resp.status_code == 200:
                f107_data = resp.json()
                if len(f107_data) > 0:
                    latest = f107_data[0]
                    data.f10_7_obs = float(latest.get('f10_7', 150))
                    data.f10_7 = float(latest.get('f10_7_adj', data.f10_7_obs))
                    if len(f107_data) >= 81:
                        avg_data = f107_data[:81]
                        data.f10_7_avg = np.mean([float(d.get('f10_7_adj', 150)) for d in avg_data])
        except Exception as e:
            logger.warning(f"F10.7 fetch failed: {e}")

        try:
            resp = self._session.get(KP_URL, timeout=10)
            if resp.status_code == 200:
                kp_data = resp.json()
                if len(kp_data) > 0:
                    latest = kp_data[0]
                    data.kp = float(latest.get('kp_index', 2.0))
                    data.ap = float(latest.get('a_value', 5.0))
                    if data.kp >= 8.0:
                        data.storm_level = 5
                    elif data.kp >= 7.0:
                        data.storm_level = 4
                    elif data.kp >= 6.0:
                        data.storm_level = 3
                    elif data.kp >= 5.0:
                        data.storm_level = 2
                    elif data.kp >= 4.0:
                        data.storm_level = 1
        except Exception as e:
            logger.warning(f"Kp fetch failed: {e}")

        try:
            resp = self._session.get(SOLAR_WIND_URL, timeout=10)
            if resp.status_code == 200:
                sw_data = resp.json()
                if 'data' in sw_data and len(sw_data['data']) > 0:
                    latest = sw_data['data'][-1]
                    data.solar_wind_speed = float(latest.get('wind_speed', 400))
                    data.solar_wind_density = float(latest.get('density', 5.0))
        except Exception as e:
            logger.warning(f"Solar wind fetch failed: {e}")

        with self._lock:
            self._current_data = data
            self._history.append(data)
        logger.debug(f"Space weather: F10.7={data.f10_7:.1f}, Kp={data.kp:.1f}, Storm=G{data.storm_level}")

    def _update_mock(self):
        t = time.time()
        data = SpaceEnvironmentData(timestamp=datetime.now(timezone.utc))

        storm_period = 10.0
        storm_phase = (t % storm_period) / storm_period
        if storm_phase < 0.3:
            storm_intensity = 1.0
        elif storm_phase < 0.6:
            storm_intensity = 1.0 - (storm_phase - 0.3) / 0.3
        else:
            storm_intensity = 0.0

        base_period = 15.0
        base_f107 = 80.0 + 100.0 * (0.5 + 0.5 * np.sin(t / base_period * 2 * np.pi))
        data.f10_7 = base_f107 + storm_intensity * 80
        data.f10_7_obs = data.f10_7
        data.f10_7_avg = base_f107 * 0.8 + 40

        kp_period = 8.0
        base_kp = 1.5 + 1.0 * (0.5 + 0.5 * np.sin(t / kp_period * 2 * np.pi))
        data.kp = min(9.0, base_kp + storm_intensity * 5.0)
        data.ap = 5.0 + data.kp * 5.0

        if data.kp >= 8.0:
            data.storm_level = 5
        elif data.kp >= 7.0:
            data.storm_level = 4
        elif data.kp >= 6.0:
            data.storm_level = 3
        elif data.kp >= 5.0:
            data.storm_level = 2
        elif data.kp >= 4.0:
            data.storm_level = 1

        data.solar_wind_speed = 350.0 + storm_intensity * 600
        data.solar_wind_density = 3.0 + storm_intensity * 15.0

        with self._lock:
            self._current_data = data
            self._history.append(data)

    def get_current(self) -> SpaceEnvironmentData:
        with self._lock:
            return self._current_data

    def get_history(self, max_points: int = 100) -> list:
        with self._lock:
            return list(self._history)[-max_points:]

    def set_data(self, data: SpaceEnvironmentData):
        with self._lock:
            self._current_data = data
            self._history.append(data)


class SpaceEnvironmentProvider:
    _instance: Optional['SpaceEnvironmentProvider'] = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._client: Optional[NOAASpaceWeatherClient] = None
        self._listeners: list = []

    def initialize(self, poll_interval: int = DEFAULT_POLL_INTERVAL,
                   use_mock_data: bool = False):
        if self._client and self._client._running:
            self._client.stop()

        self._client = NOAASpaceWeatherClient(
            poll_interval=poll_interval,
            use_mock_data=use_mock_data,
            on_update=self._on_data_update
        )
        self._client.start()
        logger.info("SpaceEnvironmentProvider initialized")

    def _on_data_update(self, data: SpaceEnvironmentData):
        for listener in self._listeners:
            try:
                listener(data)
            except Exception as e:
                logger.error(f"Listener error: {e}")

    def add_listener(self, listener: Callable[[SpaceEnvironmentData], None]):
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[SpaceEnvironmentData], None]):
        if listener in self._listeners:
            self._listeners.remove(listener)

    def get_current(self) -> SpaceEnvironmentData:
        if self._client:
            return self._client.get_current()
        return SpaceEnvironmentData(timestamp=datetime.now(timezone.utc))

    def shutdown(self):
        if self._client:
            self._client.stop()
        self._listeners.clear()
        logger.info("SpaceEnvironmentProvider shut down")
