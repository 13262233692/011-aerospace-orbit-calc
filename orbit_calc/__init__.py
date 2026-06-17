from orbit_calc.propagator import SGP4Propagator
from orbit_calc.tle_parser import TLEEntry, parse_tle_entry, parse_tle_file, parse_tle_kafka_message
from orbit_calc.db_writer import TimescaleDBWriter
from orbit_calc.kafka_consumer import TLEKafkaConsumer
from orbit_calc.engine import OrbitCalcEngine
from orbit_calc.space_environment import (
    SpaceEnvironmentProvider,
    NOAASpaceWeatherClient,
    SpaceEnvironmentData,
    PerturbationParams,
)
from orbit_calc.perturbations import (
    CombinedPerturbation,
    AtmosphericDensityModel,
    AtmosphericDragPerturbation,
    SolarRadiationPressurePerturbation,
)

_CPP_PERTURBATION_AVAILABLE = False
try:
    from orbit_calc._sgp4_binding import (
        LockFreePerturbationEngine,
        BatchPerturbationEngine,
    )
    _CPP_PERTURBATION_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    "SGP4Propagator",
    "TLEEntry",
    "parse_tle_entry",
    "parse_tle_file",
    "parse_tle_kafka_message",
    "TimescaleDBWriter",
    "TLEKafkaConsumer",
    "OrbitCalcEngine",
    "SpaceEnvironmentProvider",
    "NOAASpaceWeatherClient",
    "SpaceEnvironmentData",
    "PerturbationParams",
    "CombinedPerturbation",
    "AtmosphericDensityModel",
    "AtmosphericDragPerturbation",
    "SolarRadiationPressurePerturbation",
]

if _CPP_PERTURBATION_AVAILABLE:
    __all__.extend([
        "LockFreePerturbationEngine",
        "BatchPerturbationEngine",
    ])

