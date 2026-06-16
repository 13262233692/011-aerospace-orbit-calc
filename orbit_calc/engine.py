import logging
import os
from typing import Optional

from sgp4.api import Satrec, WGS72

from orbit_calc.propagator import SGP4Propagator
from orbit_calc.tle_parser import TLEEntry, parse_tle_kafka_message
from orbit_calc.kafka_consumer import TLEKafkaConsumer
from orbit_calc.db_writer import TimescaleDBWriter

logger = logging.getLogger(__name__)


class OrbitCalcEngine:
    def __init__(
        self,
        kafka_bootstrap_servers: Optional[str] = None,
        kafka_topic: Optional[str] = None,
        kafka_group_id: str = "orbit-calc-group",
        db_host: str = "localhost",
        db_port: int = 5432,
        db_name: str = "orbit_db",
        db_user: str = "orbit_user",
        db_password: str = "orbit_pass",
        propagation_step_seconds: int = 1,
    ):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.kafka_topic = kafka_topic or os.getenv("KAFKA_TLE_TOPIC", "tle-data")
        self.kafka_group_id = kafka_group_id
        self.propagation_step_seconds = propagation_step_seconds
        self._propagator = SGP4Propagator()

        self._db_writer = TimescaleDBWriter(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password,
        )
        self._kafka_consumer: Optional[TLEKafkaConsumer] = None

    def start(self):
        self._db_writer.connect()
        logger.info("Database connection established")

        self._kafka_consumer = TLEKafkaConsumer(
            bootstrap_servers=self.kafka_bootstrap_servers,
            group_id=self.kafka_group_id,
            topic=self.kafka_topic,
        )
        self._kafka_consumer.set_message_handler(self._handle_tle_message)

        logger.info(
            f"Starting orbit calculation engine, consuming from {self.kafka_topic}"
        )
        self._kafka_consumer.start()

    def process_tle_entry(self, entry: TLEEntry) -> bool:
        try:
            sat = Satrec.twoline2rv(entry.line1, entry.line2, WGS72)

            result = self._propagator.propagate_7days_fast(
                sat,
                epochyr=entry.epoch_year,
                epochdays=entry.epoch_day,
                step_seconds=self.propagation_step_seconds,
            )

            satellite_id = entry.satellite_name or str(entry.norad_id)
            epoch_jd = float(result["timestamps_jd"][0]) if len(result["timestamps_jd"]) > 0 else 0.0

            self._db_writer.write_numpy_batch(
                satellite_id=satellite_id,
                epoch_jd=epoch_jd,
                numpy_dict=result,
            )

            logger.info(
                f"Propagated and stored {len(result['timestamps_jd'])} positions "
                f"for satellite {satellite_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Error processing TLE for {entry.satellite_name}: {e}")
            return False

    def _handle_tle_message(self, message_value: bytes):
        entry = parse_tle_kafka_message(message_value)
        if entry is None:
            logger.warning("Failed to parse TLE from Kafka message")
            return

        logger.info(f"Received TLE for satellite: {entry.satellite_name} (NORAD: {entry.norad_id})")
        self.process_tle_entry(entry)

    def stop(self):
        if self._kafka_consumer:
            self._kafka_consumer.stop()
        self._db_writer.close()
        logger.info("Orbit calculation engine stopped")
