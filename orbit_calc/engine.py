import logging
import os
import gc
from typing import Optional, List

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
        chunk_seconds: int = 86400,
        gc_interval_satellites: int = 10,
        use_cpp_accel: bool = True,
        use_buffer: bool = True,
    ):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.kafka_topic = kafka_topic or os.getenv("KAFKA_TLE_TOPIC", "tle-data")
        self.kafka_group_id = kafka_group_id
        self.propagation_step_seconds = propagation_step_seconds
        self.chunk_seconds = chunk_seconds
        self.gc_interval = gc_interval_satellites
        self._propagator = SGP4Propagator(use_cpp_accel=use_cpp_accel, use_buffer=use_buffer)

        self._db_writer = TimescaleDBWriter(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password,
        )
        self._kafka_consumer: Optional[TLEKafkaConsumer] = None
        self._sat_counter = 0

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

            satellite_id = entry.satellite_name or str(entry.norad_id)
            total_written = 0

            for chunk in self._propagator.propagate_chunked(
                sat,
                epochyr=entry.epoch_year,
                epochdays=entry.epoch_day,
                duration_days=7,
                step_seconds=self.propagation_step_seconds,
                chunk_seconds=self.chunk_seconds,
            ):
                epoch_jd = float(chunk["timestamps_jd"][0]) if len(chunk["timestamps_jd"]) > 0 else 0.0
                chunk_written = self._db_writer.write_numpy_batch(
                    satellite_id=satellite_id,
                    epoch_jd=epoch_jd,
                    numpy_dict=chunk,
                )
                total_written += chunk_written

                del chunk

            self._sat_counter += 1
            if self._sat_counter % self.gc_interval == 0:
                gc.collect()
                logger.debug(f"Garbage collection after {self._sat_counter} satellites")

            logger.info(
                f"Propagated and stored {total_written} positions "
                f"for satellite {satellite_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Error processing TLE for {entry.satellite_name}: {e}")
            return False

    def process_constellation(self, entries: List[TLEEntry],
                              duration_days: int = 30) -> dict:
        sats = []
        for entry in entries:
            try:
                sat = Satrec.twoline2rv(entry.line1, entry.line2, WGS72)
                sats.append((entry.satellite_name or str(entry.norad_id), sat))
            except Exception as e:
                logger.error(f"Failed to parse TLE for {entry.satellite_name}: {e}")

        total_sats = 0
        total_points = 0

        for sat_name, chunk in self._propagator.propagate_constellation(
            sats,
            epochyr=entries[0].epoch_year if entries else 0,
            epochdays=entries[0].epoch_day if entries else 0.0,
            duration_days=duration_days,
            step_seconds=self.propagation_step_seconds,
            chunk_seconds=self.chunk_seconds,
            gc_interval=self.gc_interval,
        ):
            epoch_jd = float(chunk["timestamps_jd"][0]) if len(chunk["timestamps_jd"]) > 0 else 0.0
            written = self._db_writer.write_numpy_batch(
                satellite_id=sat_name,
                epoch_jd=epoch_jd,
                numpy_dict=chunk,
            )
            total_points += written
            total_sats += 1

            del chunk

        return {
            "satellites_processed": len(sats),
            "total_points_written": total_points,
        }

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
        gc.collect()
        logger.info("Orbit calculation engine stopped")
