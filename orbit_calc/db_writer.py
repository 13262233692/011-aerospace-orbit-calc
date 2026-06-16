import logging
from datetime import datetime, timezone
from typing import List, Optional

import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)


class TimescaleDBWriter:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "orbit_db",
        user: str = "orbit_user",
        password: str = "orbit_pass",
    ):
        self._conn_params = {
            "host": host,
            "port": port,
            "dbname": database,
            "user": user,
            "password": password,
        }
        self._conn: Optional[psycopg2.extensions.connection] = None

    def connect(self):
        self._conn = psycopg2.connect(**self._conn_params)
        self._conn.autocommit = False
        logger.info("Connected to TimescaleDB")
        self._ensure_schema()

    def _ensure_schema(self):
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS satellite_orbit (
                satellite_id TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                pos_x DOUBLE PRECISION NOT NULL,
                pos_y DOUBLE PRECISION NOT NULL,
                pos_z DOUBLE PRECISION NOT NULL,
                vel_x DOUBLE PRECISION NOT NULL,
                vel_y DOUBLE PRECISION NOT NULL,
                vel_z DOUBLE PRECISION NOT NULL,
                coord_system TEXT NOT NULL DEFAULT 'WGS84_ECEF',
                epoch_jd DOUBLE PRECISION,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM timescaledb_information.hypertables
                WHERE hypertable_name = 'satellite_orbit'
            );
        """)
        is_hypertable = cursor.fetchone()[0]

        if not is_hypertable:
            try:
                cursor.execute("""
                    SELECT create_hypertable('satellite_orbit', 'timestamp',
                                            chunk_time_interval => INTERVAL '1 day',
                                            migrate_data => true);
                """)
            except psycopg2.Error as e:
                logger.warning(f"Hypertable creation note: {e}")

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_satellite_id_timestamp
            ON satellite_orbit (satellite_id, timestamp DESC);
        """)

        self._conn.commit()
        cursor.close()
        logger.info("Schema ensured (hypertable + indexes)")

    def _jd_to_datetime(self, jd: float) -> datetime:
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
        second = int(((frac_day * 24 - hour) * 60 - minute) * 60)

        return datetime(year, month, day_int, hour, minute, second, tzinfo=timezone.utc)

    def write_batch_result(self, satellite_id: str, epoch_jd: float,
                           positions: list, coord_system: str = "WGS84_ECEF",
                           batch_size: int = 10000) -> int:
        if not self._conn:
            raise RuntimeError("Not connected to database")

        total_written = 0
        rows = []

        for pos in positions:
            ts = self._jd_to_datetime(pos["timestamp_jd"])
            rows.append((
                satellite_id,
                ts,
                pos["x"],
                pos["y"],
                pos["z"],
                pos["vx"],
                pos["vy"],
                pos["vz"],
                coord_system,
                epoch_jd,
            ))

            if len(rows) >= batch_size:
                self._insert_rows(rows)
                total_written += len(rows)
                rows = []

        if rows:
            self._insert_rows(rows)
            total_written += len(rows)

        self._conn.commit()
        logger.info(f"Wrote {total_written} positions for satellite {satellite_id}")
        return total_written

    def _insert_rows(self, rows: List[tuple]):
        cursor = self._conn.cursor()
        execute_values(
            cursor,
            """
            INSERT INTO satellite_orbit
                (satellite_id, timestamp, pos_x, pos_y, pos_z,
                 vel_x, vel_y, vel_z, coord_system, epoch_jd)
            VALUES %s
            """,
            rows,
            page_size=1000,
        )
        cursor.close()

    def write_numpy_batch(self, satellite_id: str, epoch_jd: float,
                          numpy_dict: dict, coord_system: str = "WGS84_ECEF",
                          batch_size: int = 10000) -> int:
        timestamps_jd = numpy_dict["timestamps_jd"]
        n = len(timestamps_jd)

        total_written = 0
        rows = []

        for i in range(n):
            ts = self._jd_to_datetime(float(timestamps_jd[i]))
            rows.append((
                satellite_id,
                ts,
                float(numpy_dict["x"][i]),
                float(numpy_dict["y"][i]),
                float(numpy_dict["z"][i]),
                float(numpy_dict["vx"][i]),
                float(numpy_dict["vy"][i]),
                float(numpy_dict["vz"][i]),
                coord_system,
                epoch_jd,
            ))

            if len(rows) >= batch_size:
                self._insert_rows(rows)
                total_written += len(rows)
                rows = []

        if rows:
            self._insert_rows(rows)
            total_written += len(rows)

        self._conn.commit()
        logger.info(f"Wrote {total_written} positions for satellite {satellite_id}")
        return total_written

    def query_positions(
        self,
        satellite_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000,
    ) -> List[dict]:
        if not self._conn:
            raise RuntimeError("Not connected to database")

        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT satellite_id, timestamp, pos_x, pos_y, pos_z,
                   vel_x, vel_y, vel_z, epoch_jd
            FROM satellite_orbit
            WHERE satellite_id = %s
              AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp ASC
            LIMIT %s
            """,
            (satellite_id, start_time, end_time, limit),
        )

        results = []
        for row in cursor.fetchall():
            results.append({
                "satellite_id": row[0],
                "timestamp": row[1],
                "pos_x": row[2],
                "pos_y": row[3],
                "pos_z": row[4],
                "vel_x": row[5],
                "vel_y": row[6],
                "vel_z": row[7],
                "epoch_jd": row[8],
            })

        cursor.close()
        return results

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("TimescaleDB connection closed")
