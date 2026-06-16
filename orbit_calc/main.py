import argparse
import logging
import os
import sys

from orbit_calc.engine import OrbitCalcEngine


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="LEO Satellite Orbit Propagation & Collision Avoidance Engine"
    )
    parser.add_argument(
        "--kafka-servers",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        help="Kafka bootstrap servers",
    )
    parser.add_argument(
        "--kafka-topic",
        default=os.getenv("KAFKA_TLE_TOPIC", "tle-data"),
        help="Kafka TLE topic",
    )
    parser.add_argument(
        "--kafka-group",
        default=os.getenv("KAFKA_GROUP_ID", "orbit-calc-group"),
        help="Kafka consumer group ID",
    )
    parser.add_argument(
        "--db-host",
        default=os.getenv("DB_HOST", "localhost"),
        help="TimescaleDB host",
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=int(os.getenv("DB_PORT", "5432")),
        help="TimescaleDB port",
    )
    parser.add_argument(
        "--db-name",
        default=os.getenv("DB_NAME", "orbit_db"),
        help="TimescaleDB database name",
    )
    parser.add_argument(
        "--db-user",
        default=os.getenv("DB_USER", "orbit_user"),
        help="TimescaleDB user",
    )
    parser.add_argument(
        "--db-password",
        default=os.getenv("DB_PASSWORD", "orbit_pass"),
        help="TimescaleDB password",
    )
    parser.add_argument(
        "--step-seconds",
        type=int,
        default=int(os.getenv("PROPAGATION_STEP_SECONDS", "1")),
        help="Propagation step in seconds",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting LEO Orbit Calculation Engine")

    engine = OrbitCalcEngine(
        kafka_bootstrap_servers=args.kafka_servers,
        kafka_topic=args.kafka_topic,
        kafka_group_id=args.kafka_group,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        propagation_step_seconds=args.step_seconds,
    )

    try:
        engine.start()
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
