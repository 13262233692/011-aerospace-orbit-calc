import logging
import signal
import sys
from typing import Callable, Optional

from confluent_kafka import Consumer, KafkaError, KafkaException

logger = logging.getLogger(__name__)


class TLEKafkaConsumer:
    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        topic: str,
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = True,
        poll_timeout: float = 1.0,
    ):
        self.topic = topic
        self.poll_timeout = poll_timeout
        self._running = False

        config = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": auto_offset_reset,
            "enable.auto.commit": enable_auto_commit,
            "session.timeout.ms": 30000,
            "max.poll.interval.ms": 600000,
        }

        self._consumer = Consumer(config)
        self._message_handler: Optional[Callable] = None

    def set_message_handler(self, handler: Callable[[bytes], None]):
        self._message_handler = handler

    def start(self):
        self._running = True
        self._consumer.subscribe([self.topic])
        logger.info(f"Kafka consumer subscribed to topic: {self.topic}")

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            while self._running:
                msg = self._consumer.poll(timeout=self.poll_timeout)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    elif msg.error().code() == KafkaError._ALL_BROKERS_DOWN:
                        logger.error("All brokers down, stopping consumer")
                        break
                    else:
                        logger.error(f"Kafka error: {msg.error()}")
                        continue

                if self._message_handler:
                    try:
                        self._message_handler(msg.value())
                    except Exception as e:
                        logger.error(f"Message handler error: {e}")

        except KafkaException as e:
            logger.error(f"Kafka exception: {e}")
        finally:
            self._consumer.close()
            logger.info("Kafka consumer closed")

    def stop(self):
        self._running = False

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, stopping consumer")
        self.stop()
