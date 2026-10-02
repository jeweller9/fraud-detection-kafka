import json
import logging
import os
import sys

import pandas as pd
from confluent_kafka import Consumer, Producer

sys.path.append(os.path.abspath('./src'))
from preprocessing import load_train_data, run_preproc  # noqa: E402
from scorer import make_pred  # noqa: E402

os.makedirs('/app/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TRANSACTIONS_TOPIC = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")
SCORING_TOPIC = os.getenv("KAFKA_SCORING_TOPIC", "scores")


class ProcessingService:
    def __init__(self):
        self.consumer = Consumer({
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'group.id': 'ml-scorer',
            'auto.offset.reset': 'earliest'
        })
        self.consumer.subscribe([TRANSACTIONS_TOPIC])
        self.producer = Producer({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})

        # Данные для препроцессинга грузятся один раз при старте
        self.train = load_train_data()

    def score_message(self, raw_value):
        """Одно сообщение из Kafka -> dict {transaction_id, score, fraud_flag}."""
        data = json.loads(raw_value.decode('utf-8'))
        transaction_id = data['transaction_id']
        input_df = pd.DataFrame([data['data']])

        processed_df = run_preproc(self.train, input_df)
        submission = make_pred(processed_df, "kafka_stream")

        return {
            'transaction_id': transaction_id,
            'score': float(submission.loc[0, 'score']),
            'fraud_flag': int(submission.loc[0, 'fraud_flag']),
        }

    def process_messages(self):
        while True:
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error("Kafka error: %s", msg.error())
                continue
            try:
                result = self.score_message(msg.value())
                self.producer.produce(
                    SCORING_TOPIC,
                    key=result['transaction_id'],
                    value=json.dumps(result)
                )
                self.producer.poll(0)
            except Exception as e:
                logger.exception("Error processing message: %s", e)


if __name__ == "__main__":
    logger.info('Starting Kafka ML scoring service...')
    service = ProcessingService()
    try:
        service.process_messages()
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
    finally:
        service.producer.flush()
        service.consumer.close()
