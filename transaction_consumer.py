from kafka import KafkaConsumer
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionConsumer:
    def __init__(self, bootstrap_servers='localhost:9092', topic='transaction_data_topic',
                 group_id='transaction_consumer_group'):
        """Initialize Kafka consumer"""
        self.topic = topic
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            key_deserializer=lambda x: x.decode('utf-8') if x else None
        )
        
    def consume_messages(self, callback=None):
        """
        Consume messages from Kafka topic
        If callback is provided, it will be called for each message
        """
        try:
            logger.info(f"Starting to consume messages from {self.topic}")
            
            for message in self.consumer:
                transaction = message.value
                logger.info(f"Received message: {transaction}")
                
                # Apply callback function if provided
                if callback:
                    callback(transaction)
                    
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        finally:
            self.close()
    
    def close(self):
        """Close the consumer connection"""
        self.consumer.close()
        logger.info("Consumer connection closed")

def process_transaction(transaction):
    """Example callback function to process transactions"""
    # Here you would typically save to a database, trigger alerts, etc.
    logger.info(f"Processing transaction ID: {transaction.get('id', 'unknown')}")
    
    # Example: Print some transaction details
    if 'amount' in transaction:
        logger.info(f"Transaction amount: {transaction['amount']}")
    
    # You could add more processing logic here

if __name__ == "__main__":
    # Create and run consumer
    consumer = TransactionConsumer()
    
    try:
        # Start consuming with the processing callback
        consumer.consume_messages(callback=process_transaction)
    except Exception as e:
        logger.error(f"Error in consumer: {e}")
        consumer.close()