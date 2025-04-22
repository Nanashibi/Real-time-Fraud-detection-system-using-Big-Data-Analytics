import pandas as pd
import json
import time
import logging
import argparse
import os
from datetime import datetime
from kafka import KafkaProducer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestTransactionProducer:
    """
    Producer to stream test transactions to Kafka.
    Uses the test data separated during model training.
    """
    
    def __init__(self, bootstrap_servers='localhost:9092', 
                 topic='transaction_data_topic',
                 test_data_path=None,
                 delay=1.0):
        """
        Initialize the transaction producer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic: Kafka topic to send transactions to
            test_data_path: Path to the test data CSV
            delay: Delay between transactions in seconds
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.delay = delay
        
        # Default test data path if not provided
        if not test_data_path:
            # First try the model output directory
            ml_output_path = os.path.join('.', 'ml_output', 'test_data.csv')
            if os.path.exists(ml_output_path):
                self.test_data_path = ml_output_path
            else:
                # Fall back to the original data file
                self.test_data_path = os.path.join('data', 't2.csv')
        else:
            self.test_data_path = test_data_path
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        logger.info(f"Transaction producer initialized: {bootstrap_servers} -> {topic}")
    
    def load_test_data(self, limit=None):
        """
        Load test transactions from CSV file.
        
        Args:
            limit: Maximum number of transactions to load
            
        Returns:
            DataFrame with test transactions
        """
        logger.info(f"Loading test data from {self.test_data_path}")
        
        try:
            # Load data from CSV
            df = pd.read_csv(self.test_data_path)
            
            # Apply limit if specified
            if limit and limit > 0:
                df = df.head(limit)
            
            logger.info(f"Loaded {len(df)} test transactions")
            return df
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return pd.DataFrame()
    
    def send_transaction(self, transaction):
        """
        Send a single transaction to Kafka.
        
        Args:
            transaction: Transaction data as dictionary
            
        Returns:
            bool: True if sent successfully
        """
        try:
            # Send to Kafka
            future = self.producer.send(self.topic, value=transaction)
            
            # Wait for the message to be delivered
            future.get(timeout=10)
            return True
        except Exception as e:
            logger.error(f"Error sending transaction: {e}")
            return False
    
    def start_streaming(self, limit=None, batch_size=1):
        """
        Start streaming test transactions to Kafka.
        
        Args:
            limit: Maximum number of transactions to send
            batch_size: Number of transactions to send in each batch
        """
        # Load test transactions
        df = self.load_test_data(limit)
        
        if df.empty:
            logger.error("No test data to stream")
            return
        
        total_transactions = len(df)
        transactions_sent = 0
        start_time = time.time()
        
        logger.info(f"Starting to stream {total_transactions} transactions to topic {self.topic}")
        logger.info(f"Batch size: {batch_size}, Delay between batches: {self.delay}s")
        
        try:
            # Process each transaction
            for i in range(0, total_transactions, batch_size):
                batch = df.iloc[i:i+batch_size]
                batch_start = time.time()
                
                # Process each transaction in the batch
                for _, row in batch.iterrows():
                    # Convert row to dictionary
                    transaction = row.to_dict()
                    
                    # Send to Kafka
                    success = self.send_transaction(transaction)
                    
                    if success:
                        transactions_sent += 1
                    
                # Calculate stats
                elapsed = time.time() - start_time
                progress = transactions_sent / total_transactions * 100
                
                # Log progress
                logger.info(f"Progress: {transactions_sent}/{total_transactions} ({progress:.1f}%) - Elapsed: {elapsed:.1f}s")
                
                # Wait before sending the next batch
                time.sleep(self.delay)
                
        except KeyboardInterrupt:
            logger.info("Streaming interrupted by user")
        finally:
            # Close producer
            self.producer.close()
            
            # Calculate final stats
            elapsed = time.time() - start_time
            logger.info(f"Streaming complete: {transactions_sent}/{total_transactions} transactions sent in {elapsed:.1f}s")
    
    def wait_for_model(self, max_wait_time=60):
        """
        Wait for the model to be ready before streaming.
        
        Args:
            max_wait_time: Maximum wait time in seconds (0 for no waiting)
            
        Returns:
            bool: True if model is ready, False otherwise
        """
        if max_wait_time <= 0:
            # Just check once without waiting
            ready = os.path.exists('model_ready.txt')
            if ready:
                with open('model_ready.txt', 'r') as f:
                    model_time = f.read()
                logger.info(f"Model is ready: {model_time}")
            else:
                logger.warning("Model ready signal not found. Model might not be trained yet.")
            return ready
        
        logger.info(f"Waiting up to {max_wait_time} seconds for model to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            if os.path.exists('model_ready.txt'):
                with open('model_ready.txt', 'r') as f:
                    model_time = f.read()
                logger.info(f"Model is ready: {model_time}")
                return True
            
            # Wait a bit before checking again
            time.sleep(1)
            
            # Log progress every 10 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0:
                logger.info(f"Still waiting for model... ({int(elapsed)}s/{max_wait_time}s)")
        
        logger.error(f"Timed out after waiting {max_wait_time}s for model to be ready")
        return False


def main():
    """Main function to run the transaction producer from command line."""
    parser = argparse.ArgumentParser(description='Stream test transactions to Kafka')
    parser.add_argument('--bootstrap-servers', default='localhost:9092', help='Kafka bootstrap servers')
    parser.add_argument('--topic', default='transaction_data_topic', help='Kafka topic')
    parser.add_argument('--test-data', default=None, help='Path to test data CSV')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between transactions in seconds')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of transactions to send')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of transactions to send in each batch')
    
    args = parser.parse_args()
    
    # Create and start producer
    producer = TestTransactionProducer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        test_data_path=args.test_data,
        delay=args.delay
    )
    
    producer.start_streaming(limit=args.limit, batch_size=args.batch_size)


if __name__ == "__main__":
    main()