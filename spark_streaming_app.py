from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import json
import logging
import os
import time
from datetime import datetime
from kafka import KafkaProducer
import argparse
import psutil

# Import our ML model module
from fraud_detection_model import FraudDetectionModel

# Set up logging
log_dir = './spark_fraud_detection'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'spark_streaming_app.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {log_file}")

# Schema for the transaction data
transaction_schema = StructType([
    StructField("step", IntegerType(), True),
    StructField("type", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("nameOrig", StringType(), True),
    StructField("oldbalanceOrg", DoubleType(), True),
    StructField("newbalanceOrig", DoubleType(), True),
    StructField("nameDest", StringType(), True),
    StructField("oldbalanceDest", DoubleType(), True),
    StructField("newbalanceDest", DoubleType(), True),
    StructField("isFraud", IntegerType(), True),
    StructField("isFlaggedFraud", IntegerType(), True)
])

class ResourceUtilization:
    """Track system resource utilization"""
    
    def __init__(self, log_dir='./spark_fraud_detection'):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'resource_usage.log')
        self.summary_file = os.path.join(log_dir, 'resource_summary.json')
        self.start_time = datetime.now()
        self.metrics = []
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("timestamp,cpu_percent,memory_percent,memory_used_mb,disk_percent,elapsed_seconds,note\n")
        
        logger.info(f"Resource utilization tracking initialized. Logging to {self.log_file}")
    
    def capture(self, note=""):
        """Capture current resource utilization"""
        try:
            timestamp = datetime.now().isoformat()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            disk_percent = psutil.disk_usage('/').percent
            elapsed_seconds = (datetime.now() - self.start_time).total_seconds()
            
            metrics = {
                "timestamp": timestamp,
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_used_mb": memory_used_mb,
                "disk_percent": disk_percent,
                "elapsed_seconds": elapsed_seconds,
                "note": note
            }
            
            self.metrics.append(metrics)
            
            # Write to log file
            with open(self.log_file, 'a') as f:
                f.write(f"{timestamp},{cpu_percent},{memory_percent},{memory_used_mb},{disk_percent},{elapsed_seconds},{note}\n")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error capturing resource usage: {e}")
            return {}
    
    def get_summary(self):
        """Get summary of resource utilization"""
        if not self.metrics:
            return {}
        
        try:
            cpu_values = [m['cpu_percent'] for m in self.metrics]
            memory_values = [m['memory_percent'] for m in self.metrics]
            memory_mb_values = [m['memory_used_mb'] for m in self.metrics]
            
            summary = {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "samples": len(self.metrics),
                "cpu_min": min(cpu_values),
                "cpu_max": max(cpu_values),
                "cpu_avg": sum(cpu_values) / len(cpu_values),
                "memory_min_percent": min(memory_values),
                "memory_max_percent": max(memory_values),
                "memory_avg_percent": sum(memory_values) / len(memory_values),
                "memory_min_mb": min(memory_mb_values),
                "memory_max_mb": max(memory_mb_values),
                "memory_avg_mb": sum(memory_mb_values) / len(memory_mb_values)
            }
            
            # Write summary to file
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating resource summary: {e}")
            return {"error": str(e)}

class SparkStreamingApp:
    """Spark Streaming Application for Fraud Detection"""
    
    def __init__(self, bootstrap_servers='localhost:9092', 
                 input_topic='transaction_data_topic',
                 output_topic='fraud_alerts_topic',
                 checkpoint_dir='./checkpoint',
                 model_path='./ml_output/fraud_detection_model'):
        """
        Initialize the Spark streaming application.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            input_topic: Topic for receiving transactions
            output_topic: Topic for sending fraud alerts
            checkpoint_dir: Directory for Spark checkpointing
            model_path: Path to pre-trained model (None to train a new model)
        """
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.checkpoint_dir = checkpoint_dir
        self.model_path = model_path
        self.log_dir = log_dir  # Use the global log_dir
        
        # Ensure directories exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize resource utilization tracking
        self.resource_util = ResourceUtilization(log_dir=self.log_dir)
        self.resource_util.capture("Application initialization")
        
        # Initialize execution stats log file
        self.exec_stats_file = os.path.join(self.log_dir, 'streaming_stats.json')
        
        # Initialize stats
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'total_batches': 0,
            'total_records': 0,
            'total_fraud_detected': 0,
            'total_processing_time_ms': 0,
            'last_update': datetime.now().isoformat()
        }
        
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("FraudDetectionStreaming") \
            .config("spark.streaming.stopGracefullyOnShutdown", "true") \
            .config("spark.sql.shuffle.partitions", "2") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2") \
            .getOrCreate()
            
        self.spark.sparkContext.setLogLevel("ERROR")
        
        # Initialize model
        self.ml_model = FraudDetectionModel(log_dir=os.path.join(log_dir, 'ml'))
        
        # Initialize Kafka producer for alerts
        self.alert_producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        logger.info(f"Spark Streaming App initialized: {bootstrap_servers}")
        self.resource_util.capture("Spark initialized")
    
    def log_execution_stats(self, batch_stats):
        """Log execution statistics to a file"""
        # Update cumulative statistics
        self.stats['total_batches'] += 1
        self.stats['total_records'] += batch_stats.get('records', 0)
        self.stats['total_fraud_detected'] += batch_stats.get('fraud_detected', 0)
        self.stats['total_processing_time_ms'] += batch_stats.get('processing_time_ms', 0)
        self.stats['last_update'] = datetime.now().isoformat()
        
        # Calculate averages
        avg_batch_size = self.stats['total_records'] / self.stats['total_batches'] if self.stats['total_batches'] > 0 else 0
        avg_processing_time = self.stats['total_processing_time_ms'] / self.stats['total_batches'] if self.stats['total_batches'] > 0 else 0
        fraud_percentage = (self.stats['total_fraud_detected'] / self.stats['total_records'] * 100) if self.stats['total_records'] > 0 else 0
        
        # Add resource utilization
        current_resources = self.resource_util.capture("Logging execution stats")
        
        # Prepare stats for logging
        log_stats = {
            **self.stats,
            'avg_batch_size': avg_batch_size,
            'avg_processing_time_ms': avg_processing_time,
            'fraud_percentage': fraud_percentage,
            'latest_batch': batch_stats,
            'current_resources': current_resources
        }
        
        # Write to log file
        with open(self.exec_stats_file, 'w') as f:
            json.dump(log_stats, f, indent=2)
        
        logger.info(f"Updated execution stats: {self.stats['total_batches']} batches, {self.stats['total_records']} records, {fraud_percentage:.2f}% fraud")
    
    def prepare_model(self, training_data_path='data/t2.csv'):
        """
        Prepare the ML model - either load or train it.
        
        Args:
            training_data_path: Path to training data (used if no model is provided)
        """
        start_time = time.time()
        
        # Capture resource usage before model preparation
        self.resource_util.capture("Before model preparation")
        
        # Check for model in the default output location
        model_output_path = './ml_output/fraud_detection_model'
        signal_file = 'model_ready.txt'
        model_loaded = False
        
        # First check if model exists in the default output directory
        logger.info(f"Checking for model in {model_output_path}")
        if os.path.exists(model_output_path) and os.path.isdir(model_output_path):
            try:
                logger.info(f"Found model directory, attempting to load from {model_output_path}")
                self.ml_model.model = self.ml_model.load_model(model_output_path)
                logger.info("Successfully loaded pre-trained model from default path")
                model_loaded = True
                self.resource_util.capture("Model loaded from default path")
            except Exception as e:
                logger.warning(f"Error loading model from {model_output_path}: {e}")
                self.resource_util.capture("Failed to load model from default path")
        
        # If model_path is specified and exists, try to load from there as a backup option
        if not model_loaded and self.model_path and os.path.exists(self.model_path) and os.path.isdir(self.model_path):
            try:
                # Load existing model from specified path
                logger.info(f"Attempting to load model from specified path {self.model_path}")
                self.ml_model.model = self.ml_model.load_model(self.model_path)
                logger.info("Successfully loaded pre-trained model from specified path")
                model_loaded = True
                self.resource_util.capture("Model loaded from specified path")
            except Exception as e:
                logger.warning(f"Error loading model from {self.model_path}: {e}")
                self.resource_util.capture("Failed to load model from specified path")
            
        # Only train if we couldn't load a model
        if not model_loaded:
            logger.info(f"No pre-trained model could be loaded. Training new model using {training_data_path}")
            self.resource_util.capture("Starting model training")
            
            # Split data and only use training portion
            train_df, _ = self.ml_model.prepare_data(training_data_path, test_fraction=0.3)
            self.ml_model.build_pipeline()
            self.ml_model.model = self.ml_model.train(train_df)
            
            # Save model
            model_save_path = self.ml_model.save_model()
            logger.info(f"New model trained and saved to {model_save_path}")
            
            self.resource_util.capture("Model training completed")
        
        prep_time = time.time() - start_time
        logger.info(f"Model preparation completed in {prep_time:.2f} seconds")
        
        # Initialize stats with model info
        initial_stats = {
            'application_start': self.stats['start_time'],
            'model_preparation_time_seconds': prep_time,
            'model_newly_trained': not model_loaded,
            'model_preparation_complete': datetime.now().isoformat()
        }
        with open(self.exec_stats_file, 'w') as f:
            json.dump(initial_stats, f, indent=2)
        
        return self.ml_model.model
    
    def process_batch(self, batch_df, batch_id):
        """
        Process a batch of transactions.
        
        Args:
            batch_df: DataFrame with batch of transactions
            batch_id: Batch ID
        """
        try:
            # Start timer
            batch_start_time = time.time()
            
            # Capture resource usage at start of batch
            self.resource_util.capture(f"Start batch {batch_id}")
            
            if batch_df.isEmpty():
                logger.info(f"Batch {batch_id} is empty")
                # Log empty batch stats
                batch_stats = {
                    'batch_id': batch_id,
                    'timestamp': datetime.now().isoformat(),
                    'records': 0,
                    'fraud_detected': 0,
                    'processing_time_ms': 0,
                    'empty_batch': True
                }
                self.log_execution_stats(batch_stats)
                return
                
            # Count records
            record_count = batch_df.count()
            logger.info(f"Processing batch {batch_id} with {record_count} records")
            
            # Make predictions
            predictions = self.ml_model.model.transform(batch_df)
            
            # Filter fraud predictions
            fraud_predictions = predictions.filter(col("prediction") == 1.0)
            fraud_count = fraud_predictions.count()
            
            if fraud_count > 0:
                logger.info(f"Found {fraud_count} potential fraud transactions")
                
                # Send fraud alerts to Kafka
                fraud_rows = fraud_predictions.collect()
                for row in fraud_rows:
                    alert = {
                        "nameOrig": row.nameOrig if hasattr(row, "nameOrig") else "unknown",
                        "nameDest": row.nameDest if hasattr(row, "nameDest") else "unknown",
                        "amount": float(row.amount) if hasattr(row, "amount") else 0.0,
                        "prediction": float(row.prediction),
                        "probability": float(row.probability[1])  # Probability of being fraud
                    }
                    
                    # Send alert to Kafka
                    self.alert_producer.send(self.output_topic, value=alert)
                    logger.info(f"Sent fraud alert for transaction from {alert['nameOrig']} to {alert['nameDest']}")
            else:
                logger.info("No fraud transactions detected in this batch")
            
            # Calculate processing time
            processing_time_ms = int((time.time() - batch_start_time) * 1000)
            
            # Capture resource usage at end of batch
            self.resource_util.capture(f"End batch {batch_id}")
            
            # Log batch stats
            batch_stats = {
                'batch_id': batch_id,
                'timestamp': datetime.now().isoformat(),
                'records': record_count,
                'fraud_detected': fraud_count,
                'processing_time_ms': processing_time_ms,
                'empty_batch': False
            }
            self.log_execution_stats(batch_stats)
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}", exc_info=True)
            
            # Capture resource usage on error
            self.resource_util.capture(f"Error in batch {batch_id}")
            
            # Log error in stats
            error_stats = {
                'batch_id': batch_id,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'records': 0,
                'fraud_detected': 0,
                'processing_time_ms': 0
            }
            self.log_execution_stats(error_stats)
    
    def start_streaming(self):
        """Start the streaming process to detect fraud in real-time."""
        logger.info(f"Setting up streaming from Kafka topic: {self.input_topic}")
        self.resource_util.capture("Setting up streaming")
        
        try:
            # Create DataFrame representing the stream of input from Kafka
            df = self.spark \
                .readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", self.bootstrap_servers) \
                .option("subscribe", self.input_topic) \
                .option("startingOffsets", "earliest") \
                .option("failOnDataLoss", "false") \
                .load()
                
            # Parse value from Kafka
            transactions = df.selectExpr("CAST(value AS STRING)") \
                .select(from_json(col("value"), transaction_schema).alias("data")) \
                .select("data.*")
            
            # Start streaming query
            query = transactions \
                .writeStream \
                .foreachBatch(self.process_batch) \
                .option("checkpointLocation", self.checkpoint_dir) \
                .start()
                
            logger.info("Streaming query started - waiting for data from Kafka")
            self.resource_util.capture("Streaming query started")
            
            # Wait for termination
            query.awaitTermination()
            
        except Exception as e:
            logger.error(f"Error in streaming process: {e}", exc_info=True)
            self.resource_util.capture(f"Streaming error: {str(e)[:50]}")
    
    def close(self):
        """Close resources."""
        logger.info("Closing resources")
        self.resource_util.capture("Closing resources")
        
        if hasattr(self, 'alert_producer'):
            self.alert_producer.close()
        
        if hasattr(self, 'ml_model'):
            self.ml_model.close()
            
        if hasattr(self, 'spark'):
            self.spark.stop()
        
        # Generate resource utilization summary
        resource_summary = self.resource_util.get_summary()
        logger.info(f"Resource utilization summary: CPU avg={resource_summary.get('cpu_avg', 0):.1f}%, "
                    f"Memory avg={resource_summary.get('memory_avg_percent', 0):.1f}%")
        
        logger.info("All resources closed")


def main():
    """Main function to run the streaming application from command line."""
    parser = argparse.ArgumentParser(description='Spark Streaming Fraud Detection')
    parser.add_argument('--bootstrap-servers', default='localhost:9092', help='Kafka bootstrap servers')
    parser.add_argument('--input-topic', default='transaction_data_topic', help='Input Kafka topic')
    parser.add_argument('--output-topic', default='fraud_alerts_topic', help='Output Kafka topic')
    parser.add_argument('--checkpoint-dir', default='./checkpoint', help='Checkpoint directory')
    parser.add_argument('--model-path', default='./ml_output/fraud_detection_model', 
                        help='Path to pre-trained model directory')
    parser.add_argument('--training-data', default='data/t2.csv', help='Path to training data')
    
    args = parser.parse_args()
    
    # Create and run application
    app = SparkStreamingApp(
        bootstrap_servers=args.bootstrap_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        checkpoint_dir=args.checkpoint_dir,
        model_path=args.model_path
    )
    
    try:
        # Prepare the model (train or load)
        app.prepare_model(args.training_data)
        
        # Start streaming
        app.start_streaming()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()