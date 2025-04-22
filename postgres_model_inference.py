import pandas as pd
import logging
import time
import json
import os
import argparse
from datetime import datetime
from sqlalchemy import create_engine
import psutil
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, lit

# Use a path that's more likely to be writable
home_dir = os.path.expanduser("~")  # Gets the user's home directory
logs_dir = os.path.join(home_dir, 'fraud_detection_logs')
try:
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, 'postgres_inference.log')
except PermissionError:
    # If home directory doesn't work, try the current directory
    logs_dir = os.path.join(os.getcwd(), 'logs')
    try:
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, 'postgres_inference.log')
    except PermissionError:
        # As a last resort, use /tmp directory which is typically writable
        logs_dir = '/tmp/fraud_detection_logs'
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, 'postgres_inference.log')

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(log_file)
                    ])
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")

class ResourceMonitor:
    """Simple resource monitor for tracking CPU and memory usage"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.metrics = []
    
    def capture_metrics(self, label=''):
        """Capture current resource metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'label': label,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / (1024 * 1024),
            'elapsed_sec': (datetime.now() - self.start_time).total_seconds()
        }
        self.metrics.append(metrics)
        return metrics
    
    def get_summary(self):
        """Get summary of collected metrics"""
        if not self.metrics:
            return {}
        
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_percent'] for m in self.metrics]
        
        return {
            'total_duration_sec': self.metrics[-1]['elapsed_sec'],
            'measurement_count': len(self.metrics),
            'cpu_min': min(cpu_values),
            'cpu_max': max(cpu_values),
            'cpu_avg': sum(cpu_values) / len(cpu_values),
            'memory_min': min(memory_values),
            'memory_max': max(memory_values),
            'memory_avg': sum(memory_values) / len(memory_values),
            'metrics': self.metrics
        }


class PostgresModelInference:
    """Run inference on PostgreSQL data using saved Spark ML model"""
    
    def __init__(self, 
                 db_name='fraud_detection', 
                 model_path='./ml_output/fraud_detection_model',
                 output_dir=None):
        """
        Initialize the inference engine
        
        Args:
            db_name: Name of the PostgreSQL database
            model_path: Path to the saved Spark ML model
            output_dir: Directory to store output files
        """
        self.db_name = db_name
        self.model_path = model_path
        
        # Set up output directory
        if output_dir is None:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_dir = os.path.join(project_dir, 'postgres_inference_output')
        else:
            self.output_dir = output_dir
            
        # Create output directory with error handling
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Output directory set to: {self.output_dir}")
        except PermissionError:
            import tempfile
            self.output_dir = os.path.join(tempfile.gettempdir(), 'postgres_inference_output')
            os.makedirs(self.output_dir, exist_ok=True)
            logger.warning(f"Permission denied on original path. Using temporary directory: {self.output_dir}")
        
        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor()
        
        # Set up stats tracking
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'data_load_time_sec': 0,
            'prediction_time_sec': 0,
            'total_batches': 0,
            'total_records': 0,
            'total_fraud_detected': 0
        }
        
        # Connect to database
        self._connect_to_db()
        
        # Initialize Spark session
        self.spark = self._create_spark_session()
        
        # Load ML model
        self.model = self._load_model()
    
    def _connect_to_db(self):
        """Connect to PostgreSQL database using peer authentication"""
        try:
            # Simplified PostgreSQL connection using peer authentication
            connection_string = f"postgresql:///{self.db_name}"
            self.engine = create_engine(connection_string)
            
            # Test connection
            with self.engine.connect() as conn:
                pass
                
            logger.info(f"Connected to PostgreSQL database: {self.db_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def _create_spark_session(self):
        """Create a Spark session"""
        logger.info("Initializing Spark session")
        self.resource_monitor.capture_metrics('before_spark_init')
        
        spark = SparkSession.builder \
            .appName("PostgresModelInference") \
            .config("spark.sql.shuffle.partitions", "2") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("ERROR")
        
        self.resource_monitor.capture_metrics('after_spark_init')
        logger.info("Spark session initialized")
        
        return spark
    
    def _load_model(self):
        """Load the saved Spark ML model"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.resource_monitor.capture_metrics('before_model_load')
            
            model = PipelineModel.load(self.model_path)
            
            self.resource_monitor.capture_metrics('after_model_load')
            logger.info("Model loaded successfully")
            
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _get_table_info(self):
        """Get information about the test data table"""
        try:
            with self.engine.connect() as conn:
                # Check if transactions table exists - using text() for SQL expressions
                from sqlalchemy import text
                result = conn.execute(text("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'transactions')")).fetchone()
                if not result or not result[0]:
                    raise ValueError("'transactions' table does not exist in the database")
                
                # Get record count
                count_result = conn.execute(text("SELECT COUNT(*) FROM transactions")).fetchone()
                total_count = count_result[0] if count_result else 0
                
                # Get column names
                columns_result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'transactions'")).fetchall()
                columns = [row[0] for row in columns_result]
                
                logger.info(f"Found 'transactions' table with {total_count} records")
                logger.info(f"Columns: {', '.join(columns)}")
                
                return {
                    'exists': True,
                    'count': total_count,
                    'columns': columns
                }
        except Exception as e:
            logger.error(f"Error getting table information: {e}")
            return {
                'exists': False,
                'count': 0,
                'columns': []
            }
    
    def process_batches(self, batch_size=100,max_batches=None):
        """
        Process data in batches and make predictions
        
        Args:
            batch_size: Number of records in each batch
            max_batches: Maximum number of batches to process (None for all)
            
        Returns:
            Dictionary with prediction results
        """
        self.resource_monitor.capture_metrics('before_batch_processing')
        
        try:
            # Get table information
            table_info = self._get_table_info()
            if not table_info['exists']:
                raise ValueError("Transactions table not found in database")
            
            total_count = table_info['count']
            total_batches = (total_count + batch_size - 1) // batch_size
            
            if max_batches:
                total_batches = min(total_batches, max_batches)
                logger.info(f"Processing {total_batches} batches (limited by max_batches)")
            else:
                logger.info(f"Processing {total_batches} batches to cover all {total_count} records")
            
            # Process batches
            batch_num = 0
            total_processed = 0
            total_fraud_detected = 0
            start_time = time.time()
            
            # Create output CSV for fraud predictions
            fraud_output_path = os.path.join(self.output_dir, 'fraud_predictions.csv')
            with open(fraud_output_path, 'w') as f:
                # We'll write the header later when we know the columns
                pass
            
            header_written = False
            
            while batch_num < total_batches:
                batch_start_time = time.time()
                self.resource_monitor.capture_metrics(f'batch_{batch_num}_start')
                
                # Load batch from PostgreSQL
                offset = batch_num * batch_size
                query = f"SELECT * FROM transactions OFFSET {offset} LIMIT {batch_size}"
                
                logger.info(f"Loading batch {batch_num+1}/{total_batches} (offset={offset}, limit={batch_size})")
                df_pandas = pd.read_sql(query, self.engine)
                
                if df_pandas.empty:
                    logger.info(f"No more records to process after batch {batch_num}")
                    break
                
                records_in_batch = len(df_pandas)
                total_processed += records_in_batch
                
                # Convert to Spark DataFrame
                df_spark = self.spark.createDataFrame(df_pandas)
                logger.info(f"Converted PostgreSQL data to Spark DataFrame with {records_in_batch} records")
                
                # Make predictions with Spark ML model
                predictions = self.model.transform(df_spark)
                
                # Filter fraud predictions
                fraud_predictions = predictions.filter(col("prediction") == 1.0)
                fraud_count = fraud_predictions.count()
                total_fraud_detected += fraud_count
                
                # Log results for this batch
                logger.info(f"Batch {batch_num+1}: Processed {records_in_batch} records, detected {fraud_count} potential fraud cases")
                
                # If we found fraud transactions, save them
                if fraud_count > 0:
                    # Convert fraud predictions to pandas and append to CSV
                    fraud_pandas = fraud_predictions.toPandas()
                    
                    # Write header only once
                    fraud_pandas.to_csv(fraud_output_path, mode='a', header=not header_written, index=False)
                    if not header_written:
                        header_written = True
                
                batch_duration = time.time() - batch_start_time
                logger.info(f"Batch {batch_num+1} processing time: {batch_duration:.2f} seconds")
                
                self.resource_monitor.capture_metrics(f'batch_{batch_num}_end')
                
                # Update statistics
                self.stats['total_batches'] += 1
                self.stats['total_records'] += records_in_batch
                self.stats['total_fraud_detected'] += fraud_count
                
                batch_num += 1
            
            # Calculate overall duration
            total_duration = time.time() - start_time
            self.stats['prediction_time_sec'] = total_duration
            
            # Log final results
            logger.info(f"Completed processing {self.stats['total_batches']} batches with {self.stats['total_records']} records")
            logger.info(f"Detected {self.stats['total_fraud_detected']} potential fraud cases ({self.stats['total_fraud_detected'] / self.stats['total_records'] * 100:.2f}%)")
            logger.info(f"Total processing time: {total_duration:.2f} seconds")
            logger.info(f"Fraud predictions saved to {fraud_output_path}")
            
            self.resource_monitor.capture_metrics('after_batch_processing')
            
            # Save statistics to file
            self._save_results()
            
            return {
                'total_batches': self.stats['total_batches'],
                'total_records': self.stats['total_records'],
                'total_fraud_detected': self.stats['total_fraud_detected'],
                'fraud_percentage': self.stats['total_fraud_detected'] / self.stats['total_records'] * 100 if self.stats['total_records'] > 0 else 0,
                'processing_time_sec': total_duration,
                'fraud_output_path': fraud_output_path
            }
            
        except Exception as e:
            logger.error(f"Error processing batches: {e}", exc_info=True)
            self.resource_monitor.capture_metrics('batch_processing_error')
            raise
    
    def _save_results(self):
        """Save statistics and resource usage to files"""
        # Update final timestamp
        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['total_duration_sec'] = (datetime.now() - datetime.fromisoformat(self.stats['start_time'])).total_seconds()
        
        # Save statistics
        stats_path = os.path.join(self.output_dir, 'inference_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Save resource usage
        resource_summary = self.resource_monitor.get_summary()
        resource_path = os.path.join(self.output_dir, 'resource_usage.json')
        with open(resource_path, 'w') as f:
            json.dump(resource_summary, f, indent=2)
            
        logger.info(f"Statistics saved to {stats_path}")
        logger.info(f"Resource usage saved to {resource_path}")
    
    def close(self):
        """Close resources"""
        logger.info("Closing resources")
        
        # Stop Spark session
        if hasattr(self, 'spark') and self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")


def main():
    """Main function to run inference from command line"""
    parser = argparse.ArgumentParser(description='Run fraud detection inference on PostgreSQL data using saved Spark ML model')
    parser.add_argument('--db-name', default='fraud_detection', help='PostgreSQL database name')
    parser.add_argument('--model-path', default='./ml_output/fraud_detection_model', help='Path to saved Spark ML model')
    parser.add_argument('--batch-size', type=int, default=200, help='Number of records in each batch')
    parser.add_argument('--max-batches', type=int, default=None, help='Maximum number of batches to process (None for all)')
    parser.add_argument('--output-dir', default=None, help='Directory to store output files')
    
    args = parser.parse_args()
    
    inference_engine = PostgresModelInference(
        db_name=args.db_name,
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    try:
        inference_engine.process_batches(
            batch_size=args.batch_size,
            max_batches=args.max_batches
        )
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Error during inference: {e}")
    finally:
        inference_engine.close()


if __name__ == "__main__":
    main()