import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
import logging
import os
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudDetectionModel:
    """
    Fraud Detection Model using Spark ML.
    """
    
    def __init__(self, log_dir='./ml_logs', 
                 output_dir='./ml_output', 
                 test_file_path=None):
        """
        Initialize the fraud detection model.
        
        Args:
            log_dir: Directory for logs
            output_dir: Directory for model output
            test_file_path: Path to save test data for later streaming
        """
        self.log_dir = log_dir
        self.output_dir = output_dir
        
        # Default test file path if not provided
        if not test_file_path:
            self.test_file_path = os.path.join(self.output_dir, 'test_data.csv')
        else:
            self.test_file_path = test_file_path
        
        # Create directories if they don't exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("FraudDetectionModel") \
            .config("spark.sql.shuffle.partitions", "2") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("ERROR")
        
        # Initialize model and pipeline
        self.model = None
        self.pipeline = None
        self.feature_columns = []
        
        logger.info("Fraud Detection Model initialized")
    
    def prepare_data(self, file_path, test_fraction=0.2):
        """
        Load and prepare data for model training.
        
        Args:
            file_path: Path to the data file
            test_fraction: Fraction of data to use for testing
            
        Returns:
            tuple: (train_df, test_df)
        """
        logger.info(f"Loading data from {file_path}")
        
        try:
            # Load data
            df = self.spark.read.csv(file_path, header=True, inferSchema=True)
            
            # Basic data info
            total_rows = df.count()
            fraud_count = df.filter(col("isFraud") == 1).count()
            fraud_pct = fraud_count / total_rows * 100
            
            logger.info(f"Loaded {total_rows} transactions, {fraud_count} fraud cases ({fraud_pct:.2f}%)")
            
            # Split data
            train_df, test_df = df.randomSplit([1 - test_fraction, test_fraction], seed=42)
            
            logger.info(f"Split data: {train_df.count()} training rows, {test_df.count()} test rows")
            
            # Save test data for later streaming tests
            test_df_pd = test_df.toPandas()
            os.makedirs(os.path.dirname(self.test_file_path), exist_ok=True)
            test_df_pd.to_csv(self.test_file_path, index=False)
            logger.info(f"Saved test data to {self.test_file_path} for later streaming")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def build_pipeline(self):
        """
        Build the ML pipeline for fraud detection.
        """
        logger.info("Building ML pipeline")
        
        try:
            # Define categorical and numeric columns
            categorical_cols = ["type"]
            numeric_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", 
                            "oldbalanceDest", "newbalanceDest"]
            
            # Store for future reference
            self.feature_columns = categorical_cols + numeric_cols
            
            # Create stages list
            stages = []
            
            # Process categorical columns
            for col_name in categorical_cols:
                # String indexer
                indexer = StringIndexer(
                    inputCol=col_name,
                    outputCol=f"{col_name}_indexed",
                    handleInvalid="keep"
                )
                stages.append(indexer)
                
                # One-hot encoder
                encoder = OneHotEncoder(
                    inputCols=[f"{col_name}_indexed"],
                    outputCols=[f"{col_name}_encoded"]
                )
                stages.append(encoder)
            
            # Create input columns list for assembler
            input_cols = []
            input_cols.extend([f"{col}_encoded" for col in categorical_cols])
            input_cols.extend(numeric_cols)
            
            # Feature assembler
            assembler = VectorAssembler(
                inputCols=input_cols,
                outputCol="features_raw",
                handleInvalid="keep"
            )
            stages.append(assembler)
            
            # Standard scaler
            scaler = StandardScaler(
                inputCol="features_raw",
                outputCol="features",
                withStd=True, 
                withMean=True
            )
            stages.append(scaler)
            
            # Create logistic regression model
            lr = LogisticRegression(
                featuresCol="features",
                labelCol="isFraud",
                maxIter=10,
                regParam=0.3,
                elasticNetParam=0.8,
                threshold=0.5,
                standardization=True
            )
            stages.append(lr)
            
            # Create pipeline
            self.pipeline = Pipeline(stages=stages)
            logger.info(f"ML pipeline built with {len(stages)} stages")
            
        except Exception as e:
            logger.error(f"Error building pipeline: {e}")
            raise
    
    def train(self, train_df):
        """
        Train the model using the pipeline.
        
        Args:
            train_df: Training data DataFrame
            
        Returns:
            PipelineModel: Trained model
        """
        if not self.pipeline:
            logger.warning("Pipeline not built yet, building now")
            self.build_pipeline()
        
        logger.info("Training fraud detection model")
        
        try:
            # Get counts for class weighting
            fraud_count = train_df.filter(col("isFraud") == 1).count()
            non_fraud_count = train_df.filter(col("isFraud") == 0).count()
            
            # Calculate class weights
            if fraud_count > 0:
                weight_ratio = non_fraud_count / fraud_count
                logger.info(f"Applying class weights: fraud weight = {weight_ratio:.2f}")
                
                # Add weight column based on class
                weighted_df = train_df.withColumn(
                    "classWeight",
                    (col("isFraud") * weight_ratio) + (1.0 - col("isFraud"))
                )
            else:
                logger.warning("No fraud cases in training data - not applying weights")
                weighted_df = train_df
            
            # Fit model
            start_time = datetime.now()
            
            model = self.pipeline.fit(weighted_df)
            
            training_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Model training completed in {training_time:.2f} seconds")
            
            self.model = model
            return model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def evaluate(self, test_df):
        """
        Evaluate the trained model.
        
        Args:
            test_df: Test data DataFrame
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.model:
            raise ValueError("Model has not been trained yet")
        
        logger.info("Evaluating fraud detection model")
        
        try:
            # Make predictions
            predictions = self.model.transform(test_df)
            
            # Create evaluator
            evaluator = BinaryClassificationEvaluator(
                labelCol="isFraud",
                rawPredictionCol="rawPrediction",
                metricName="areaUnderROC"
            )
            
            # Compute AUC
            auc = evaluator.evaluate(predictions)
            
            # Compute accuracy manually
            correct_predictions = predictions.filter(
                (col("prediction") == 0) & (col("isFraud") == 0) |
                (col("prediction") == 1) & (col("isFraud") == 1)
            ).count()
            
            total = predictions.count()
            accuracy = correct_predictions / total if total > 0 else 0
            
            # Get confusion matrix
            tp = predictions.filter((col("prediction") == 1) & (col("isFraud") == 1)).count()
            fp = predictions.filter((col("prediction") == 1) & (col("isFraud") == 0)).count()
            tn = predictions.filter((col("prediction") == 0) & (col("isFraud") == 0)).count()
            fn = predictions.filter((col("prediction") == 0) & (col("isFraud") == 1)).count()
            
            # Compute precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Create metrics dictionary
            metrics = {
                "auc": auc,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": {
                    "tp": tp, "fp": fp,
                    "tn": tn, "fn": fn
                }
            }
            
            # Log metrics
            logger.info(f"Evaluation metrics: AUC = {auc:.4f}, Accuracy = {accuracy:.4f}")
            logger.info(f"Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
            logger.info(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            
            # Save metrics
            metrics_path = os.path.join(self.output_dir, 'model_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def save_model(self, model_path=None):
        """
        Save the trained model and feature columns.
        
        Args:
            model_path: Path to save the model (optional)
            
        Returns:
            str: Path where the model was saved
        """
        if not self.model:
            raise ValueError("No model to save - train a model first")
        
        # Use default path if not provided
        if not model_path:
            model_path = os.path.join(self.output_dir, 'fraud_detection_model')
        
        logger.info(f"Saving model to {model_path}")
        
        try:
            # Save the model
            self.model.write().overwrite().save(model_path)
            
            # Save feature columns list
            feature_path = os.path.join(self.output_dir, 'feature_columns.json')
            with open(feature_path, 'w') as f:
                json.dump(self.feature_columns, f)
            
            # Create a signal file to indicate model is ready
            with open('model_ready.txt', 'w') as f:
                f.write(datetime.now().isoformat())
                
            logger.info("Model and feature columns saved successfully")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_path):
        """
        Load a previously saved model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            PipelineModel: Loaded model
        """
        logger.info(f"Loading model from {model_path}")
        
        try:
            # Load the model
            loaded_model = PipelineModel.load(model_path)
            
            # Try to load feature columns
            feature_path = os.path.join(os.path.dirname(model_path), 'feature_columns.json')
            if os.path.exists(feature_path):
                with open(feature_path, 'r') as f:
                    self.feature_columns = json.load(f)
                logger.info(f"Loaded feature columns: {self.feature_columns}")
            
            self.model = loaded_model
            logger.info("Model loaded successfully")
            return loaded_model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, df):
        """
        Make predictions using the trained model.
        
        Args:
            df: DataFrame with input data
            
        Returns:
            DataFrame: Prediction results
        """
        if not self.model:
            raise ValueError("No model available for prediction")
        
        try:
            # Make predictions
            predictions = self.model.transform(df)
            
            # Return predictions with relevant columns
            return predictions.select(
                "*", 
                col("prediction").cast("int").alias("fraud_prediction"),
                col("probability")[1].alias("fraud_probability")
            )
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def close(self):
        """Close resources."""
        if hasattr(self, 'spark') and self.spark:
            self.spark.stop()
            logger.info("SparkSession stopped")


if __name__ == "__main__":
    """
    Run model training and evaluation as a standalone script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Fraud Detection Model')
    parser.add_argument('--data', default='data/t2.csv', help='Path to the data file')
    parser.add_argument('--test-fraction', type=float, default=0.2, 
                        help='Fraction of data to use for testing')
    parser.add_argument('--output-dir', default='./ml_output', 
                        help='Directory to save model and outputs')
    
    args = parser.parse_args()
    
    try:
        # Create model
        model = FraudDetectionModel(output_dir=args.output_dir)
        
        # Prepare data
        train_df, test_df = model.prepare_data(args.data, test_fraction=args.test_fraction)
        
        # Build pipeline
        model.build_pipeline()
        
        # Train model
        trained_model = model.train(train_df)
        
        # Evaluate model
        metrics = model.evaluate(test_df)
        
        # Save model
        model_path = model.save_model()
        
        logger.info(f"Model training and evaluation complete. Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Error in model training process: {e}", exc_info=True)
    finally:
        # Clean up resources
        if 'model' in locals():
            model.close()