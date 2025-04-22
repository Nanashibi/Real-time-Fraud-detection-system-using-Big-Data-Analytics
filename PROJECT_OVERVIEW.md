# Database Management System Mini Project Overview

## Project Title
Fraud Detection System with Real-time Transaction Processing

## Team Members
- [Team Member 1]
- [Team Member 2]
- [Team Member 3]

## Problem Statement
This project aims to develop a fraud detection system that can analyze financial transactions in real-time and identify potentially fraudulent activities. The system addresses the challenge of detecting fraud in payment transactions by implementing a machine learning pipeline with a robust database backend for efficient storage and retrieval of transaction data.

## Objectives
- Design and implement a normalized database schema for transaction data storage
- Build a machine learning model for fraud detection using Spark ML
- Create a real-time streaming pipeline for transaction processing
- Implement efficient data transfer between CSV files and PostgreSQL database
- Generate comprehensive fraud detection reports and analytics

## Technologies Used
- Database: PostgreSQL
- Machine Learning: Apache Spark ML
- Stream Processing: Spark Structured Streaming
- Messaging: Apache Kafka
- Languages: Python
- Additional Tools: SQLAlchemy, pandas, PySpark

## System Architecture
The project consists of several interconnected components:

1. **Data Storage Layer**: PostgreSQL database for storing transaction data
2. **Machine Learning Component**: Fraud detection model built with Spark ML
3. **Stream Processing Pipeline**: Real-time transaction processing with Spark Structured Streaming
4. **Test Data Generator**: Tool for simulating transaction streams
5. **Database ETL**: Data transfer between CSV files and PostgreSQL

## Key Components

### Fraud Detection Model (`fraud_detection_model.py`)
- Machine learning pipeline for fraud detection
- Features preprocessing, model training, and evaluation
- Model persistence and loading capabilities
- Support for batch and streaming inference

### CSV to PostgreSQL Transfer (`c_to_p.py`)
- Data transfer tool to load transaction data from CSV files to PostgreSQL
- Database schema creation and management
- Efficient batch processing for large datasets

### Spark Streaming Application (`spark_streaming_app.py`)
- Real-time transaction processing with Spark Structured Streaming
- Integration with Kafka for message handling
- Real-time fraud detection using the pre-trained model
- Resource utilization monitoring and performance tracking

### Test Transaction Producer (`test_transaction_producer.py`)
- Simulation tool for generating transaction streams
- Kafka integration for publishing test data
- Configurable parameters for testing different scenarios

### PostgreSQL Model Inference (`postgres_model_inference.py`)
- Batch processing for fraud detection on stored PostgreSQL data
- Integration between PostgreSQL and Spark ML model
- Performance monitoring and resource usage tracking
- Output generation for fraud detection results

## Data Flow
1. Transaction data is either loaded from CSV files to PostgreSQL or streamed through Kafka
2. The fraud detection model processes transactions in batch or streaming mode
3. Potential fraud cases are identified and flagged
4. Results are stored and can be analyzed for further investigation

## Implementation Highlights
- Real-time fraud detection with sub-second latency
- Machine learning pipeline with feature engineering optimized for fraud detection
- Resource-efficient processing with monitoring capabilities
- Scalable architecture using industry-standard technologies

## Implementation Timeline
- Week 1-2: Database design and schema creation
- Week 3-4: Basic CRUD operations implementation
- Week 5-6: Advanced query development
- Week 7-8: User interface development and integration
- Week 9-10: Testing, optimization, and documentation

## Challenges and Solutions
[Document any significant challenges encountered during development and how they were addressed]

## Future Enhancements
- Development of a web-based dashboard for fraud monitoring
- Additional machine learning models for comparative analysis
- Enhanced feature engineering for improved fraud detection accuracy
- Integration with notification systems for real-time alerts

## Conclusion
[Summary of project outcomes and learning experience]