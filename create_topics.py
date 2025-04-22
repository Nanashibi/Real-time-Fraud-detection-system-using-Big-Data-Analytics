from kafka.admin import KafkaAdminClient, NewTopic
import logging
import subprocess
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_topics(bootstrap_servers='localhost:9092'):
    """
    Delete Kafka topics if they exist using the kafka-topics.sh script
    """
    kafka_bin_path = "/usr/local/kafka/bin"
    topics_to_delete = ["transaction_data_topic", "fraud_alerts_topic"]
    
    logger.info("Attempting to delete existing topics")
    
    for topic in topics_to_delete:
        try:
            # Command to delete the topic
            delete_cmd = [
                f"{kafka_bin_path}/kafka-topics.sh",
                "--bootstrap-server", bootstrap_servers,
                "--delete",
                "--topic", topic
            ]
            
            logger.info(f"Executing: {' '.join(delete_cmd)}")
            result = subprocess.run(delete_cmd, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully requested deletion of topic '{topic}'")
            else:
                logger.warning(f"Topic deletion request for '{topic}' might have failed: {result.stderr}")
                
            # Give Kafka some time to process the deletion request
            time.sleep(2)
                
        except Exception as e:
            logger.error(f"Error attempting to delete topic '{topic}': {e}")

def create_topics(bootstrap_servers='localhost:9092'):
    """
    Create necessary Kafka topics
    """
    # First delete existing topics
    delete_topics(bootstrap_servers)
    
    try:
        # Create admin client
        admin_client = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id='kafka-admin'
        )
        
        # Define topics
        topics = [
            NewTopic(name="transaction_data_topic", num_partitions=3, replication_factor=1),
            NewTopic(name="fraud_alerts_topic", num_partitions=3, replication_factor=1)
        ]
        
        # Check existing topics to avoid recreation error
        existing_topics = admin_client.list_topics()
        topics_to_create = [topic for topic in topics if topic.name not in existing_topics]
        
        if topics_to_create:
            # Create topics
            admin_client.create_topics(topics_to_create)
            logger.info(f"Successfully created topics: {[t.name for t in topics_to_create]}")
        else:
            logger.info("All topics already exist")
            
    except Exception as e:
        logger.error(f"Failed to create topics: {e}")
    finally:
        if 'admin_client' in locals():
            admin_client.close()

if __name__ == "__main__":
    create_topics()
    print("Kafka topics setup completed.")