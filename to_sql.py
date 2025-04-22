import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Define database connection parameters for Ubuntu with peer authentication
# When running as the postgres user with sudo privileges
DB_NAME = 'fraud_detection'

# Define the SQLAlchemy model
Base = declarative_base()

class Transaction(Base):
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    step = Column(Integer)
    type = Column(String)
    amount = Column(Float)
    nameOrig = Column(String)
    oldbalanceOrg = Column(Float)
    newbalanceOrig = Column(Float)
    nameDest = Column(String)
    oldbalanceDest = Column(Float)
    newbalanceDest = Column(Float)
    isFraud = Column(Integer)
    isFlaggedFraud = Column(Integer)

def main():
    # Create a database connection using peer authentication
    # This assumes your script runs with the same user that has PostgreSQL access
    connection_string = f"postgresql:///{DB_NAME}"
    
    # If you need to run the script as the postgres user:
    # First set up the database with: sudo -u postgres createdb your_database
    # Then run this script with: sudo -u postgres python3 your_script.py
    
    engine = create_engine(connection_string)
    
    # Drop the existing transactions table if it exists
    with engine.connect() as connection:
        connection.execute(text("DROP TABLE IF EXISTS transactions"))
        print("Dropped existing transactions table (if it existed)")
    
    # Create the table
    Base.metadata.create_all(engine)
    print("Created new transactions table")
    
    # Read CSV file
    csv_file_path = 'your_data.csv'
    df = pd.read_csv(csv_file_path)
    
    # Write to PostgreSQL
    try:
        # For better performance with large datasets, use to_sql with method='multi'
        df.to_sql(
            'transactions', 
            engine, 
            if_exists='append',  # Using 'append' since we already dropped the table
            index=False,
            chunksize=1000  # Adjust based on your data size
        )
        print(f"Successfully imported {len(df)} records to the database.")
    except Exception as e:
        print(f"Error importing data: {e}")

if __name__ == "__main__":
    main()