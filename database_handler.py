import psycopg2
from datetime import datetime, timedelta
import redis
import time

ABNORMALITY_THRESHOLD = timedelta(seconds=1)
MACHINE_ID = 1

db_settings = {
    'dbname': 'audio_analysis',
    'user': 'infy',
    'password': 'infy123',
    'host': 'localhost',
    'port': '5432',
}

def log_abnormality(timestamp_start, timestamp_end, machine_id):
    # Calculate the duration
    duration = (timestamp_end - timestamp_start).total_seconds()
    try:
        # Connect to PostgreSQL
        connection = psycopg2.connect(**db_settings)
        cursor = connection.cursor()

        # Insert data
        insert_query = """
            INSERT INTO machine_abnormalities (timestamp_start, timestamp_end, duration_seconds, machine_id)
            VALUES (%s, %s, %s, %s);
        """
        cursor.execute(insert_query, (
            timestamp_start, timestamp_end, duration, machine_id
        ))

        # Commit the transaction and close connection
        connection.commit()
        cursor.close()
        connection.close()

        print("Abnormality logged successfully.")

    except Exception as e:
        print("Error logging abnormality:", e)
        if connection:
            connection.rollback()

def consec_abnorm(label, current_abnormality_start, current_abnormality_end, timestamp_mel, abnormality_active):
    if label == 'Abnormal':
        if not abnormality_active:
            # Start of a new abnormality
            current_abnormality_start = timestamp_mel
            abnormality_active = True
        current_abnormality_end = timestamp_mel
    else:
        if abnormality_active:
            # End of an abnormality, log it
            log_abnormality(current_abnormality_start, current_abnormality_end, MACHINE_ID)
            abnormality_active = False
            current_abnormality_start = None
            current_abnormality_end = None

    return current_abnormality_start, current_abnormality_end, abnormality_active

def main():
    r = redis.Redis(host='localhost', port=6379, db=0)
    current_abnormality_start = None
    current_abnormality_end = None
    abnormality_active = False  # Tracks if an abnormality is ongoing

    while True:
        keys_mel = sorted(r.keys('mel_spec:*'))
        if keys_mel:
            latest_key_mel = keys_mel[-1]
            timestamp_mel = latest_key_mel.decode('utf-8').split(':')[1]
            prediction_score = r.get(f'prediction:{timestamp_mel}')

            # Check if prediction_score is None before proceeding
            if prediction_score is not None:
                prediction_score = float(prediction_score)
                label = 'Abnormal' if prediction_score > 0.5 else 'Normal'
                print(f"Prediction score found for timestamp: {timestamp_mel} and label is: {label}")

                # Update abnormality tracking
                current_abnormality_start, current_abnormality_end, abnormality_active = consec_abnorm(
                    label, current_abnormality_start, current_abnormality_end, datetime.fromtimestamp(float(timestamp_mel)), abnormality_active
                )
            else:
                print(f"No prediction score found for timestamp: {timestamp_mel}")

        time.sleep(1)

if __name__ == '__main__':
    main()
