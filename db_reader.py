import sqlite3
import csv
import os

def convert_sqlite_to_csv():
    # Set the current directory and paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'database.db')
    
    # Create 'data_csv' folder if it doesn't exist
    csv_folder = os.path.join(current_dir, 'data_csv')
    os.makedirs(csv_folder, exist_ok=True)
    
    csv_path = os.path.join(csv_folder, 'output.csv')

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Execute a SELECT query to fetch all data from the table
        cursor.execute("SELECT * FROM daily_reports")

        # Fetch all rows
        rows = cursor.fetchall()

        # Get column names
        column_names = [description[0] for description in cursor.description]

        # Write data to CSV file
        with open(csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Write header
            csv_writer.writerow(column_names)
            
            # Write data rows
            csv_writer.writerows(rows)

        print(f"Data successfully exported to {csv_path}")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    convert_sqlite_to_csv()