import pandas as pd
import sqlite3
import re
import os
import kagglehub


class HouseRentDatabase:
    def __init__(self, db_name="house_rent.db"):
        self.db_path = os.path.join(os.getcwd(), db_name)

    def download_dataset(self):
        # Kagglehub handles the download and caching
        path = kagglehub.dataset_download(
            "iamsouravbanerjee/house-rent-prediction-dataset"
        )
        # The file is typically in a subdirectory named after the dataset
        csv_file = os.path.join(path, "House_Rent_Dataset.csv")
        return csv_file

    def ingest_data(self):
        csv_path = self.download_dataset()
        df = pd.read_csv(csv_path)

        # Clean Size column
        df["Size"] = df["Size"].astype(str).apply(lambda x: re.sub(r"[^0-9]", "", x))
        df["Size"] = pd.to_numeric(df["Size"])

        # Save to SQLite
        conn = sqlite3.connect(self.db_path)
        df.to_sql("house_rent", conn, if_exists="replace", index=False)
        conn.close()
        print(f"Data ingested and saved to {self.db_path}")

    def get_data(self):
        if not os.path.exists(self.db_path):
            self.ingest_data()

        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql("SELECT * FROM house_rent", conn)
        except pd.errors.DatabaseError:
            self.ingest_data()
            df = pd.read_sql("SELECT * FROM house_rent", conn)
        conn.close()
        return df
