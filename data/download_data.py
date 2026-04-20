import os
import requests
import zipfile
import pandas as pd

def download_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    data_dir = "data"
    zip_path = os.path.join(data_dir, "smsspamcollection.zip")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    print("Downloading dataset...")
    response = requests.get(url)
    with open(zip_path, "wb") as f:
        f.write(response.content)
        
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)
        
    # Read and convert to CSV for easier use
    df = pd.read_csv(os.path.join(data_dir, "SMSSpamCollection"), sep='\t', names=['label', 'text'])
    df.to_csv(os.path.join(data_dir, "spam_dataset.csv"), index=False)
    print(f"Dataset saved to {os.path.join(data_dir, 'spam_dataset.csv')}")

if __name__ == "__main__":
    download_data()
