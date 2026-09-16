import requests
import os

# Dataset folder create karega
os.makedirs("dataset", exist_ok=True)

# Categories
categories = ["star", "heart", "moon", "apple","ambulance","basket","banana","bat","bee","book"]

# Quick Draw dataset ka base URL
base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

for category in categories:

    print(f"Downloading {category}...")

    url = base_url + category + ".npy"

    response = requests.get(url)

    if response.status_code == 200:

        file_path = f"dataset/{category}.npy"

        with open(file_path, "wb") as file:
            file.write(response.content)

        print(f"{category}.npy downloaded successfully!")

    else:
        print(f"Could not download {category}")
        print("Status code:", response.status_code)

print("\nDataset download completed!")