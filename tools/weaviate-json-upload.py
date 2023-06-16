import weaviate
import os
from dotenv import load_dotenv

load_dotenv()

client = weaviate.Client(
    url="https://ski-cluster-9u6a794j.weaviate.network",
    additional_headers = {
        "X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY')  # Replace with your inference API key
    }  # Replace with your endpoint
)

# ===== add schema =====
class_obj = {
    "class": "ShredData",
      "description": "Ski product data",
      "vectorizer": "text2vec-openai",
      "moduleConfig": {
        "generative-openai": {

        }
      },
    "properties": [
        {
            "dataType": ["text"],
            "name": "productType",
        },
        {
            "dataType": ["text"],
            "name": "productName",
        },
        {
            "dataType": ["text"],
            "name": "price",
        },
        {
            "dataType": ["text"],
            "name": "url",
        },
        {
            "dataType": ["text"],
            "name": "description",
        },
        {
            "dataType": ["text"],
            "name": "spec",
        },
    ],
    "vectorizer": "text2vec-openai"  # This could be any vectorizer
}

client.schema.create_class(class_obj)

# ===== import data =====
# Load data
import csv

filename = "skiData200.csv"

rows = []

with open(filename, "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        rows.append(row)



# Configure a batch process
with client.batch as batch:
    batch.batch_size=100
    # Batch import all Questions
    for i, r in enumerate(rows):
        print(f"importing ski: {i+1}")

        properties = {
            "productType": r[0],
            "productName": r[1],
            "price": r[2],
            "url": r[3],
            "description": r[4],
            "spec": r[5],
        }

        client.batch.add_data_object(properties, "ShredData")

get = client.schema.get()
print(get)