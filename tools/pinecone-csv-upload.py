from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Replace with your api key
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV ="us-west4-gcp-free"

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)

from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='output.csv')

data = loader.load()
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

index_name="shred-data"
Pinecone.from_documents(data , embeddings, index_name=index_name)
print(data)