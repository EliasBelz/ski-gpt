from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

OPENAI_API_KEY="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"
PINECONE_API_KEY ="048203ad-67d2-41d6-97c9-a1a083c8fce9"
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



