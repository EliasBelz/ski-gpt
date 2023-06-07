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
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

docsearch = Pinecone.from_existing_index("shred-data", embedding=embeddings)
query = "What are the best ski for doing tricks?"
docs = docsearch.similarity_search(query)
for doc in docs:
  print("\n"+doc.page_content)