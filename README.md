## ski-gpt
Ski-GPT is your personal shred curator! 🤟
With knowledge of 400 skis and snowboards from evo.com, Ski-GPT will find the right gear for you!
Unlike chatGPT, Ski-GPT will only answer using injected knowlege from evo.com

# A side project by Elias Belzberg

ebelz@cs.washington.edu

ebelzberg@evo.com

# This tech project uses:

- OpenAI gpt3.5 turbo LLM
- Pinecone vector database
- Langchain python library
- Streamlit

## App Files

# Take1
Uses weaviate vector database. Their servers are in europe which made everything super slow, so I pivoted. Weaviate is an opensource project and can be ran locally, just not on my laptop.

# Take2
Uses Pinecone vector databse, which in not opensource but has servers in America. This version also uses langchain and conversational chat agents, which in the longeterm could be better but the responses werent as good.

# Take3
Uses Pinecone and langchain but no chat agent. This is a simpler more limited approach, but on average had the best results.

## Tools
Tools for embedding data and uploading to vector databases.

# jsonToCsv.py
Takes JSON formatted data from evo-scraper and converts it to a csv to preparer it for embedding.

# pinecone-csv-upload.py
Embeds csv file data and uploads it to pinecone
