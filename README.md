# ski-gpt
Ski-GPT is your personal shred curator! 🤟
With knowledge of 400 skis and snowboards from evo.com, Ski-GPT will find the right gear for you!
Unlike chatGPT, Ski-GPT will only answer using injected knowledge from evo.com

### A side project by Elias Belzberg

ebelz@cs.washington.edu

ebelzberg@evo.com

### Tech this project uses:

- OpenAI gpt3.5 turbo LLM
- Pinecone vector database
- Langchain Python library
- Streamlit

## How it works
Ski-GPT uses pinecone vector db to inject product knowledge into openai's 3.5 turbo model.

Example of injected knowldge for "Powder skis"
![Ski knowledge](/img/context.png)

## App Files:

### Take1
/!\ No longer works. DB expired /!\\

Uses weaviate vector database. Their servers are in Europe which made everything super slow, so I pivoted. Weaviate is an open-source project and can be run locally, just not on my laptop.

### Take2
Uses Pinecone vector database, which is not opensource but has servers in America. This version also uses langchain and conversational chat agents, which in the long term could be better but the responses weren't as good.

Example of agent logic for "best Libtech snowboard?":
![Ski knowledge](/img/agent.png)

### Take3
Uses Pinecone and langchain but no chat agent. This is a simpler more limited approach, but on average has the best results.

### Take4
/!\ Currently hosted on ski-gpt.streamlit.app /!\\

Uses Pinecone and OpenAI function calling for tool use. Also added conversational memory.
## Tools:
Tools for embedding data and uploading to vector databases.

### jsonToCsv.py
Takes JSON formatted data from `evo-scraper` and converts it to a csv file to get ready for embedding.

### pinecone-csv-upload.py
Embeds csv file data and uploads it to Pinecone

### weaviate-csv-upload.py
Embeds csv file data and uploads it to Weaviate

## How to run locally
- Create `.env` file and enter OPENAI_API_KEY and PINECONE_API_KEY
- Create Pinecone database and run tools to load data.
- In command line `streamlit run <file.py>`