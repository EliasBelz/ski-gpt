import openai
import streamlit as st
import pinecone
from streamlit_chat import message
import random
import os
from dotenv import load_dotenv

#=====================================================#
#                      API SETUP                      #
#=====================================================#

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Replace with your api key
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = "us-west4-gcp-free"
model_name = 'text-embedding-ada-002'

if(not (OPENAI_API_KEY and PINECONE_API_ENV)):
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)

# Define the name of the index and the dimensionality of the embeddings
index_name = "shred-data"
dimension = 1536

index = pinecone.Index(index_name)

openai.api_key = OPENAI_API_KEY

def prodSearch(query):
    # Embed question
    xq = openai.Embedding.create(input=query, engine=model_name)['data'][0]['embedding']
    res = index.query([xq], top_k=5, include_metadata=True)
    products = ""
    for match in res['matches']:
        products+=(f"Match: {match['score']:.2f}: {match['metadata']['text']}")
    return products


#=====================================================#
#                     Chat Code                       #
#=====================================================#

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if "avatars" not in st.session_state:
    st.session_state.avatars = {"user": random.randint(0,100), "bot": random.randint(0,100)}

# Chat history for openai
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": "You are a friendly chatbot with the personality of a rad ski shop employee that works at evo Seattle!"})

def chat():
    user_input = st.session_state.input
    print(prodSearch(user_input))
    st.session_state.input = ""
    st.session_state.messages.append({"role": "user", "content": user_input})
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=st.session_state.messages
    )
    output = completion.choices[0].message
    st.session_state.messages.append(output)
    st.session_state.generated.append(output.content)
    st.session_state.past.append(user_input)


#=====================================================#
#               Font-end, yup thats it!               #
#=====================================================#

st.set_page_config(page_title="Ski-GPT", page_icon="🎿", layout="wide", initial_sidebar_state="expanded")

st.header("🎿Ski-GPT is like chatGPT for personalized ski and snowboard recommendations!\n")

with st.sidebar:
    st.markdown("# About 🙌")
    st.markdown("Ski-GPT is your personal shred curator! 🤟")
    st.markdown("With knowledge of 400 skis and snowboards from evo.com, Ski-GPT will find the right gear for you!")
    st.markdown("Unlike chatGPT, Ski-GPT will only answer using injected knowlege from evo.com.")
    st.markdown("---")
    st.markdown("A side project by Elias Belzberg")
    st.markdown("ebelz@cs.washington.edu")
    st.markdown("ebelzberg@evo.com")
    st.markdown("Code available here!\n"
                "[github.com/EliasBelz/ski-gpt](https://github.com/EliasBelz/ski-gpt)")
    st.markdown("---")
    st.markdown("Tech this project uses:\n"
                "- OpenAI gpt3.5 turbo LLM\n"
                "- Pinecone vector database\n"
                "- Langchain python library\n"
                "- Streamlit")
    st.markdown("---")

# We will get the user's input by calling the get_text function
input_text = st.text_input("Input a question here! For example: \"What are the best Skis for powder?\", \
                                \"Compare the Season Nexus and Forma snowboards.\"",
                                placeholder="Enter prompt: ", key="input", on_change=chat)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, avatar_style="adventurer",seed=st.session_state.avatars["bot"], key=str(i) + '_user')
        message(st.session_state["generated"][i],seed=st.session_state.avatars["user"] , key=str(i))
