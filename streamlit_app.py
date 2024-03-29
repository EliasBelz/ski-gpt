import openai
import streamlit as st
import pinecone
from streamlit_chat import message
import random
import os
import json
from dotenv import load_dotenv

DEBUG = False

#=====================================================#
#                      API SETUP                      #
#=====================================================#

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Replace with your api key
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = "us-west4-gcp-free"
MAX_CONTEXT = 5 # conversational memory window. First index is system call
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
dimension = 1536 # opeanAI default

index = pinecone.Index(index_name)

openai.api_key = OPENAI_API_KEY

# Product Search using Pinecone vector db
def prod_search(query):
    # Embed question
    xq = openai.Embedding.create(input=query, engine=model_name)['data'][0]['embedding']
    res = index.query([xq], top_k=3, include_metadata=True)
    products = ""
    for match in res['matches']:
        products+=(f"{match['metadata']['text']}")
    return products


#=====================================================#
#                     Chat Code                       #
#=====================================================#

# Storing the chat
# Intro chat generated, but hardcoded for load time
if 'generated' not in st.session_state:
    st.session_state['generated'] = [
        "Hey there! I'm an AI working as a ski shop employee at evo Seattle. I'm here to help you find the perfect skis or snowboard for your next adventure on the slopes!\
        Just let me know what you're looking for, and I'll do my best to recommend some awesome gear for you. 🏂⛷️"
        ]

if 'past' not in st.session_state:
    st.session_state['past'] = ["What's up?! What do you do?"]

if "avatars" not in st.session_state:
    st.session_state.avatars = {"user": random.randint(0,100), "bot": random.randint(0,100)}

# Chat history for openai
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {
            "role": "system",
            "content":
                "You are a helpful ai acting as a ski shop employee with the personality of a cool snowboarder at evo Seattle!\
                Your job is to recommend skis and snowboards! You can use  prod_search function to search a product database and use it to help answer user questions.\
                Always try to make good recommendations and only recommend gear from prod_search function.\
                If you don't have enough detail to search for the right product, act like a ski shop employee and ask for more info from the user.\
                Add personality with emojis and always include product links next to recommendations."
        }
    )

functions = [
        {
            "name": "prod_search",
            "description": "Use this function to get background information on skis and snowboards from evo. Use this function whenever talking about skis or snowboards/",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Nautral language semantic query to search skis and snowboards. Be as thorough as possible.",
                    },
                },
                "required": ["query"],
            },
        }
    ]
func_responses = []
def chat(user_input=""):
    if user_input == "":
        user_input = st.session_state.input
    st.session_state.input = ""
    st.session_state.messages.append({"role": "user", "content": user_input})

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    functions=functions,
    messages=st.session_state.messages,
    temperature=0.2
    )

    output = completion.choices[0].message
    if DEBUG : print(output)
    query = ""
    # check if GPT wanted to call a function
    if output.get("function_call"):
        # call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "prod_search": prod_search,
        }  # only one function in this example, but you can have multiple
        function_name = output["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(output["function_call"]["arguments"])
        query = function_args.get("query")
        function_response = fuction_to_call(
            query=query,
        )

        # send the info from the function call and function response to GPT
        st.session_state.messages.append(output)  # extend conversation with assistant's reply
        res = {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        del st.session_state.messages[-1] # delete the return of the function from chat history to conserve tokens
        st.session_state.messages.append(res)  # extend conversation with function response
        second_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=st.session_state.messages,
            temperature=0.8
        )
        output = second_completion.choices[0].message
        if DEBUG : print(output)
    # remove chats from context window
    if len(st.session_state.messages) > MAX_CONTEXT - 1:
        # keeps system call at index 0
        del st.session_state.messages[1]
    # add to GPT state
    st.session_state.messages.append(output)
    # front end bot state
    st.session_state.generated.append(output.content)
    # product search message
    if query != "":
            st.session_state.generated.append(f'Searching for: {query}')
            st.session_state.past.append("")
    # front end user state
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
    st.markdown("Unlike chatGPT, Ski-GPT will answer using injected knowlege from evo.com.")
    st.markdown("---")
    st.markdown("A side project by Elias Belzberg")
    st.markdown("ebelz@cs.washington.edu")
    st.markdown("ebelzberg@evo.com")
    st.markdown("Code available here!\n"
                "[github.com/EliasBelz/ski-gpt](https://github.com/EliasBelz/ski-gpt)")
    st.markdown("---")
    st.markdown("Tech this project uses:\n"
                "- OpenAI gpt3.5 turbo LLM\n"
                "- OpenAI function calling\n"
                "- Pinecone vector database\n"
                "- Streamlit")
    st.markdown("---")

# We will get the user's input by calling the chat function
input_text = st.text_input("Input a question here! For example: \"What are the best Skis for powder?\", \
                                \"Compare the Season Nexus and Forma.\"",
                                placeholder="Enter prompt: ", key="input", on_change=chat)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        if st.session_state['past'][i] != "":
            message(st.session_state['past'][i], is_user=True, avatar_style="adventurer",seed=st.session_state.avatars["bot"], key=str(i) + '_user')
        message(st.session_state["generated"][i],seed=st.session_state.avatars["user"] , key=str(i))