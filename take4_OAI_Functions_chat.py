import openai
import streamlit as st
import pinecone
from streamlit_chat import message
import random
import os
import json
from dotenv import load_dotenv

# TODO: COntext window

DEBUG = True

#=====================================================#
#                      API SETUP                      #
#=====================================================#

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Replace with your api key
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = "us-west4-gcp-free"
MAX_CONTEXT = 5;
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
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if "avatars" not in st.session_state:
    st.session_state.avatars = {"user": random.randint(0,100), "bot": random.randint(0,100)}

# Chat history for openai
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {
            "role": "system",
            "content":
                "You are a helpful ai acting as a ski shop employee with the personality of a rad snowboarder that works at evo Seattle!\
                Your job is to help recommend skis and snowboards! You have access to a product database and can use it asnwer user questions,\
                Always try to include a recommended product in the reponse.\
                If you don't know what to recommend, give a genral overview or ask for more details, don't try to make up an answer.\
                Always include the url of any product meantioned."
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
                        "description": "Nautral language query to search skis and snowboards. You should rephrase the question to get unique results.",
                    },
                },
                "required": ["query"],
            },
        }
    ]
func_responses = []
def chat():
    user_input = st.session_state.input
    st.session_state.input = ""
    st.session_state.messages.append({"role": "user", "content": user_input})

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    functions=functions,
    messages=st.session_state.messages,
    temperature=0.8
    )

    output = completion.choices[0].message
    if DEBUG : print(output)
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
        function_response = fuction_to_call(
            query=function_args.get("query"),
        )

        # send the info on the function call and function response to GPT
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
            temperature=0.2
        )
        output = second_completion.choices[0].message
        if DEBUG : print(output)

    if len(st.session_state.messages) > MAX_CONTEXT - 1:
        del st.session_state.messages[1]
    st.session_state.messages.append(output)
    st.session_state.generated.append(output.content)
    st.session_state.past.append(user_input)
    if output.get("function_call"):
        st.session_state.generated.append(f'Searching for: {output["function_call"]["arguments"]}')
        st.session_state.past.append("")


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
                "- OpenAI function calling\n"
                "- Pinecone vector database\n"
                "- Streamlit")
    st.markdown("---")

# We will get the user's input by calling the get_text function
input_text = st.text_input("Input a question here! For example: \"What are the best Skis for powder?\", \
                                \"Compare the Season Nexus and Forma snowboards.\"",
                                placeholder="Enter prompt: ", key="input", on_change=chat)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        if st.session_state['past'][i] != "":
            message(st.session_state['past'][i], is_user=True, avatar_style="adventurer",seed=st.session_state.avatars["bot"], key=str(i) + '_user')
        message(st.session_state["generated"][i],seed=st.session_state.avatars["user"] , key=str(i))
