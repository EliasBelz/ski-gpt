import openai
import streamlit as st
from streamlit_chat import message
import pinecone
import random
import os
from dotenv import load_dotenv


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate

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
    environment=PINECONE_API_ENV # next to api key in console
)

openai.api_key=OPENAI_API_KEY

embed=OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)
# random user picture
user_av = random.randint(0, 100)

# random bott picture
bott_av = random.randint(0, 100)

def randomize_array(arr):
    sampled_arr = []
    while arr:
        elem = random.choice(arr)
        sampled_arr.append(elem)
        arr.remove(elem)
    return sampled_arr

st.set_page_config(page_title="Ski-GPT", page_icon="🎿", layout="wide")

st.header("Ski-GPT is like chatGPT for personalized gear recomendations!\n")

# Define the name of the index and the dimensionality of the embeddings
index_name = "shred-data"
dimension = 1536

index = pinecone.Index(index_name)


text_field = 'text'

vectorstore = Pinecone (
    index, embed.embed_query, text_field
)

retriever = vectorstore.as_retriever()

# chat completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.1
)

# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True
)

prompt_template = """You are a friendly chatbot with the personality of a rad ski shop employee that works at evo Seattle!
                     Use the following pieces of product context to answer the question at the end.
                     If you don't know the answer, give a genral overview or ask for more details, don't try to make up an answer.
                     Always include the url of any product meantioned.

                    {context}

                    Question: {question}
                    Answer like a ski shop employee and include the URL link of any products:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs=chain_type_kwargs,
)

tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description=(
            'use this tool when answering ski or snowboard knowledge queries to get '
            'more information about the topic'
        )
    )
]

# Example of what is given to the llm
def get_tools(query):
    for doc in retriever.get_relevant_documents(query):
        print(doc.page_content)
        print(f"Meta Data:{doc.metadata}")
# Lags page, so commented.
# get_tools("Powder Skis")

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=2,
    early_stopping_method='generate',
    memory=conversational_memory
)

with st.sidebar:
    st.markdown("# About 🙌")
    st.markdown("Ski-GPT is your personal shred curator! 🤟")
    st.markdown("With knowledge of 400 skis and snowboards from evo.com, Ski-GPT will find the right gear for you!")
    st.markdown("Unlike chatGPT, Ski-GPT will only answer using injected knowlege from evo.com")
    st.markdown("---")
    st.markdown("A side project by Elias Belzberg")
    st.markdown("ebelz@cs.washington.edu")
    st.markdown("ebelzberg@evo.com")
    st.markdown("---")
    st.markdown("This tech project uses:\n"
                "- OpenAI gpt3.5 turbo LLM\n"
                "- Pinecone vector databse\n"
                "- Langchain python library\n"
                "- Streamlit")
    st.markdown("---")

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def clear_text():
    st.session_state["input"] = ""

# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("Input a question here! For example: \"What are the best Skis for powder?\".", placeholder="Enter prompt: ", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = agent.run(user_input)
    # store the output
    st.session_state.generated.append(output)
    st.session_state.past.append(user_input)


if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True,avatar_style="adventurer",seed=user_av, key=str(i) + '_user')
        message(st.session_state["generated"][i],seed=bott_av , key=str(i))