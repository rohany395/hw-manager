import streamlit as st
from openai import OpenAI
from pathlib import Path
from PyPDF2 import PdfReader
import sys 
import pysqlite3
sys.modules['sqlite3']=pysqlite3
import chromadb
from bs4 import BeautifulSoup

# Show title and description.
st.title("📄 Rohan's Chatbot with integrated vector DB")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="lab4Collection")

if 'openai_client' not in st.session_state:
    openai_api_key = st.secrets["API_KEY"]
    st.session_state.openai_client = OpenAI(api_key=openai_api_key)

def add_to_collection(collection, text, filename):
    openai_client = st.session_state.openai_client
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding

    collection.add(
        documents=[text],
        ids=[filename],
        embeddings=[embedding]
    )

#building vector DB after chunking
if collection.count() == 0:
    html_dir = Path("htmlFiles")

    for html_file in html_dir.glob("*.html"):
        with open(html_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            
            # Split the file into two equal parts for better embedding retrieval.
            half = len(text) // 2
            chunks = [text[:half], text[half:]]

            for idx, chunk in enumerate(chunks):
                add_to_collection(collection, chunk, f"{html_file.stem}_part{idx+1}")

if 'messages' not in st.session_state:
    st.session_state['messages']=[{'role':'assistant','content':'Hi how can I help?'}]



#chat input
prompt=st.chat_input('Talk to me Goose')
if prompt:
    # Append user message
    st.session_state.messages.append({'role':'user','content':prompt})
    st.session_state.messages = st.session_state.messages[-10:]

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.conversation
    )

for msg in st.session_state.messages:
    chat_msg=st.chat_message(msg['role'])
    chat_msg.write(msg['content'])

openai_client = st.session_state.openai_client
response=openai_client.embeddings.create(
    input=prompt,
    model="text-embedding-3-small"
)

query_embedding = response.data[0].embedding

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

for i in range(len(results['documents'][0])):
    doc=results['documents'][0][i]
    doc_id=results['ids'][0][i]
    st.write(f"The following file might be helpfu;l: {doc_id}")

    
