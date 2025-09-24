import streamlit as st
from openai import OpenAI
from pathlib import Path
from PyPDF2 import PdfReader
import sys 
import pysqlite3
sys.modules['sqlite3']=pysqlite3
import chromadb
from bs4 import BeautifulSoup
from anthropic import Anthropic
import google.generativeai as genai

# Show title and description.
st.title("📄 Rohan's Chatbot with integrated vector DB")
llm_choice=st.sidebar.selectbox("Select a service",("Gemini","Chatgpt","Claude"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="lab4Collection")

if 'openai_client' not in st.session_state:
    openai_api_key = st.secrets["API_KEY"]
    st.session_state.openai_client = OpenAI(api_key=openai_api_key)

if 'gemini_client' not in st.session_state and "GEMINI_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
    st.session_state.gemini_model = genai.GenerativeModel('gemini-2.5-flash')

if 'claude_client' not in st.session_state and "CLAUDE_KEY" in st.secrets:
    st.session_state.claude_client = Anthropic(api_key=st.secrets["CLAUDE_KEY"])

openai_client = st.session_state.openai_client

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

    for msg in st.session_state.messages:
        chat_msg=st.chat_message(msg['role'])
        chat_msg.write(msg['content'])

    openai_client = st.session_state.openai_client
    query_embedding=openai_client.embeddings.create(
        input=prompt,
        model="text-embedding-3-small"
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_context = "\n\n".join(results['documents'][0])

    # 3️⃣ Build prompt for OpenAI using user question + retrieved context
    augmented_prompt = (
        f"Use the following context to answer the question accurately.\n\n"
        f"Context:\n{retrieved_context}\n\n"
        f"Question: {prompt}"
    )

    if llm_choice == "Chatgpt":
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages + [
                {"role": "system", "content": augmented_prompt}
            ]
        )
        bot_reply = response.choices[0].message.content

    elif llm_choice == "Gemini":
        gemini_model = st.session_state.gemini_model
        response = gemini_model.generate_content(
            "\n".join([m["content"] for m in st.session_state.messages]) + "\n" + augmented_prompt
        )
        bot_reply = response.text

    elif llm_choice == "Claude":
        claude_client = st.session_state.claude_client
        response = claude_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=500,
            messages=[{"role": "user", "content": augmented_prompt}]
        )
        bot_reply = response.content[0].text
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

    
