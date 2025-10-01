import streamlit as st
from openai import OpenAI
from pathlib import Path
from PyPDF2 import PdfReader
import sys 
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import chromadb

# Show title and description
st.title("📄 Rohan's Chatbot")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="courseInfoCollection")

# Initialize OpenAI client
if 'openai_client' not in st.session_state:
    openai_api_key = st.secrets["API_KEY"]
    st.session_state.openai_client = OpenAI(api_key=openai_api_key)

# Initialize messages in session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'Hi! How can I help you with course information?'}]

# Function to add documents to collection
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

if 'pdfs_loaded' not in st.session_state:
    pdf_dir = Path("pdfFiles")
    if pdf_dir.exists():
        for pdf_file in pdf_dir.glob("*.pdf"):
            try:
                reader = PdfReader(str(pdf_file))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                add_to_collection(collection, text, pdf_file.stem)
            except Exception as e:
                st.error(f"Error loading {pdf_file.name}: {e}")
        st.session_state['pdfs_loaded'] = True

# Function to get relevant course info from vector search
def get_relevant_course_info(query):
    openai_client = st.session_state.openai_client
    
    # Get embedding for the query
    response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    # Search the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # Get top 3 most relevant documents
    )
    
    # Combine the relevant documents into a single string
    relevant_info = ""
    if results['documents'] and len(results['documents'][0]) > 0:
        for i, doc in enumerate(results['documents'][0]):
            doc_id = results['ids'][0][i]
            relevant_info += f"\n\n--- From document: {doc_id} ---\n{doc[:1000]}"  # Limit length
    else:
        relevant_info = "No relevant course information found."
    
    return relevant_info

# Display chat messages
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg['role'])
    chat_msg.write(msg['content'])

# Chat input
if prompt := st.chat_input('Ask me about the courses'):
    # Maintain short-term memory (last 12 messages = 6 exchanges)
    st.session_state.messages = st.session_state.messages[-12:]
    
    # Append user message
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    # Display user message
    with st.chat_message('user'):
        st.write(prompt)
    
    # Get relevant course info from vector search
    with st.spinner('Searching course documents...'):
        relevant_course_info = get_relevant_course_info(prompt)
    
    # Create system message with the relevant course info
    system_message = {
        'role': 'system',
        'content': f"""You are a helpful course assistant. Use the following relevant course information to answer the user's question. 
        If the information doesn't contain the answer, say so politely.
        
        Relevant Course Information:
        {relevant_course_info}
        
        Please provide a clear, concise answer based on this information."""
    }
    
    api_messages = [system_message] + st.session_state.messages
    
    # Generate response using OpenAI API with vector search results
    with st.chat_message('assistant'):
        openai_client = st.session_state.openai_client
        
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=api_messages,
            stream=True,
        )
        
        response = st.write_stream(stream)
    
    # Append assistant response to messages
    st.session_state.messages.append({'role': 'assistant', 'content': response})