__import__('pysqlite3')
import sys 
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
import json
import bs4
import requests

st.title("📄 Rohan's News bot")
model_choice=st.sidebar.selectbox("Select a service",("Gemini","Chatgpt"))

df = pd.read_csv('./excelFile/Example_news_info_for_testing.csv')

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

#use openai embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=st.secrets["API_KEY"],
    model_name="text-embedding-3-small"
)

#creates collection for embeddings
collection = chroma_client.get_or_create_collection(
    name="newsCollection",
    embedding_function=openai_ef)

if collection.count() == 0:
    documents = []
    metadatas = []
    ids = []
    for idx, row in df.iterrows():
        text = f"{row['Document']}. 'url:'{row['URL']}"
        
        documents.append(text)
        ids.append(f"article_{idx}")
        metadatas.append({
            "title": row['Document'],
            "date": str(row['Date']),
            "URL": row['URL'],
            "company": row['company_name'],
        })

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )


# Initialize OpenAI client
if 'openai_client' not in st.session_state:
    openai_api_key = st.secrets["API_KEY"]
    st.session_state.openai_client = OpenAI(api_key=openai_api_key)

if 'gemini_configured' not in st.session_state:
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
    st.session_state.gemini_configured = True

    # Initialize messages in session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'Hi! I am your news assistant. How can I help you?'}]

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = bs4.BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None
    
def rank_interesting_news(n_results=10):
    """Get most interesting news for law firms"""
    legal_keywords = [
        "litigation", "regulatory", "compliance", "merger", "acquisition",
        "SEC", "settlement", "lawsuit", "investigation", "enforcement",
        "antitrust", "securities", "fraud"
    ]
    
    all_results = []
    
    for keyword in legal_keywords:
        results = collection.query(
            query_texts=[keyword],
            n_results=3
        )
        
        for i in range(len(results['ids'][0])):
            all_results.append({
                "id": results['ids'][0][i],
                "title": results['metadatas'][0][i]['title'],
                "url": results['metadatas'][0][i]['URL'],
                "company": results['metadatas'][0][i]['company'],
                "date": results['metadatas'][0][i]['date'],
                "relevance_score": round(1 - results['distances'][0][i], 3),
                "matched_keyword": keyword
            })
    
    # Remove duplicates, sort by relevance
    seen_ids = set()
    unique_results = []
    for result in sorted(all_results, key=lambda x: x['relevance_score'], reverse=True):
        if result['id'] not in seen_ids:
            seen_ids.add(result['id'])
            unique_results.append(result)
    
    return unique_results[:n_results]

def search_news(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    articles = []
    for i in range(len(results['ids'][0])):
        newsText=read_url_content(results['metadatas'][0][i]['URL'])
        articles.append({
            "title": results['metadatas'][0][i]['title'],
            "content": newsText,
            "URL": results['metadatas'][0][i]['URL'],
            "company": results['metadatas'][0][i]['company'],
            "date": results['metadatas'][0][i]['date'],
            "relevance_score": 1 - results['distances'][0][i]
        })
    
    return articles

def get_context(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    context = "Here are relevant news articles from our database:\n\n"
    for i in range(len(results['ids'][0])):
        context += f"{i+1}. **{results['metadatas'][0][i]['title']}**\n"
        context += f"   Company: {results['metadatas'][0][i]['company']}\n"
        context += f"   Date: {results['metadatas'][0][i]['date']}\n"
        context += f"   URL: {results['metadatas'][0][i]['URL']}\n\n"
    
    return context

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search for news articles about a specific topic or company. Use this when user asks about specific case about a news",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or topic"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rank_interesting_news",
            "description": "Get the most legally interesting/important news. Use when user asks for 'interesting', 'important', or 'top' news in general.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_results": {
                        "type": "integer",
                        "description": "Number of top items (default 10)",
                        "default": 10
                    }
                },
                "required": []
            }
        }
    },
]

gemini_tools = [
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="search_news",
                description="Search for news articles about a specific topic or company.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results (default 5)"
                        }
                    },
                    "required": ["query"]
                }
            ),
            genai.protos.FunctionDeclaration(
                name="rank_interesting_news",
                description="Get the most legally interesting news.",
                parameters={
                    "type": "object",
                    "properties": {
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results (default 10)"
                        }
                    }
                }
            )
        ]
    )
]

def convert_messages_for_gemini(messages):
    """Convert shared message format to Gemini format"""
    gemini_history = []
    for msg in messages:
        if msg['role'] == 'system':
            continue  # Gemini uses system_instruction instead
        elif msg['role'] == 'user':
            gemini_history.append({
                'role': 'user',
                'parts': [msg['content']]
            })
        elif msg['role'] == 'assistant':
            gemini_history.append({
                'role': 'model',
                'parts': [msg['content']]
            })
    return gemini_history

SYSTEM_PROMPT  = """You are a news assistant for a global law firm. You help lawyers find and understand relevant news.

When answering questions:
1. Always base answers on the news articles provided
2. Provide article titles, companies, and URLs
3. Be concise and professional
4. If asked about interesting news, focus on legal implications

You have access to functions to search url of articles and find interesting news in general. Use them appropriately."""

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Gemini"

if model_choice != st.session_state.selected_model:
        st.session_state.selected_model = model_choice
        st.rerun()

for message in st.session_state.messages:
    if message['role'] == 'system':
            continue
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chat input
if prompt := st.chat_input("Ask about news..."):

    relevant_context = get_context(prompt, n_results=3)
    if len(st.session_state.messages) == 0:
        st.session_state.messages.append({
            'role': 'system',
            'content': SYSTEM_PROMPT
        })
    st.session_state.messages.append({'role': 'user', 'content': f"{prompt}\n\n{relevant_context}"})
    
    with st.chat_message('user'):
        st.markdown(prompt)
    
    with st.chat_message('assistant'):
        message_placeholder = st.empty()

        if model_choice == "Chatgpt":
            # Call OpenAI with function calling
            response = st.session_state.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages,
                tools=tools,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            print(tool_calls,'tool_calls')
            
            if tool_calls:
                # Add assistant's tool call message
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': response_message.content,
                    'tool_calls': [
                        {
                            'id': tc.id,
                            'type': tc.type,
                            'function': {
                                'name': tc.function.name,
                                'arguments': tc.function.arguments
                            }
                        } for tc in tool_calls
                    ]
                })
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name == "search_news":
                        function_response = search_news(**function_args)
                    elif function_name == "rank_interesting_news":
                        function_response = rank_interesting_news(**function_args)
                    
                    # Add function response to messages
                    st.session_state.messages.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'name': function_name,
                        'content': json.dumps(function_response)
                    })
                
                # Get final response from model with function results
                second_response = st.session_state.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.messages
                )
                
                final_message = second_response.choices[0].message.content
            else:
                final_message = response_message.content
        else:
            gemini_history = convert_messages_for_gemini(st.session_state.messages)
    
            # Create model with tools
            model = genai.GenerativeModel(
                model_name='gemini-1.5-flash',
                tools=gemini_tools,
                system_instruction=SYSTEM_PROMPT
            )
            
            chat = model.start_chat(history=gemini_history[:-1] if gemini_history else [])
            
            # Send last message
            last_message = gemini_history[-1]['parts'][0] if gemini_history else ""
            response = chat.send_message(last_message)
            
            # Handle function calls
            if response.candidates[0].content.parts[0].function_call:
                function_call = response.candidates[0].content.parts[0].function_call
                function_name = function_call.name
                function_args = dict(function_call.args)
                
                # Execute function
                if function_name == "search_news":
                    function_response = search_news(**function_args)
                elif function_name == "rank_interesting_news":
                    function_response = rank_interesting_news(**function_args)
                
                # Send function response back
                response = chat.send_message(
                    genai.protos.Content(
                        parts=[genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=function_name,
                                response={'result': function_response}
                            )
                        )]
                    )
                )
            
            final_message = response.text
        

        # Display and save final response
        st.markdown(final_message)
        st.session_state.messages.append({'role': 'assistant', 'content': final_message})
