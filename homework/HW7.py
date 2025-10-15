__import__('pysqlite3')
import sys 
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from openai import OpenAI
import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
import json

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

# results = collection.query(
#     query_texts=["jp mprgan zelle"],
#     n_results=1
# )

# print("\n📰 Top 3 results for 'corporate law':")
# for i, meta in enumerate(results['metadatas'][0]):
#     print(f"{i+1}. {meta['URL']}")


# Initialize OpenAI client
if 'openai_client' not in st.session_state:
    openai_api_key = st.secrets["API_KEY"]
    st.session_state.openai_client = OpenAI(api_key=openai_api_key)

    # Initialize messages in session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'Hi! I am your news assistant. How can I help you?'}]

def search_news(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    articles = []
    for i in range(len(results['ids'][0])):
        articles.append({
            "title": results['metadatas'][0][i]['title'],
            "URL": results['metadatas'][0][i]['URL'],
            "company": results['metadatas'][0][i]['company'],
            "date": results['metadatas'][0][i]['date'],
            "relevance_score": 1 - results['distances'][0][i]
        })
    
    return articles

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
]

# Chat input
if prompt := st.chat_input("Ask about news..."):
    # Add user message to chat
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    with st.chat_message('user'):
        st.markdown(prompt)
    
    # Get AI response with function calling
    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        
        # Call OpenAI with function calling
        response = st.session_state.openai_client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4o" for better quality
            messages=st.session_state.messages,
            tools=tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        # Check if the model wants to call a function
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
            
            # Execute each function call
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Call the appropriate function
                if function_name == "search_news":
                    function_response = search_news(**function_args)
                # elif function_name == "rank_interesting_news":
                #     function_response = rank_interesting_news(**function_args)
                
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
        
        # Display and save final response
        message_placeholder.markdown(final_message)
        st.session_state.messages.append({'role': 'assistant', 'content': final_message})
