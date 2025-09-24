import streamlit as st
from openai import OpenAI
import requests
import bs4
import anthropic
import google.generativeai as genai

# Show title and description.
st.title("📄 Rohan's Chatbot")

#sidebar
url1=url1=st.sidebar.text_area("Upload your first url",placeholder="www.xyz.com")
url2=st.sidebar.text_area("Upload your second url (optional)",placeholder="www.xyz.com")

modelSelected=st.sidebar.selectbox("Select a service",("Gemini","Chatgpt","Claude"))

def read_url_content(url):
    if not url or not url.strip():
        return None
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = bs4.BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None


#checkbox
modelVersion=st.radio("Model",("Mini","Advanced Model"))
gptVersion="gpt-5"

if modelVersion=="Mini" and modelSelected=="Chatgpt":
    gptVersion="gpt-4o-mini"
elif modelVersion=="Advanced Model" and modelSelected=="Chatgpt":
    gptVersion="gpt-4o"
elif modelVersion=="Mini" and modelSelected=="Gemini":
    gptVersion="gemini-2.0-flash"
elif modelVersion=="Advanced Model" and modelSelected=="Gemini":
    gptVersion="gemini-2.5-flash"
elif modelVersion=="Mini" and modelSelected=="Claude":
    gptVersion="claude-3-5-haiku-20241022"
elif modelVersion=="Advanced Model" and modelSelected=="Claude":
    gptVersion="claude-3-5-sonnet-20240620"

#memory type
memory_type = st.sidebar.selectbox(
    "Conversation Memory",
    ("Buffer of 6 questions", "Buffer of 2,000 tokens")
)


if 'client' not in st.session_state:
    openai_api_key = st.secrets["API_KEY"]
    st.session_state.client = OpenAI(api_key=openai_api_key)

if 'messages' not in st.session_state:
    st.session_state['messages']=[{'role':'assistant','content':'Hi how can I help?'}]

if modelSelected =="Gemini":
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
    client = genai.GenerativeModel(gptVersion)
elif modelSelected =="Chatgpt":
    client = OpenAI(api_key=st.secrets["API_KEY"])
elif modelSelected =="Claude":
    client = anthropic.Anthropic(api_key=st.secrets["CLAUDE_KEY"])



def convert_messages_for_gemini(messages):
    gemini_messages = []
    for msg in messages:
        if msg['role'] == 'user':
            gemini_messages.append({'role': 'user', 'parts': [msg['content']]})
        elif msg['role'] == 'assistant':
            gemini_messages.append({'role': 'model', 'parts': [msg['content']]})
    return gemini_messages


document1 = read_url_content(url1)
document2 = read_url_content(url2)
#chat input
if prompt:=st.chat_input('Talk to me Goose'):
    if memory_type == "Buffer of 6 questions" and len(st.session_state.messages) > 12:
        st.session_state.messages = st.session_state.messages[-12:]
    elif memory_type == "Buffer of 2,000 tokens":
        #remove oldest if messages exceed 2,000 tokens
        while len(st.session_state.messages) > 50:  # rough approximation
            st.session_state.messages.pop(0)

    # Append user message
    content = prompt
    if document1:
        content += f" First document is the following: {document1}"
    if document2:
        content += f" Second document is the following: {document2}"
    st.session_state.messages.append({'role':'user','content':content})

for msg in st.session_state.messages:
    chat_msg=st.chat_message(msg['role'])
    chat_msg.write(msg['content'])
    

# Generate an answer using the OpenAI API.

if modelSelected == "Gemini":
    # Method 1: Using Google's official generativeai library
    def generate_gemini_response():
        try:
            # Convert messages to Gemini format
            gemini_messages = convert_messages_for_gemini(st.session_state.messages[:-1])
            
            # Start a chat session
            chat = client.start_chat(history=gemini_messages)
            
            response = chat.send_message(content, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            yield f"Error: {str(e)}"

    with st.chat_message('assistant'):
        response = st.write_stream(generate_gemini_response())
    
    st.session_state.messages.append({'role':'assistant','content':response})

elif modelSelected == "Claude":
        def generate_response():
            with client.messages.stream(
                model=gptVersion,
                max_tokens=1024,
                system="You are a helpful assistant that always replies nicely.",
                messages=st.session_state.messages
            ) as stream:
                for text in stream.text_stream:
                    yield text

        with st.chat_message('assistant'):
            response=st.write_stream(generate_response())

        st.session_state.messages.append({'role':'assistant','content':f'{response}'})
else:
    if prompt:
        client=st.session_state.client
        stream = client.chat.completions.create(
            model=gptVersion,
            messages=st.session_state.messages,
            stream=True,
        )

        with st.chat_message('assistant'):
            response=st.write_stream(stream)

        st.session_state.messages.append({'role':'assistant','content':f'{response}'})

    
