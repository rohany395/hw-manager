import streamlit as st
from openai import OpenAI
import bs4
import requests
import anthropic

#sidebar
add_sidebar=st.sidebar.selectbox("Summary options",("Summarize the document in 100 words","Summarize the document in 2 connecting paragraphs","Summarize the document in 5 bullet points"))
modelSelected=add_sidebar=st.sidebar.selectbox("Summary options",("Gemini","Chatgpt","Claude"))


# Show title and description.
st.title("📄 Rohan's URL summarizer")

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = bs4.BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

if modelSelected =="Gemini":
    client = OpenAI(
        api_key=st.secrets["GEMINI_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
elif modelSelected =="Chatgpt":
    client = OpenAI(api_key=st.secrets["API_KEY"])
elif modelSelected =="Claude":
    client = anthropic.Anthropic(api_key=st.secrets["claude"]["CLAUDE_key"])

givenUrl = st.text_area(
    "Upload a URL that you want to summarize",
    placeholder="www.xyz.com",
)

#checkbox
modelVersion=st.radio("Model",("Mini","Advanced Model"))

#language dropdown
options = ["French", "German", "English", "Hindi"]
languageChoice = st.selectbox("Choose an option:", options)

gptVersion="gpt-5"

if modelVersion=="Mini" and modelSelected=="Chatgpt":
    gptVersion="gpt-4o-mini"
elif modelVersion=="Advanced Model" and modelSelected=="Chatgpt":
    gptVersion="gpt-4o"
elif modelVersion=="Mini" and modelSelected=="Gemini":
    gptVersion="gemini-2.5-flash"
elif modelVersion=="Advanced Model" and modelSelected=="Gemini":
    gptVersion="gemini-2.5-pro"
elif modelVersion=="Mini" and modelSelected=="Claude":
    gptVersion="claude-3-5-haiku-20240620"
elif modelVersion=="Advanced Model" and modelSelected=="Claude":
    gptVersion="claude-3-5-sonnet-20240620"

if givenUrl and add_sidebar:

    # Process the uploaded file and question.
    document = read_url_content(givenUrl)
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant that always replies in {languageChoice}."
        },
        {
            "role": "user",
            "content": f"Here's a document: {document} \n\n---\n\n {add_sidebar}",
        }
    ]

    # Generate an answer using the OpenAI API.
    

    if modelSelected == "Claude":
        stream = client.messages.stream(
            model=gptVersion,
            max_tokens=500,
            messages=messages,
        )
    else:
        stream = client.chat.completions.create(
            model=gptVersion,
            messages=messages,
            stream=True,
        )

    # Stream the response to the app using `st.write_stream`.
    st.write_stream(stream)