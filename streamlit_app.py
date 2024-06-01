import streamlit as st
from langchain.llms import OpenAI

st.title('LLM App: RAG with RBAC v0.01')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

def generate_response(input_text):
    llm = OpenAI(temperature=0.6, openai_api_key=openai_api_key)
    st.info(llm(input_text))

with st.form('simple_llm_form'):
    text = st.text_area('Ask the copilot:', 'What can you tell me about quarterly projections?')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)
