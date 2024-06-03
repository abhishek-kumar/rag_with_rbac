import json
import os

import google_auth_oauthlib.flow
from googleapiclient.discovery import build

import streamlit as st
from streamlit_javascript import st_javascript

from langchain.llms import OpenAI


def nav_to(url):
    js = f'window.open("{url}").then(r => window.parent.location.href);'
    st_javascript(js)

def auth_flow(client_secrets, redirect_uri):
    """Handles user authentication via Google OAuth."""
    st.write("Retrieval-augmented-generation with role based access control.")
    auth_code = st.query_params.get("code")
    client_secrets_json = json.loads(client_secrets)
    flow = google_auth_oauthlib.flow.Flow.from_client_config(
        client_secrets_json,
        scopes=["https://www.googleapis.com/auth/userinfo.email", "openid"],
        redirect_uri=redirect_uri,
    )
    if auth_code:
        try:
            flow.fetch_token(code=auth_code)
        except Exception:
            auth_code = None
    if auth_code:
        credentials = flow.credentials
        user_info_service = build(
            serviceName="oauth2",
            version="v2",
            credentials=credentials,
        )
        user_info = user_info_service.userinfo().get().execute()
        assert user_info.get("email"), "Email not found in infos"
        st.session_state["google_auth_code"] = auth_code
        st.session_state["user_info"] = user_info
    else:
        if st.button("Sign in with Google"):
            authorization_url, state = flow.authorization_url(
                access_type="offline",
                include_granted_scopes="true",
            )
            st.warning(f"DEBUG: Opening new tab with '{authorization_url}'...")
            nav_to(authorization_url)


def generate_response(input_text, openai_api_key):
    """Generates response via OpenAI LLM call."""
    try:
        llm = OpenAI(temperature=0.6, openai_api_key=openai_api_key)
        st.info(llm(input_text))
    except Exception as ex:
        st.warning(f"OpenAI Error: {str(ex)}")


def main():
    redirect_uri = os.environ.get("REDIRECT_URI", "https://rag-rbac.streamlit.app")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    client_secrets = os.environ.get("GOOGLE_AUTH_CLIENT_SECRETS", "")

    st.title('LLM App: RAG with RBAC v0.01')
    if not openai_api_key:
        openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

    if not client_secrets:
        client_secrets = st.sidebar.text_area('Google auth client secrets (JSON)')

    if "google_auth_code" not in st.session_state:
        if not client_secrets:
            st.warning('Please enter Google Auth secrets JSON (in sidebar)!', icon='⚠')
            return
        auth_flow(client_secrets, redirect_uri)
        if "google_auth_code" not in st.session_state:
            return

    email = st.session_state["user_info"].get("email")
    st.write(f"Logged in: {email}")
    with st.form("ask_copilot_form"):
        text = st.text_area('Ask the copilot:', 'What can you tell me about quarterly projections?')
        submitted = st.form_submit_button('Submit')
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
        if submitted and openai_api_key.startswith('sk-'):
            generate_response(text, openai_api_key)


if __name__ == "__main__":
    main()
