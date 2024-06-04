import json
import logging
import os
import traceback

import google_auth_oauthlib.flow
from googleapiclient.discovery import build

import streamlit as st
from streamlit_javascript import st_javascript

from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.embeddings import SentenceTransformerEmbeddings

from pinecone import Pinecone

def nav_to(url):
    js = f'window.open("{url}", "_self").then(r => window.parent.location.href);'
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
        authorization_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true")
        st.link_button(
            label="Sign in with Google",
            url=authorization_url,
            help=f"Redirecting to '{authorization_url}' ...")

def generate_response(input_text, openai_api_key, index, groups):
    """Generates response via OpenAI LLM call."""
    try:
        llm = OpenAI(temperature=0.2, openai_api_key=openai_api_key)
        response = index.query_with_sources(
            input_text,
            llm=llm,
            retriever_kwargs={
                "search_kwargs": {
                    "filter": {
                        "roles": {"$in": [f"role:{group}" for group in groups]}}}}
        )
        sources = ""
        for source in response['sources'].split(','):
            sources += f"  * :blue-background[{source.strip()}]\n"
        st.markdown(
            body=f"**:blue[Copilot:]** *```text\n{response['answer'].strip()}\n```*\n{sources}",
            help="Response from the LLM with RAC, applying role based access controls.")
        st.warning(f"DEBUG: user groups {groups}.")
        st.warning(f"DEBUG: answer:\n{response['answer'].strip()}")
        logging.warning(f"DEBUG: answer:\n{response['answer'].strip()}")
    except Exception as ex:
        st.warning(f"OpenAI Error: {str(ex)}")


def main():
    redirect_uri = os.environ.get("REDIRECT_URI", "https://rag-rbac.streamlit.app")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    client_secrets = os.environ.get("GOOGLE_AUTH_CLIENT_SECRETS", "")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "")

    st.title('LLM App: RAG with RBAC v0.01')
    if not openai_api_key:
        openai_api_key = st.sidebar.text_input('OpenAI API key', type='password')
    if not pinecone_api_key:
        pinecone_api_key = st.sidebar.text_input('Pinecone API key', type='password')
    if not pinecone_index_name:
        pinecone_index_name = st.sidebar.text_input('Pinecone index name')
    if not client_secrets:
        client_secrets = st.sidebar.text_area('Google auth client secrets (JSON)')
    user_groups = st.sidebar.text_input(label="User groups", value="fin_users, engineering")
    collect_groups = lambda x : [f"{group.strip()}" for group in x.split(',') if group.strip() != ""]
    collected_groups = collect_groups(user_groups)
    groups_markdown = [f":blue-background[{group}]" for group in collected_groups]
    st.sidebar.markdown(body=" ".join(groups_markdown))

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
        text = st.text_area(label='', value='Tell me about the quarterly projections.')
        submitted = st.form_submit_button('Submit')
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
        if not pinecone_api_key:
            st.warning('Please enter your Pinecone API key!', icon='⚠')
        if not pinecone_index_name:
            st.warning('Please enter your Pinecone index name!', icon='⚠')
        if submitted and openai_api_key.startswith('sk-'):
            os.environ['PINECONE_API_KEY'] = pinecone_api_key
            # st.write(f"DEBUG: {pinecone_api_key=}, {pinecone_index_name=}.")
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index(pinecone_index_name)
            try:
                embed = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = PineconeVectorStore(index, embed.embed_query, "text")
                index_wrapper = VectorStoreIndexWrapper(vectorstore=vector_store)
                generate_response(
                    text,
                    openai_api_key,
                    index_wrapper,
                    collected_groups)
            except Exception as ex:
                st.warning(traceback.format_exc())


if __name__ == "__main__":
    main()
