import json
import logging
import os
import re
import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed

import google_auth_oauthlib.flow
from googleapiclient.discovery import Resource, build

import streamlit as st
from streamlit_javascript import st_javascript


from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings

from pinecone import Pinecone

import index_utils

from typing import List, Tuple


# OpenAIEmbeddings model.
_MODEL_NAME: str = "text-embedding-3-small"


def nav_to(url: str):
    js = f'window.open("{url}").then(r => window.parent.location.href);'
    st_javascript(js)

def escape_markdown(text: str) -> str:
    match_md = r'((([_*]).+?\3[^_*]*)*)([_*])'
    result = re.sub(match_md, "\g<1>\\\\\g<4>", text)
    return result.replace("$", "\$")

def is_user_authenticated() -> bool:
    return "google_auth_code" in st.session_state

def show_authentication_form_or_result(client_secrets: str, redirect_uri: str) -> None:
    """Handles user authentication via Google OAuth."""
    st.write("Retrieval-augmented-generation with role based access control.")
    auth_code = st.query_params.get("code")
    client_secrets_json = json.loads(client_secrets)
    flow = google_auth_oauthlib.flow.Flow.from_client_config(
        client_secrets_json,
        scopes=[
            "https://www.googleapis.com/auth/userinfo.email",
            "openid",
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/drive.metadata.readonly"
        ],
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
        drive_service: Resource = build(
            serviceName="drive",
            version="v3",
            credentials=credentials)
        st.session_state["drive_service"] = drive_service
        logging.info(f"Instantiated {drive_service=}.")
        user_info = user_info_service.userinfo().get().execute()
        assert user_info.get("email"), "Email not found in infos"
        st.session_state["google_auth_code"] = auth_code
        st.session_state["user_info"] = user_info
        return  # Authenticated.
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true")
    logging.info(f"Authenticating user. {authorization_url=}, {state=}.")
    st.link_button(
        label="Sign in with Google",
        url=authorization_url,
        help=f"Redirecting for authentication to '{authorization_url}' ...")

@st.cache_resource(ttl=600)
def get_llm(openai_api_key) -> OpenAI:
    return OpenAI(temperature=0.2, openai_api_key=openai_api_key)

@st.cache_resource(ttl=600)
def get_vectorstore_indexwrapper(
        pinecone_api_key: str, pinecone_index_name: str) -> VectorStoreIndexWrapper:
    os.environ['PINECONE_API_KEY'] = pinecone_api_key
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    embed = OpenAIEmbeddings(model=_MODEL_NAME)
    vector_store = PineconeVectorStore(index, embed.embed_query, "text")
    return VectorStoreIndexWrapper(vectorstore=vector_store)

def query_rag_with_rbac(
        input_text: str,
        llm: OpenAI,
        index: VectorStoreIndexWrapper,
        groups: List[str]) -> Tuple[str, List[str]]:
    """
    Generates response via OpenAI LLM call, with RAG - retrieval from index.

    Returns a tuple of <answer, list of sources>.
    """
    response = index.query_with_sources(
        input_text,
        llm=llm,
        retriever_kwargs={
            "search_kwargs": {
                "filter": {
                    "read_access": {"$in": groups}}}},
        reduce_k_below_max_tokens=True
    )
    answer = escape_markdown(response["answer"].strip())
    sources = [source.strip() for source in response["sources"].split(",")]
    logging.warning(f"DEBUG: query_rag_with_rbac: user groups {groups}.")
    logging.warning(f"DEBUG: query_rag_with_rbac: answer:\n{answer}")
    logging.warning(f"DEBUG: query_rag_with_rbac: sources:\n{sources}")
    return (answer, sources)


def main():
    st.title('LLM App: RAG with RBAC v0.1')

    # Step 1: handle authentication.
    if not is_user_authenticated():
        client_secrets = os.environ.get("GOOGLE_AUTH_CLIENT_SECRETS", "")
        redirect_uri = os.environ.get("REDIRECT_URI", "https://rag-rbac.streamlit.app")
        if not client_secrets:
            client_secrets = st.sidebar.text_area('Google auth client secrets (JSON)')
        if not client_secrets:
            st.warning('Please enter Google Auth secrets JSON (in sidebar)!', icon='⚠')
            return
        show_authentication_form_or_result(client_secrets, redirect_uri)
        if not is_user_authenticated():
            return
    email = st.session_state["user_info"].get("email")
    st.markdown(body=f"**:gray[Logged in:]** :blue-background[`{email}`]")
    
    # Step 2: initialize resources - OpenAI, Pinecone.
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        openai_api_key = st.sidebar.text_input('OpenAI API key', type='password')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
        return
    llm = get_llm(openai_api_key=openai_api_key)    

    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
    if not pinecone_api_key:
        pinecone_api_key = st.sidebar.text_input('Pinecone API key', type='password')
    if not pinecone_api_key:
        st.warning('Please enter your Pinecone API key!', icon='⚠')
        return
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "")    
    if not pinecone_index_name:
        pinecone_index_name = st.sidebar.text_input('Pinecone index name')
    if not pinecone_index_name:
        st.warning('Please enter your Pinecone index name!', icon='⚠')
        return
    index = get_vectorstore_indexwrapper(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name)
    
    # Google Drive.
    google_drive_root_folder_id = os.environ.get("GOOGLE_DRIVE_ROOT_FOLDER_ID", "")
    if not google_drive_root_folder_id:
        google_drive_root_folder_id = st.sidebar.text_input("Google Drive root folder id")
    if not google_drive_root_folder_id:
        st.warning('Please enter your Google Drive root folder id (that will be traversed to construct the vector store)', icon='⚠')
        return
    user_groups = st.sidebar.text_input(label="User groups", value="fin_users, engineering")
    collect_groups = lambda x : [f"{group.strip()}" for group in x.split(',') if group.strip() != ""]
    collected_groups = collect_groups(user_groups)
    groups_markdown = [f":blue-background[{group}]" for group in collected_groups]
    st.sidebar.markdown(body=" ".join(groups_markdown))
    collected_groups.append(index_utils.PUBLIC_USERS_GROUP)
    
    # Step 3: The main form that does RAG with RBAC.
    with st.form("ask_copilot_form"):
        text = st.text_area(label='', value='Tell me about the quarterly projections.')
        submitted = st.form_submit_button('Submit')
        if submitted and llm and index:
            try:
                answer, sources = query_rag_with_rbac(
                    input_text=text,
                    llm=llm,
                    index=index,
                    groups=collected_groups)
                # BEGIN: debug block (to be removed)
                try:
                    logging.info(f"Attempting to read files from drive.")
                    documents = index_utils.read_documents(
                        folder_id=google_drive_root_folder_id,
                        drive_service=st.session_state.drive_service,
                        include_files_with_extensions=["pdf"])
                    logging.info(f"DEBUG: fetched the following documents from Google Drive: {documents}.")
                except Exception as ex:
                    logging.error(ex)
                    logging.error(traceback.format_exc())
                # END: debug block (to be removed)
                response_markdown = f"**:blue[Copilot:]** *{answer}*\n\n"
                response_markdown += f"**:blue[Sources]**\n\n"
                for source in sources:
                    response_markdown += f"  * :blue-background[{source}]\n"
                if not sources:
                    response_markdown += f"  :blue-background[None found.]\n"                
                st.markdown(
                    body=response_markdown,
                    help="Response from the LLM with RAC, applying role based access controls.")
            except Exception as ex:
                st.warning(f"LLM/index error: {str(ex)}")
                logging.warning(traceback.format_exc())


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        st.warning(f"Internal error. {ex}")
        logging.warning(traceback.format_exc())
