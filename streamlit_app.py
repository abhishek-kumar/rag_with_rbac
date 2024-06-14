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

from typing import List, Optional, Tuple


# OpenAIEmbeddings model.
_MODEL_NAME: str = "text-embedding-3-small"


def nav_to(url: str):
    js = f'window.open("{url}").then(r => window.parent.location.href);'
    st_javascript(js)

def escape_markdown(text: str) -> str:
    match_md = r'((([_*]).+?\3[^_*]*)*)([_*])'
    result = re.sub(match_md, "\g<1>\\\\\g<4>", text)
    return result.replace("$", "\$")

def get_justification_message_for_updating_index(
        sources_to_be_deleted: index_utils.IndexManifest,
        to_be_added: index_utils.IndexManifest,
        sources_to_be_updated: index_utils.IndexManifest) -> str:
    """
    Generates a user-readable message justifying why we need to
    update our index before answering their question.
    """
    message: str = ""
    delimiter = ""
    if sources_to_be_deleted:
        message += (
            "My response is drawn from deleted sources "
            f"{', '.join([doc.name for doc in sources_to_be_deleted.values()])}, "
            "Since my response is stale, I need to update "
            "my data before giving you an accurate response.")
        delimiter = " "
    if to_be_added:
        message += (
            f"{delimiter}There are new file(s) that have been recently added "
            "which might have a better response to your query, "
            "but I haven't processed them yet: "
            f"{', '.join([doc.name for doc in to_be_added.values()])}.")
        delimiter = " "
    if sources_to_be_updated:
        message += (
            f"{delimiter}My response is drawn from stale sources "
            f"{', '.join([doc.name for doc in sources_to_be_updated])}, "
            "which have been modified recently. "
            "Since my response is stale, I need to update "
            "my data before giving you a more accurate response.")
    if message:
        logging.info(message)
    return message


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
        url=authorization_url)

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

def check_for_updates(
        sources: List[str],
        latest_documents: List[index_utils.Document],
        existing_index_manifest: index_utils.IndexManifest) -> Tuple[index_utils.IndexManifest, index_utils.IndexManifest, index_utils.IndexManifest]:
    """
    Checks if the sources of an LLM's response are out of date w.r.t. Google Drive.
    Returns a tuple of:
      1. Documents to be deleted.
      2. Documents to be added (new).
      3. Documents in the index to be updated.
    """
    try:
        source_set = set(sources)
        logging.info(f"check_for_updates: Fetched {len(latest_documents)} documents from Google Drive.")
        new_manifest = {doc.file_id: doc for doc in latest_documents}
        to_be_deleted, to_be_added, to_be_updated = index_utils.compare_index_manifests(
            previous_manifest=existing_index_manifest,
            new_manifest=new_manifest)
        logging.info(
            "check_for_updates: Comparing the index to latest document metadata, "
            "the following updates are necessary:\n"
            f"  - {len(to_be_deleted)} documents to be deleted.\n"
            f"  - {len(to_be_added)} new documents to be added.\n"
            f"  - {len(to_be_updated)} documents to be updated.")
        sources_to_be_deleted: index_utils.IndexManifest = {
            doc.file_id: doc for doc in to_be_deleted.values()
            if doc.name in source_set
        }
        sources_to_be_updated: index_utils.IndexManifest = {
            doc.file_id: doc for doc in to_be_updated.values()
            if existing_index_manifest[doc.file_id].name in source_set
        }
        return (sources_to_be_deleted, to_be_added, sources_to_be_updated)
    except Exception as ex:
        logging.error(ex)
        logging.error(traceback.format_exc())
        raise Exception(f"Unable to check provenance of the sources in the LLM's response. {ex}.")

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
    sources = [source for source in sources if source != ""]
    logging.info(f"query_rag_with_rbac: user groups {groups}.")
    logging.info(f"query_rag_with_rbac: answer:\n{answer}")
    logging.info(f"query_rag_with_rbac: sources:\n{sources}")
    return (answer, sources)

def query_rag_and_check_drive_for_updates(
        input_text: str,
        llm: OpenAI,
        index: VectorStoreIndexWrapper,
        groups: List[str],
        pinecone_api_key:str ,
        pinecone_index_name: str,
        google_drive_root_folder_id: str) -> Tuple[str, List[str], index_utils.IndexManifest, Optional[str], List[index_utils.Document]]:
    """
    Returns a tuple of:
      1. Answer.
      2. Sources.
      3. Index manifest (from the index) as of now.
      4. Justification message in case we need to update the index, None otherwise.
      5. List of documents that should be represented in the index to answer the given user query.
    """
    answer: str
    sources: List[str]
    index_manifest: index_utils.IndexManifest
    latest_documents: List[index_utils.Document]
    updates: Optional[str]

    with st.spinner(text="Asking the LLM..."):
        answer, sources = query_rag_with_rbac(
            input_text=input_text,
            llm=llm,
            index=index,
            groups=groups)
    st.success(
        body="Successfully fetched the results and prepared a response.",
        icon=":material/check_circle:")

    with st.spinner(f"Analyzing sources for correctness: {', '.join(sources)}"):
        index_manifest = index_utils.get_index_manifest(
            pinecone_api_key=pinecone_api_key,
            pinecone_index_name=pinecone_index_name)
        latest_documents = index_utils.read_documents(
                folder_id=google_drive_root_folder_id,
                drive_service=st.session_state.drive_service,
                include_files_with_extensions=["pdf"])
        sources_to_be_deleted, to_be_added, sources_to_be_updated = check_for_updates(
            sources=sources,
            latest_documents=latest_documents,
            existing_index_manifest=index_manifest)
        documents = []  # We'll update our index to have all these documents only.
        for doc in index_manifest.values():  # Existing documents in index.
            if doc.file_id in sources_to_be_deleted:
                continue
            if doc.file_id in sources_to_be_updated:
                documents.append(sources_to_be_updated[doc.file_id])
                continue
            documents.append(doc)
        for doc in to_be_added.values():
            documents.append(doc)

    updates = None
    if sources_to_be_deleted or to_be_added or sources_to_be_updated:
        updates = get_justification_message_for_updating_index(
            sources_to_be_deleted=sources_to_be_deleted,
            to_be_added=to_be_added,
            sources_to_be_updated=sources_to_be_updated)
        st.warning(
            body="Done analyzing sources for correctness.",
            icon=":material/sync_problem:")
    else:
        st.success(
            body="Done analyzing sources for correctness.",
            icon=":material/inventory:")        
    return (answer, sources, index_manifest, updates, documents)

def main():
    st.title('LLM App: RAG with RBAC v0.2')

    # Step 1: handle authentication.
    if not is_user_authenticated():
        client_secrets = os.environ.get("GOOGLE_AUTH_CLIENT_SECRETS", "")
        redirect_uri = os.environ.get("REDIRECT_URI", "https://rag-rbac.streamlit.app")
        if not client_secrets:
            client_secrets = st.sidebar.text_area('Google auth client secrets (JSON)')
        if not client_secrets:
            st.warning("Please enter Google Auth secrets JSON (in sidebar)!",
                       icon=":material/edit_note:")
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
        st.warning("Please enter your OpenAI API key!",
                   icon=":material/edit_note:")
        return
    llm = get_llm(openai_api_key=openai_api_key)    

    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
    if not pinecone_api_key:
        pinecone_api_key = st.sidebar.text_input('Pinecone API key', type='password')
    if not pinecone_api_key:
        st.warning("Please enter your Pinecone API key!",
                   icon=":material/edit_note:")
        return
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "")    
    if not pinecone_index_name:
        pinecone_index_name = st.sidebar.text_input('Pinecone index name')
    if not pinecone_index_name:
        st.warning("Please enter your Pinecone index name!",
                   icon=":material/edit_note:")
        return
    index = get_vectorstore_indexwrapper(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name)
    
    # Google Drive.
    google_drive_root_folder_id = os.environ.get("GOOGLE_DRIVE_ROOT_FOLDER_ID", "")
    if not google_drive_root_folder_id:
        google_drive_root_folder_id = st.sidebar.text_input("Google Drive root folder id")
    if not google_drive_root_folder_id:
        st.warning(
            "Please enter your Google Drive root folder id "
            "(that will be traversed to construct the vector store)",
            icon=":material/edit_note:")
        return
    
    # Collect the groups that the user is a part of.
    user_groups = st.sidebar.text_input(label="User groups", value="fin_users, eng_users")
    collect_groups = lambda x : [f"{group.strip()}" for group in x.split(',') if group.strip() != ""]
    collected_groups = collect_groups(user_groups)
    groups_markdown = [f":blue-background[{group}]" for group in collected_groups]
    st.sidebar.markdown(body=" ".join(groups_markdown))
    collected_groups.append(index_utils.PUBLIC_USERS_GROUP)
    if email:
        collected_groups.append(email.split("@")[0])
    logging.info(f"Full list of user's read_access membership set: {collected_groups}.")
    
    # Step 3: The main form that does RAG with RBAC.
    with st.form("ask_copilot_form"):
        text = st.text_area(
            label="Your question",
            value="Tell me about the quarterly projections.",
            label_visibility="hidden")
        submitted = st.form_submit_button(label="Submit")
        if submitted and llm and index:
            try:
                answer, sources, index_manifest, updates, latest_documents = query_rag_and_check_drive_for_updates(
                    input_text=text,
                    llm=llm,
                    index=index,
                    groups=collected_groups,
                    pinecone_api_key=pinecone_api_key,
                    pinecone_index_name=pinecone_index_name,
                    google_drive_root_folder_id=google_drive_root_folder_id)
                num_index_updates = 0
                while updates is not None:
                    if num_index_updates >= 2:
                        raise Exception(
                            f"Too many frequent and recent updates to files "
                            "in Google Drive. Please check back later after "
                            "all changes are done.")
                    st.markdown(
                        body=f"**:blue[Copilot:]** *{escape_markdown(updates)}*\n\n",
                        help="Response from the LLM with RAC, applying role based access controls.")
                    with st.spinner("Updating the index with recent file changes..."):
                        docs_updated = index_utils.update_index_with_latest_documents(
                            latest_documents=latest_documents,
                            drive_service=st.session_state.drive_service,
                            pinecone_api_key=pinecone_api_key,
                            pinecone_index_name=pinecone_index_name,
                            existing_index_manifest=index_manifest,
                            write_index_manifest_after_update=True)
                        num_index_updates += 1
                    logging.info(f"Updated the index. {docs_updated=}")
                    st.success(f"Updated the knowledge base with {len(docs_updated)} recent change(s).",
                               icon=":material/drive_folder_upload:")
                    answer, sources, index_manifest, updates, latest_documents = query_rag_and_check_drive_for_updates(
                        input_text=text,
                        llm=llm,
                        index=index,
                        groups=collected_groups,
                        pinecone_api_key=pinecone_api_key,
                        pinecone_index_name=pinecone_index_name,
                        google_drive_root_folder_id=google_drive_root_folder_id)
                response_markdown = f"**:blue[Copilot:]** *{answer}*\n\n"
                if sources:
                    response_markdown += f"**:blue[Sources]**\n\n"
                    for source in sources:
                        response_markdown += f"  * :blue-background[{source}]\n"              
                st.markdown(
                    body=response_markdown,
                    help="Response from the LLM with RAC, applying role based access controls.")                
            except Exception as ex:
                st.warning(f"LLM/index error: {str(ex)}", icon=":material/error:")
                logging.warning(traceback.format_exc())


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    try:
        main()
    except Exception as ex:
        st.warning(f"Internal error. {ex}", icon=":material/error:")
        logging.warning(traceback.format_exc())
