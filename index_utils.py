"""
Builds a Pinecone index for LLM applications involving RAG.
Each document in the index also has metadata containing ACL
information such as groups that have read permission on that document.
"""

import io
import logging
import os
import traceback

from dataclasses import dataclass

# Retrieval and indexing of web data.
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Pinecone
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_pinecone import PineconeVectorStore

# Google Drive.
from googleapiclient.discovery import Resource
from googleapiclient.http import MediaIoBaseDownload

from typing import Any, Dict, List, Optional, Set, Tuple, Union

@dataclass
class Document:
  # unique id of this document.
  file_id: str

  # filename, e.g. "abc.pdf"
  name: str

  # List of groups or accounts that have read access to this document.
  # E.g. ["hr_users", "finance_users", "engineering_users", ...]
  read_access: List[str]

  # Last modified time of this file or document.
  modified_time: Optional[str] = None

  # Size in bytes, of the data in this document.
  size: Optional[str] = None

  def __eq__(self, other):
    return (
      other.file_id == self.file_id and
      other.name == self.name and
      len(other.read_access) == len(self.read_access) and
      all([item[0] == item[1] for item in zip(other.read_access, self.read_access)]) and
      other.modified_time == self.modified_time and
      other.size == self.size)

# An index manifest represents the state of an index at a point in time.
# It stores metadata of all the documents in the index.
IndexManifest = Dict[str, Document]

def read_documents(
    folder_id: str,
    drive_service: Resource,
    include_files_with_extensions: Optional[List[str]] = None) -> List[Document]:
  """
  Retrieves the metadata of all files in the provided folder
  (recursively traversing sub-folders).

  Args:
    folder_id: The id of the folder in Drive whose files' metadata will be fetched.
    drive_service: The drive resource to use for fetching data.
    include_files_with_extension: If set, only include files with the specified
        extensions (e.g. "pdf").

  Returns: A list of documents from provided folder, traversed recursively.
  """
  if include_files_with_extensions is None:
    include_files_with_extensions = ["pdf"]

  files = drive_service.files().list(
      q=f"'{folder_id}' in parents and trashed=false",
      pageSize=10,
      fields="nextPageToken, files(id, name, permissions, mimeType, modifiedTime, size)").execute()
  items = files.get('files', [])

  if not items:
      return []

  result = []
  for item in items:
    # Found a folder.
    if "mimeType" in item and item["mimeType"].endswith(".folder"):
      logging.info(f"Traversing sub-folder '{item['name']}'.")
      folder_results = read_documents(
        folder_id=item['id'],
        drive_service=drive_service,
        include_files_with_extensions=include_files_with_extensions)
      result.extend(folder_results)
      continue

    # Found a file.
    if include_files_with_extensions is not None:
      if not any([item["name"].endswith(extension) for extension in include_files_with_extensions]):
        logging.warning(
          f"Ignoring file '{item['name']}' "
          f"because it doesn't match extensions {include_files_with_extensions}.")
        continue
    if "name" not in item or "id" not in item:
      logging.warning(
        f"Ignoring file '{item['name']}' because the "
        f"Google Drive API get response is malformed. {item=}")
      continue
    read_access = []
    if "permissions" in item:
      for permission in item["permissions"]:
        read_access.append(permission['displayName'])
    if not read_access:
      # Assumption: if item does not have permissions, it is a public document.
      read_access = ["all_users"]
    modified_time = None
    if "modifiedTime" in item:
      modified_time = item["modifiedTime"]
    size = None
    if "size" in item:
      size = item["size"]
    doc = Document(
      file_id=item["id"],
      name=item["name"],
      read_access=read_access,
      modified_time=modified_time,
      size=size)
    result.append(doc)
    logging.info(f"Read document from Drive: {doc}")
  return result
  

def read_file(file_id: str, drive_service: Resource) -> bytes:
  """Reads the contents of a file as bytes, from Google drive."""
  request = drive_service.files().get_media(fileId=file_id)
  downloaded = io.BytesIO()
  downloader = MediaIoBaseDownload(downloaded, request)
  done = False
  while done is False:
    status, done = downloader.next_chunk()
  downloaded.seek(0)
  logging.debug(f"Read {status.total_size} bytes from file id '{file_id}'.")
  return downloaded.read()

def clear_index(
    pinecone_api_key: str, pinecone_index_name: str) -> None:
  """
  Deletes the requested pinecone index data.
  The index itself is not deleted, but it is emptied.

  Raises:
    Exception if the index wasn't cleared successfully.
  """
  os.environ['PINECONE_API_KEY'] = pinecone_api_key
  vectorstore = PineconeVectorStore(
      index_name=pinecone_index_name,
      embedding=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
  try:
    vectorstore.delete(delete_all=True)
  except Exception as ex:
    if "Not Found" in str(ex):
      # already deleted.
      logging.warning(f"Index {pinecone_index_name} is already empty.")
    else:
      raise
  logging.warning(
    f"Successfully deleted all documents in index {pinecone_index_name}.")


def build_index(
    documents: List[Document],
    drive_service: Resource,
    pinecone_api_key: str,
    pinecone_index_name: str) -> Tuple[VectorStoreIndexWrapper, IndexManifest]:
  """
  Builds the Pinecone index with the supplied documents.
  Returns a VectorStoreIndexWrapper that can be used to query the index.

  Returns:
    A tuple of:
      1. VectorStoreIndexWrapper.
      2. Manifest of the index with metadata of all documents in it.
  """
  os.environ['PINECONE_API_KEY'] = pinecone_api_key
  embedding=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
  index_manifest: IndexManifest = {}
  indexable_documents = []
  for document in documents:
    print(f"Processing '{document.name}'.")
    if not os.path.isfile(document.name):
      with open(f"{document.name}", 'wb') as fd:
        fd.write(read_file(file_id=document.file_id, drive_service=drive_service))
      print(f"\tDownloaded '{document.name}' from '{document.file_id}'.")
    loader = PyPDFLoader(document.name)
    pages = loader.load_and_split()
    for page_index, page in enumerate(pages):
      readable_page_content = page.page_content.replace("\n", " ")[:300]
      page.metadata["read_access"] = ",".join(sorted(document.read_access))
      page.metadata["name"] = document.name
      page.metadata["file_id"] = document.file_id
      if document.modified_time is not None:
        page.metadata["modified_time"] = document.modified_time
      if document.size is not None:
        page.metadata["size"] = document.size
      index_manifest[document.file_id] = document
      indexable_documents.append(page)
      print(f'\tPage {page_index}: {readable_page_content} ...')
      for metadata_key, metadata_val in page.metadata.items():
        if metadata_key == "name":
          continue
        print(f'\t\t{metadata_key}: {metadata_val}')
  vectorstore = PineconeVectorStore(
      index_name=pinecone_index_name, embedding=embedding)
  vs = vectorstore.from_documents(
      documents=indexable_documents,
      embedding=embedding,
      index_name=pinecone_index_name)
  return VectorStoreIndexWrapper(vectorstore=vs), index_manifest

if __name__ == "__main__":
    print(f"Please run the notebook, which imports this module.")
