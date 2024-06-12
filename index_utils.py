"""
Builds a Pinecone index for LLM applications involving RAG.
Each document in the index also has metadata containing ACL
information such as groups that have read permission on that document.
"""

import io
import logging
import os
import pickle
import sys
import traceback

import dataclasses
from dataclasses import dataclass

# Retrieval and indexing of web data.
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Pinecone
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Google Drive.
from googleapiclient.discovery import Resource
from googleapiclient.http import MediaIoBaseDownload

from typing import Any, Dict, List, Optional, Set, Tuple, Union


# OpenAIEmbeddings model.
_MODEL_NAME: str = "text-embedding-3-small"
_MODEL_EMBEDDING_SIZE: int = 1536

# Pinecone index constants.
_INDEX_MAXIMUM_METADATA_SIZE_BYTES = 35 * 1024  # The official limit is 40 kb.
_MANIFEST_KEY = "index_manifest"



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
  size: Optional[int] = None

  # If this document has been written to the index,
  # the list of record ids associated with this doc.
  index_record_ids: Optional[Set[str]] = None

  def __eq__(self, other):
    """Note that index_record_ids is not checked for equality."""
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
      logging.info(f"Reading from Google Drive: traversing sub-folder '{item['name']}'.")
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
          f"Reading from Google Drive: ignoring file '{item['name']}' "
          f"because it doesn't match extensions {include_files_with_extensions}.")
        continue
    if "name" not in item or "id" not in item:
      logging.warning(
        f"Reading from Google Drive: ignoring file '{item['name']}' because the "
        f"Google Drive API get response is malformed. {item=}")
      continue
    read_access = []
    if "permissions" in item:
      for permission in item["permissions"]:
        if "displayName" not in permission:
          if "id" in permission and permission["id"] == "anyoneWithLink":
            read_access = ["all_users"]
            break
          logging.error(
            f"Received {permission=} from Drive API without displayName in it. "
            "Check if you have authorized scopes to read this data.")
          continue
        read_access.append(permission["displayName"])
    if not read_access:
      # Assumption: if item does not have permissions, it is a public document.
      read_access = ["all_users"]
    modified_time = None
    if "modifiedTime" in item:
      modified_time = item["modifiedTime"]
    size = None
    if "size" in item:
      size = int(item["size"])
    doc = Document(
      file_id=item["id"],
      name=item["name"],
      read_access=read_access,
      modified_time=modified_time,
      size=size)
    result.append(doc)
    logging.info(f"Reading from Google Drive: read document from Drive:\n\t{doc}")
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
  logging.debug(f"Reading from Google Drive: read {status.total_size} bytes from file id '{file_id}'.")
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
      embedding=OpenAIEmbeddings(model=_MODEL_NAME))
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


def add_documents_to_index(
    documents: List[Document],
    drive_service: Resource,
    pinecone_api_key: str,
    pinecone_index_name: str,
    existing_index_manifest: Optional[IndexManifest] = None) -> IndexManifest:
  """
  Builds the Pinecone index with the supplied documents.
  Returns the updated IndexManifest with the newly added documents.
  """
  os.environ['PINECONE_API_KEY'] = pinecone_api_key
  embedding=OpenAIEmbeddings(model=_MODEL_NAME)

  # index_manifest will be the updated index manifest after we're done.
  index_manifest: IndexManifest
  if existing_index_manifest is not None:
    index_manifest = {k:v for k,v in existing_index_manifest.items()}
  else:
    index_manifest = get_index_manifest(
      pinecone_api_key=pinecone_api_key,
      pinecone_index_name=pinecone_index_name)
  existing_doc_ids = set([doc.file_id for doc in documents]).intersection(
    index_manifest.keys())
  if existing_doc_ids:
    msg = (
      f"Can't add documents {existing_doc_ids} -- already in the index. "
      f"{[index_manifest[doc_id] for doc_id in existing_doc_ids]}")
    logging.error(msg)
    raise ValueError(msg)

  vectorstore = PineconeVectorStore(
      index_name=pinecone_index_name, embedding=embedding)
  for document in documents:
    logging.info(f"Processing '{document.name}'.")
    if not os.path.isfile(document.name):
      with open(f"{document.name}", 'wb') as fd:
        fd.write(read_file(
          file_id=document.file_id,
          drive_service=drive_service))
      logging.info(f"\tDownloaded '{document.name}' from '{document.file_id}'.")
    loader = PyPDFLoader(document.name)
    pages = loader.load_and_split()
    index_records = []
    for page_index, page in enumerate(pages):
      readable_page_content = page.page_content.replace("\n", " ")[:300]
      page.metadata["read_access"] = list(sorted(document.read_access))
      page.metadata["name"] = document.name
      page.metadata["file_id"] = document.file_id
      if document.modified_time is not None:
        page.metadata["modified_time"] = document.modified_time
      if document.size is not None:
        page.metadata["size"] = str(document.size)
      index_records.append(page)
      logging.debug(f'\tPage {page_index}: {readable_page_content} ...')
      for metadata_key, metadata_val in page.metadata.items():
        if metadata_key == "name":
          continue
        logging.debug(f'\t\t{metadata_key}: {metadata_val}')
    if index_records:
      logging.info(
        f"\tUploading {len(index_records)} records (for {document.file_id=}, {document.name=}) "
        f"to the index '{pinecone_index_name}'.")
      added_record_ids = vectorstore.add_documents(index_records) # one per page.
      if len(added_record_ids) != len(index_records):
        logging.error(
          "\tError while uploading records to index. "
          f"We attempted to write {len(index_records)} records, "
          f"but got back confirmation for {len(added_record_ids)}. "
          f"{added_record_ids=}")
        # TODO: decide whether to raise an exception. For now, we continue.
      logging.info(
        f"\tUploaded {len(added_record_ids)} records (for {document.file_id=}, {document.name=}) "
        f"to the index '{pinecone_index_name}'. {added_record_ids=}")
      index_manifest[document.file_id] = dataclasses.replace(document, index_record_ids=set(added_record_ids))
    logging.info(f"Finished processing '{document.name}'.")
  # Update the index_manifest that tells the index what it holds.
  #write_index_manifest(
  #  index_manifest=index_manifest,
  #  pinecone_api_key=pinecone_api_key,
  #  pinecone_index_name=pinecone_index_name)
  return index_manifest

def delete_documents_from_index(
    to_be_deleted: IndexManifest,
    pinecone_api_key: str,
    pinecone_index_name: str,
    existing_index_manifest: Optional[IndexManifest] = None) -> IndexManifest:
  """
  Deletes the documents in provided manifest from the index.
  They should've been previously written to the index, for them to be
  deleted (the Documents in the manifest should have index_record_ids set
  from the result of the write to the index).

  Returns the updated IndexManifest after deleted the requested documents.
  """
  # index_manifest will be the updated index manifest after we're done.
  index_manifest: IndexManifest
  if existing_index_manifest is not None:
    index_manifest = {k:v for k,v in existing_index_manifest.items()}
  else:
    index_manifest = get_index_manifest(
      pinecone_api_key=pinecone_api_key,
      pinecone_index_name=pinecone_index_name)
  record_ids_to_delete: Set[str] = set()
  for doc in to_be_deleted.values():
    if not doc.index_record_ids:
      raise ValueError(
        f"Document {doc.file_id} ({doc.name}) "
        "has no records in the index to delete.")
    if doc.file_id not in index_manifest:
      raise ValueError(f"Document requested for deletion is not in index manifest: {doc}")
    record_ids_to_delete.update(doc.index_record_ids)
  if not record_ids_to_delete:
    logging.info(f"No records to delete for {to_be_deleted=}.")
    return
  logging.info(
    f"Going to delete {len(to_be_deleted)} documents "
    f"({len(record_ids_to_delete)} records) "
    f"from the index '{pinecone_index_name}'.")
  pc = Pinecone(api_key=pinecone_api_key)
  index = pc.Index(pinecone_index_name)
  delete_result = index.delete(ids=list(record_ids_to_delete))
  if delete_result:  # should be an empty dictionary on success
    raise ValueError(
      f"Failed to delete documents {to_be_deleted.keys()} "
      f"(records {record_ids_to_delete}).\n"
      f"{delete_result=}")
  return {k:v for k, v in index_manifest.items() if k not in to_be_deleted}

def get_index_manifest(
    pinecone_api_key: str, pinecone_index_name: str) -> IndexManifest:
  """Fetches the index manifest from Pinecone."""
  pc = Pinecone(api_key=pinecone_api_key)
  index = pc.Index(pinecone_index_name)
  vectors = index.fetch([_MANIFEST_KEY]).vectors
  if not vectors or _MANIFEST_KEY not in vectors:
    # Brand new index, or it was recently cleared so it has no data.
    return {}
  index_manifest_str = vectors[_MANIFEST_KEY]["metadata"][_MANIFEST_KEY]
  index_manifest_bytes = index_manifest_str.encode()
  index_manifest = pickle.loads(index_manifest_bytes)
  return index_manifest

def write_index_manifest(
    index_manifest: IndexManifest, pinecone_api_key: str, pinecone_index_name: str):
  """Writes (upserts) the provided index manifest to the index."""
  # Write index_manifest to the index as well, at the 0th vector.
  # Keep this in sync with de-serialization logic in get_index_manifest.
  serialized_manifest = pickle.dumps(index_manifest, protocol=0).decode()
  manifest_size = sys.getsizeof(serialized_manifest)
  manifest_vector = [0.0] * _MODEL_EMBEDDING_SIZE
  manifest_vector[-1] = 0.01  # Pinecone disallows the 0-vector.
  if manifest_size >= _INDEX_MAXIMUM_METADATA_SIZE_BYTES:
    raise ValueError(
      f"There are too many files for the index. "
      f"The index manifest is {manifest_size} bytes, "
      f"but the limit is {_INDEX_MAXIMUM_METADATA_SIZE_BYTES} bytes.")
  logging.info(f"Writing index manifest of size {manifest_size} bytes to index [0].")
  pc = Pinecone(api_key=pinecone_api_key)
  response = pc.Index(pinecone_index_name).upsert(
    vectors=[
      {
        "id": _MANIFEST_KEY, 
        "values": manifest_vector, 
        "metadata": {_MANIFEST_KEY: serialized_manifest}
      }
    ]
  )
  logging.info(f"Upsert index manifest {response=}.")

def compare_index_manifests(
    previous_manifest: IndexManifest,
    new_manifest: IndexManifest) -> Tuple[IndexManifest, IndexManifest, IndexManifest]:
  """
  Compares the new manifest with the previous (existing) index manifest and returns a tuple of:
    1. Manifest of documents to be deleted.
    2. Manifest of documents to be added (new).
    3. Manifest of documents that are to be modified.
  """
  all_ids = set(previous_manifest.keys()).union(new_manifest.keys())
  to_be_deleted: IndexManifest = {}
  to_be_added: IndexManifest = {}
  to_be_updated: IndexManifest = {}
  for doc_id in all_ids:
    if doc_id in previous_manifest and doc_id not in new_manifest:
      to_be_deleted[doc_id] = previous_manifest[doc_id]
      continue
    if doc_id not in previous_manifest and doc_id in new_manifest:
      to_be_added[doc_id] = new_manifest[doc_id]
      continue
    if previous_manifest[doc_id] == new_manifest[doc_id]:
      # Unchanged.
      continue
    to_be_updated[doc_id] = dataclasses.replace(new_manifest[doc_id], index_record_ids=previous_manifest[doc_id].index_record_ids)
  return (to_be_deleted, to_be_added, to_be_updated)

def update_index_with_latest_documents(
    latest_documents: List[Document],
    drive_service: Resource,
    pinecone_api_key: str,
    pinecone_index_name: str,
    existing_index_manifest: Optional[IndexManifest] = None,
    write_index_manifest_after_update: bool = True) -> List[str]:
  """
  Updates the index with a provided list of all latest documents that the index
  should have.

  Internally, we will only do the minimum necessary modifications required
  to update the index.

  Returns a list of document ids updated (deleted, added or modified).
  """
  # index_manifest will be the updated index manifest after we're done.
  index_manifest: IndexManifest
  if existing_index_manifest is not None:
    index_manifest = {k:v for k,v in existing_index_manifest.items()}
  else:
    index_manifest = get_index_manifest(
      pinecone_api_key=pinecone_api_key,
      pinecone_index_name=pinecone_index_name)
  new_manifest: IndexManifest = {doc.file_id: doc for doc in latest_documents}
  to_be_deleted, to_be_added, to_be_updated = compare_index_manifests(
    previous_manifest=index_manifest, new_manifest=new_manifest)
  logging.info(
    "Comparing the index to latest document metadata, "
    "the following updates will be performed:\n"
    f"  - {len(to_be_deleted)} documents to be deleted {list(to_be_deleted.values())}\n"
    f"  - {len(to_be_added)} new documents to be added {list(to_be_added.values())}\n"
    f"  - {len(to_be_updated)} documents to be updated {list(to_be_updated.values())}")
  to_be_deleted.update(to_be_updated)
  to_be_added.update(to_be_updated)
  if to_be_deleted:
    logging.info(f"Going to delete documents from index {list(to_be_deleted.keys())}.")
    index_manifest = delete_documents_from_index(
      to_be_deleted=to_be_deleted,
      pinecone_api_key=pinecone_api_key,
      pinecone_index_name=pinecone_index_name,
      existing_index_manifest=index_manifest)
    logging.info(f"Successfully deleted documents from index {list(to_be_deleted.keys())}.")
  if to_be_added:
    logging.info(f"Going to add new documents to index {list(to_be_added.keys())}.")
    index_manifest = add_documents_to_index(
      documents=list(to_be_added.values()),
      drive_service=drive_service,
      pinecone_api_key=pinecone_api_key,
      pinecone_index_name=pinecone_index_name,
      existing_index_manifest=index_manifest)
    logging.info(f"Successfully added new documents to index {list(to_be_added.keys())}.")
  if set(index_manifest.keys()).symmetric_difference(new_manifest.keys()):
    raise ValueError(
      f"After updating, the index manifest is not as expected.\n"
      f"Expected documents {new_manifest.keys()};\n"
      f"Actual documents {index_manifest.keys()}.")
  if (to_be_deleted or to_be_added) and write_index_manifest_after_update:
    write_index_manifest(
      index_manifest=index_manifest,
      pinecone_api_key=pinecone_api_key,
      pinecone_index_name=pinecone_index_name)
  return list(set(to_be_deleted.keys()).union(to_be_added.keys()))
  

if __name__ == "__main__":
    print(f"Please run the notebook, which imports this module.")
