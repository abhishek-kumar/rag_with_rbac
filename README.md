LLM applications typically use vector databases for RAG (retrieval augmented generation), but most vector databases today don't
natively support role based access controls (RBAC). They do support multi-tenancy and some access protections but these are not intended for the application layer, they are intended for service accounts and infrastructure layers. With role based access control (RBAC), change management is non-trivial - how do we keep the index up to date with files and metadata in storage (Google Drive or elsewhere)? This demo shows an implementation of how to do this. 

# RAG with RBAC.

## LLM App on Streamlit.
This is an LLM App using retrieval augmented generation (RAG) with role based access controls (RBAC) on your data.

## RAG Implementation.
The large language model being used here is OpenAI (default - GPT3.5).
The Vector DB used is Pinecone, vector size 1536 (corresponding to OpenAI embeddings).
Building the index is done via Google Drive API (see the `index_utils` module). Every pdf file in the root drive folder is scanned, split into pages, and each page is added to the index as a separate document. Each page is also chunked and indexed by its corresponding embedding vector.
The embedding model used for indexing documents (and queries) is OpenAI's `text-embedding-3-small`.

## ACL enforcement through role based access control.
The Pinecone index has been made ACL aware with the use of metadata and search filters.
The logged-in user is only able to query data that they have read permissions for.

## Change management - processing file and ACL changes.
Version 0.2 implements a lazy approach to applying real-time ACL changes on the data.
We don't update the index unless it is necessary to do so.
  1. When a user's query references source files that have been deleted or modified, we require an index update of those specific documents only before re-generating a response.
  2. When new files are added to Google Drive that the index is not aware of, we require an index update to add those files to the index prior to generating a response to the user's query. This is necessary because the new files might be relevant to the user's query and we have no way of knowing unless we update the index first.
  3. If files on Drive have been updated that have no bearing to the user's query or response, we do not require an index update which is an expensive process. An offline system could periodically process these updates - the scheduling interval can be configured to achieve the right cost vs accuracy trade-off.

Please see the demo video for more.

## Demo.

[![Demo video](https://img.youtube.com/vi/zKCrEXEBQGY/0.jpg)](https://www.youtube.com/watch?v=zKCrEXEBQGY)

  * [video link](https://www.youtube.com/watch?v=zKCrEXEBQGY)


## Try out the hosted app.
https://rag-rbac.streamlit.app/

