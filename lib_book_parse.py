
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

from elasticsearch import Elasticsearch


## load book utility
## params
##  filepath: where to get the book txt ... should be utf-8
##  url: the full Elasticsearch url with username password and port embedded
##  hf: hugging face transformer for sentences
##  db: the VectorStore Langcahin object ready to go with embedding thing already set up
##  index_name: name of index to use in ES
##
##  will check if the index_name exists already in ES url before attempting split and load
def loadBook(filepath, url, hf, db, index_name):

    with Elasticsearch([url], verify_certs=True) as es:
        ## Parse the book if necessary
        if not es.indices.exists(index=index_name):
            print(f'\tThe index: {index_name} does not exist')
            print(">> 1. Chunk up the Source document")
            loader = TextLoader(filepath)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
            docs = text_splitter.split_documents(documents)
            # for doc in docs:
            #     content = doc.page_content
            #     mylen = len(content)
            #     print(mylen)
            #     print(content)
            print(">> 2. Index the chunks into Elasticsearch")
            db.from_documents(docs, embedding=hf, elasticsearch_url=url, index_name=index_name)
        else:
            print("\tLooks like the book is already loaded, let's move on")
