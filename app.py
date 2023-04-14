import os

import lib_book_parse
import lib_llm


## for vector store
from langchain.vectorstores import ElasticVectorSearch

## for embeddings
from langchain.embeddings import HuggingFaceEmbeddings




config = {
    "bookName" : "The Lord of The Rings",
    "bookIndexName": "book_lotr_embeddings",
    "bookFilePath": "./data/lotr-utf8.txt"
}


# Huggingface embedding setup
print(">> Prep. Huggingface embedding setup")
model_name = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEmbeddings(model_name=model_name)

# Elasticsearch URL setup
print(">> Prep. Elasticsearch config setup")
endpoint = os.getenv('ES_SERVER', 'ERROR') 
username = os.getenv('ES_USERNAME', 'ERROR') 
password = os.getenv('ES_PASSWORD', 'ERROR')
index_name = config['bookIndexName']
url = f"https://{username}:{password}@{endpoint}:443"

db = ElasticVectorSearch(embedding=hf, elasticsearch_url=url, index_name=index_name)

llm_chain_informed= lib_llm.make_the_llm()

lib_book_parse.loadBook(config['bookFilePath'],url,db,index_name)



def ask_a_question(question):
    print("The Question at hand: "+question)

    ## 3. get the relevant chunk from Elasticsearch for a question
    print(">> 3. get the relevant chunk from Elasticsearch for a question")
    similar_docs = db.similarity_search(question)
    print(similar_docs[0].page_content)

    ## 4. Ask Local LLM context informed prompt
    # print(">> 4. Asking The Book ... and its response is: ")
    informed_context= similar_docs[0].page_content
    response = llm_chain_informed.run(context=informed_context,question=question)
    return response



# Listen for commands
bookName = config['bookName']
print(f'I am the book, "{bookName}", ask me any question: ')

while True:
    command = input("User Question>> ")
    response = ask_a_question(command)
    print(f"\n\n I think the answer is : {response}\n")

