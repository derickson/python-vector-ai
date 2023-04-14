import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

## for vector store
from langchain.vectorstores import ElasticVectorSearch
from elasticsearch import Elasticsearch

## for embeddings
from langchain.embeddings import HuggingFaceEmbeddings

## for conversation LLM
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM


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
es = Elasticsearch([url], verify_certs=True)
db = ElasticVectorSearch(embedding=hf, elasticsearch_url=url, index_name=index_name)

# Get Offline flan-t5-large ready to go, in CPU mode
print(">> Prep. Get Offline flan-t5-large ready to go, in CPU mode")
model_id = 'google/flan-t5-large'# go for a smaller model if you dont have the VRAM
tokenizer = AutoTokenizer.from_pretrained(model_id) 
model = AutoModelForSeq2SeqLM.from_pretrained(model_id) #load_in_8bit=True, device_map='auto'
pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=100
)
local_llm = HuggingFacePipeline(pipeline=pipe)
template_informed = """
I know the following: {context}
Question: {question}
Answer: """
prompt_informed = PromptTemplate(template=template_informed, input_variables=["context", "question"])
llm_chain_informed= LLMChain(prompt=prompt_informed, llm=local_llm)




## Parse the book if necessary
if not es.indices.exists(index=index_name):
    print(f'\tThe index: {index_name} does not exist')
    print(">> 1. Chunk up the Source document")
    loader = TextLoader(config['bookFilePath'])
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

