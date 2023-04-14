
import lib_book_parse
import lib_llm
import lib_embeddings
import lib_vectordb

import lib_speak



config = {
    "bookName" : "The Lord of The Rings",
    "bookIndexName": "book_lotr_embeddings",
    "bookFilePath": "./data/lotr-utf8.txt"
}

bookName = config['bookName']
bookFilePath = config['bookFilePath']
index_name = config['bookIndexName']

# Huggingface embedding setup
hf = lib_embeddings.setup_embeddings()

## Elasticsearch as a vector db
db, url = lib_vectordb.setup_vectordb(hf,index_name)

## set up the conversational LLM
llm_chain_informed= lib_llm.make_the_llm()

## Load the book
lib_book_parse.loadBook(bookFilePath,url,hf,db,index_name)

## how to ask a question
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


# The conversational loop

lib_speak.speak(f'I am the book, "{bookName}", ask me any question: ')

while True:
    command = input("User Question>> ")
    response = ask_a_question(command)
    lib_speak.speak(f"\n\n I think the answer is : {response}\n")

