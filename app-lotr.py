
import lib_book_parse
import lib_llm
import lib_embeddings
import lib_vectordb




config = {
    "bookName" : "The Lord of The Rings",
    "bookIndexName": "book_lotr_embeddings-big1024",
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
llm_chain_ignorant= lib_llm.make_the_llm_ignorant()

## Load the book
lib_book_parse.loadBookBig(bookFilePath,url,hf,db,index_name)

## how to ask a question
def ask_a_question(question):
    # print("The Question at hand: "+question)

    ## 3. get the relevant chunk from Elasticsearch for a question
    # print(">> 3. get the relevant chunk from Elasticsearch for a question")
    similar_docs = db.similarity_search(question)
    print(f'The most relevant passage: \n\t{similar_docs[0].page_content}')

    ## 4. Ask Local LLM context informed prompt
    # print(">> 4. Asking The Book ... and its response is: ")
    informed_context= similar_docs[0].page_content
    informed_response = llm_chain_informed.run(context=informed_context,question=question)
    
    ignorant_response = llm_chain_ignorant.run(question=question)
    return informed_response, ignorant_response


# The conversational loop

print(f'I am the book, "{bookName}", ask me any question: ')

while True:
    command = input("User Question>> ")
    response, response2 = ask_a_question(command)
    print(f"\n\n I think the answer is : {response}\n Without extra context I think the answer is: {response2}")

