
from dotenv import load_dotenv


from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader



load_dotenv()
embeddings = OpenAIEmbeddings()


loader = TextLoader("./playground/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

db = FAISS.from_documents(docs, embeddings)
#print(db.index.ntotal)

loader = PyPDFLoader(file_path="./playground/schankerlaubnis.pdf")
documents = loader.load()
#text = pages[0].page_content
#db = FAISS.from_documents(docs, embeddings)
db.add_documents(documents=documents)


# faiss_index = 10

# # Get the ID of the document in the docstore
# doc_id = db.index_to_docstore_id[faiss_index]

# # Retrieve the document from the docstore
# document = db.docstore.search(doc_id)

# Now you have the document
# print(document)



retriever = db.as_retriever(search_type="mmr")
query = "ich will alkohol trinken."

# docs = retriever.get_relevant_documents(query)
# print(len(docs))
# for doc in docs:
#     print(f"content={doc.page_content}")
#     #print(f"content={doc.}")
#     print()
#     print(f"metadata={doc.metadata}") #doc.metadata)
#     #print(doc)

docs_and_scores = db.similarity_search_with_score(query)
print(len(docs_and_scores))
for doc in docs_and_scores:
    print(doc)

