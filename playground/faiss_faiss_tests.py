
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
print(db.index.ntotal)

loader = PyPDFLoader(file_path="./playground/schankerlaubnis.pdf")
documents = loader.load()
#text = pages[0].page_content
#db = FAISS.from_documents(docs, embeddings)
db.add_documents(documents=documents)


