import os
from typing import List, Tuple
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents import Document

class DocumentsStore:
    """Documents Store that uses the embeddings provided"""
    def __init__(self, embeddings:Embeddings, documents_dir_path:str):
        self._embeddings = embeddings
        self._documents_dir_path = documents_dir_path
        self._db: FAISS| None = None

    def _init_db(self)->FAISS:
        """Initializes the db"""
        
        documents: list[Document] = []
        
        # iterating over directory and subdirectory to get desired result
        for path, _, files in os.walk(self._documents_dir_path):
            for name in files:
                if name.endswith('.pdf'):
                    loader = PyPDFLoader(file_path=os.path.join(path, name))
                    # db.add_documents(documents=loader.load())
                    documents += loader.load()
        db = FAISS.from_documents(documents, self._embeddings)
        return db
    @property
    def db(self)->FAISS:
        """Returns the db"""
        if self._db is None:
            self._db = self._init_db()
        return self._db
    
    def retrieve(self, query:str, top_k:int=10)-> dict[str, list[str]|None]:
        """Retrieves documents from the db"""
        docs_and_scores: List[Tuple[Document, float]] = self.db.similarity_search_with_score(query, k=top_k)
        result: dict[str, list[str]|None] = {}
        for doc in docs_and_scores:
            fu, bar = doc
            doc_path = fu.metadata["source"]
            result_value: list[str]|None = result.get(doc_path, None)
            if result_value is None:
                result[doc_path] = [fu.page_content]
            else:
                result_value.append(fu.page_content)
        
        return result
