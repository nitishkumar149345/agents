from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
import os

class RagAgent():
    def __init__(self, files: list):
        self.files = files
        self.retriever = None

    def loader(self):
        for file in self.files:
            extension = file.split('.')[-1]
            if extension =='txt':
                loader = TextLoader(file)
                self.docs = loader.load()
            else:
                loader = PyPDFLoader(file)
                self.docs = loader.load_and_split()

    def split_docs(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size= 200,
            chunk_overlap=20,
        )

        self.chunks = splitter.split_documents(self.docs)


    def embeddings_vectordb(self):
        vectordb = FAISS.from_documents(self.chunks, OpenAIEmbeddings())
        self.retriever = vectordb.as_retriever()
       

    def create_chain(self):
        if not self.retriever:
            self.loader()
            self.split_docs()
            self.embeddings_vectordb()
        
        prompt = ChatPromptTemplate.from_messages([
            ('system','''
                        You are an assistant for question-answering tasks.
                        Use the following pieces of retrieved context to answer the questions
                        {context}
                        '''),
            ('user','{input}'),
        ])

        llm = ChatOpenAI()
        doc_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(self.retriever, doc_chain)
        return retrieval_chain
    
# files = ['/Users/omniadmin/Desktop/python-projects/agents/rag_files/Nitishkumar_resume.pdf']
# obj = RagAgent(files)
# chain = obj.create_chain()
# r= chain.invoke({"input":'what is this file about'})
# print (r['answer'])
