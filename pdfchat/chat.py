from langchain_community.vectorstores import Chroma
import chromadb
import os

from langchain_community.document_loaders import PDFMinerLoader

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI()

openai_api_key = os.environ["OPENAI_API_KEY"]

file = "examples/2310.03714.pdf"
vector_store_path = "./chromadb"

filename = os.path.basename(file)

loader = PDFMinerLoader(file)

data = loader.load()

text_splitter = SemanticChunker(OpenAIEmbeddings())

client = chromadb.PersistentClient(path=vector_store_path)

if filename not in client.list_collections():
    documents = text_splitter.split_documents(data)
    db = Chroma.from_documents(
        documents,
        OpenAIEmbeddings(),
        collection_name=filename,
        client=client,
    )
else:
    db = Chroma(collection_name=filename, client=client)

# query = "What universities are involved in this work?"

# docs = db.similarity_search(query)


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

retriever = db.as_retriever()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

output = chain.invoke("What is the title of this paper?")

print(output)
