from langchain_community.vectorstores import Chroma
import os

from langchain_community.document_loaders import PDFMinerLoader

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

openai_api_key = os.environ["OPENAI_API_KEY"]

loader = PDFMinerLoader("examples/2310.03714.pdf")

data_miner = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size=100,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False,
# )
text_splitter = SemanticChunker(OpenAIEmbeddings())

splited_text_semantic = text_splitter.split_documents(data_miner)
