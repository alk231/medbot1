import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone

# Load env variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)


# 1. Load PDF documents
def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


documents = load_pdf_files("data/")
print(f"📄 Total documents loaded: {len(documents)}")


# 2. Split documents
def create_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)


chunks = create_chunks(documents)
print(f"✂️ Total chunks created: {len(chunks)}")

# 3. Embedding model (384 dimensions)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Upload chunks to Pinecone
vectorstore = LangchainPinecone.from_documents(
    documents=chunks, embedding=embeddings, index_name=PINECONE_INDEX
)

print("🚀 Embeddings successfully uploaded to Pinecone.")
