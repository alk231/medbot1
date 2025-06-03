from langchain_community.chat_models import ChatOpenAI  # Correct import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Get the correct Groq API key from .env

# Set up LLM using Groq's base_url and the API key
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    model="llama3-70b-8192",
    openai_api_key=GROQ_API_KEY,  # Groq API key or a dummy OpenAI key
)

# Define prompt template
CUSTOM_PROMPT_TEMPLATE = """
Answer the user's question using only the information provided in the context.

- If the answer isn't in the context, respond with "I don't know."
- Do not make up information.
- Do not mention that you're using context or refer to any source.
- Keep the response clear, concise, and natural—like a helpful assistant.
- No greetings or filler text; just give the answer directly.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"], template=CUSTOM_PROMPT_TEMPLATE
)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = LangchainPinecone.from_existing_index(
    index_name=PINECONE_INDEX, embedding=embeddings
)

# Set up retriever and chain
retriever = vectorstore.as_retriever()
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

# Query example
user_query = input("Write Query Here: ")
response = retrieval_chain.invoke({"query": user_query})

print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
