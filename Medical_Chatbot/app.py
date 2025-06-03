import os
from flask import Flask, render_template, request, session, redirect, url_for
from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_community.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone as LangchainPinecone

# Load environment variables
load_dotenv()

# Get environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = LangchainPinecone.from_existing_index(
    index_name=PINECONE_INDEX, embedding=embeddings
)

# Define custom prompt
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

# Set up prompt and retrieval chain
prompt = PromptTemplate(
    input_variables=["context", "question"], template=CUSTOM_PROMPT_TEMPLATE
)

llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    model="llama3-70b-8192",
    openai_api_key=GROQ_API_KEY,
)

retriever = vectorstore.as_retriever()
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)


# Routes
@app.route("/", methods=["GET"])
def home():
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template("index.html", chat_history=session["chat_history"])


@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form["user_input"]

    if "chat_history" not in session:
        session["chat_history"] = []

    try:
        result = retrieval_chain.invoke({"query": user_input})
        bot_response = result["result"]

        # Save in chat history
        session["chat_history"].append({"user": user_input, "bot": bot_response})
        session.modified = True

        return redirect(url_for("home"))

    except Exception as e:
        session["chat_history"].append({"user": user_input, "bot": f"Error: {str(e)}"})
        session.modified = True
        return redirect(url_for("home"))


@app.route("/clear", methods=["GET"])
def clear():
    session.pop("chat_history", None)
    return redirect(url_for("home"))


# Run app
if __name__ == "__main__":
    app.run(debug=False)
