import os
from flask import Flask, request, jsonify, render_template
from langchain_groq import ChatGroq, GroqEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone

# ----------------------------
# Flask App Setup
# ----------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback_secret")

# ----------------------------
# API Keys
# ----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ----------------------------
# Pinecone Setup
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "alok"   # your Pinecone index

# ----------------------------
# Groq Embeddings
# ----------------------------
embeddings = GroqEmbeddings(
    groq_api_key=GROQ_API_KEY,
    model="nomic-embed-text-v1.5"
)

# ----------------------------
# Groq LLM
# ----------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192"
)

# ----------------------------
# Vector Store
# ----------------------------
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        user_question = request.json.get("question")

        # Retrieve relevant docs from Pinecone
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(user_question)

        context = "\n".join([d.page_content for d in docs])

        # Run query through Groq LLM
        response = llm.predict(
            f"Answer the following based on context:\n\n{context}\n\nQuestion: {user_question}\nAnswer:"
        )

        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
