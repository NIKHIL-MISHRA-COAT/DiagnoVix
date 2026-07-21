import os
import asyncio
from dotenv import load_dotenv

# LangChain & HuggingFace Imports
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# 1. Load environment variables
# Path-specific loading to ensure it finds .env on Windows systems
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'), verbose=True)

# 🔥 Get credentials from .env
# Note: Using the exact name you created on the HuggingFace website
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HUGGINGFACE_REPO_ID = os.getenv("HUGGINGFACE_REPO_ID") or "mistralai/Mistral-7B-Instruct-v0.3"

# 2. Setup LLM (Mistral via HuggingFace Hub)
def load_llm(repo_id):
    if not HF_TOKEN:
        print("❌ Error: HUGGINGFACEHUB_API_TOKEN not found in .env file.")
        return None

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        # Updated parameter name for newer langchain-huggingface versions
        max_new_tokens=512, 
        timeout=300
    )
    return llm

# 3. Define the Medical Expert Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's medical question.
If you don't know the answer based on the context, just say that you don't know. 
Do not try to make up an answer. Keep the response grounded in the provided data.

Context: {context}
Question: {question}

Helpful Answer:
"""

def set_custom_prompt(template):
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

# 4. Main Execution Logic
def main():
    print("🔄 Initializing Medical RAG System...")
    
    # Path to your vector database
    DB_FAISS_PATH = "vectorstore/db_faiss"

    # Initialize Embedding Model (Must match create_memory_for_llm.py)
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load Database with Pydantic v2 compatibility
    try:
        db = FAISS.load_local(
            DB_FAISS_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        print("✅ Vector Database loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load FAISS: {e}")
        return

    # Load the LLM
    llm = load_llm(HUGGINGFACE_REPO_ID)
    if not llm:
        return

    # Create the RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )

    # 5. Interactive Query Loop
    print("\n--- DiagnoVix Backend Test Active ---")
    user_query = input("💬 What is your medical query? ")
    
    if user_query:
        print("🤖 Mistral is thinking...")
        try:
            response = qa_chain.invoke({'query': user_query})
            
            print("\n" + "="*30)
            print("📝 RESULT:", response["result"])
            print("="*30)
            
            # Print the source (which page of the PDF it read)
            if response["source_documents"]:
                print("\n📖 SOURCE DATA:")
                for doc in response["source_documents"]:
                    print(f"- From: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})")
        except Exception as e:
            print(f"❌ Error during inference: {e}")

if __name__ == "__main__":
    main()