import fitz  # PyMuPDF for extracting text
from langchain_community.vectorstores import FAISS
import numpy as np
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from sentence_transformers import SentenceTransformer
# import openai
# from dotenv import load_dotenv
import os
import yaml
import re
from utils.prompt_template import PROMPT
from utils.main_utils import process_llm_response
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
# from langchain_community.cache import SQLiteCache

# === STEP 1: Extract Text from PDFs ===
def extract_text_from_pdfs(pdf_directory):
    all_docs = []
    
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"): 
            pdf_path = os.path.join(pdf_directory, filename)
            print(f"Extracting text from: {filename}")

            # Load PDF and extract text
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            all_docs.extend(docs)  # Append extracted documents

    print(f"Total documents: {len(all_docs)}")

    print(f"Total characters: {len(all_docs[0].page_content)}")

    return all_docs

# === STEP 2: Chunk Text for Better Retrieval ===
def chunk_text(all_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    print(f"Total document chunks: {len(chunks) if chunks else 0}")
    if chunks:
        print(f"First chunk preview: {chunks[0].page_content[:200]}")
    else:
        print("No text extracted. Check PDF processing.")

    return chunks

# === STEP 3: Create Embeddings and persist in vector store ===
def create_embeddings(chunks, embedding_model, device):

    print(f"Using device: {device}")
    embeddings = HuggingFaceInstructEmbeddings(
        model_name = embedding_model,
        model_kwargs = {"device": device}
    )

    ### create embeddings and DB
    vectordb = FAISS.from_documents(
        documents = chunks, 
        embedding = embeddings
    )

    ### persist vector database
    vectordb.save_local(f"{embeddings_path}/faiss_index")
    print("✅ FAISS index saved successfully!")

def load_embeddings(embedding_model, embeddings_path, device):
    print("Loading embeddings and FAISS index...")
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device}
    )
    # Load FAISS index from saved path
    vectordb = FAISS.load_local(embeddings_path + '/faiss_index', embeddings, allow_dangerous_deserialization=True)
    print("FAISS index loaded successfully!")

    search_type = 'creatures'
    ### test if vector DB was loaded correctly
    results = vectordb.similarity_search(search_type, top_k=3)
    print(f"🔍 Testing the vector source. Provide top 3 most similar results for: {search_type}")
    for i, doc in enumerate(results):
        cleaned_text = clean_text(doc.page_content)
        print(f"Result {i+1}: {cleaned_text[:200]}...")

    return vectordb

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = text.strip()  # Remove leading/trailing spaces
    return text

def retrieval_chain(vectordb):
# Load CUDA-compatible local LLM using ctransformers
    llm = CTransformers(
        model="C:\\Users\\Liman\\Downloads\\ask_terry\\.huggingface\\download\\mistral-7b-instruct-v0.2.Q4_K_M.gguf", 
        model_type="mistral",
        config={
            "gpu_layers": 20,      # gpu usage, adjust as needed
            "temperature": 0.1,      
            "max_new_tokens": 2000, # 
            "context_length": 2048, # 
            "batch_size": 16       # Adjust batch size to avoid memory issues
        }
    )
    question = "Describe Rincewind's Luggage in full detail, including its physical appearance, abilities, and behavior. Provide examples from different books."
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 5, "search_type": "similarity"})
    
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,  
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True,
    verbose=True
    )

    # # Enable caching
    # qa_chain.cache = SQLiteCache(database_path="./cache.db")
            
    # Retrieve context and generate the answer
    result = qa_chain.invoke({"query": question})
    ans = process_llm_response(result)

    # for source in result.get("source_documents", []):
    #     print(source.metadata)
    print(f"Query: {question}")
    print("\n🤖 Final Answer:", result["result"] + ans)

    # # Show retrieved documents for debugging
    # print("\n📚 Retrieved Documents:")
    # for doc in result["source_documents"]:
    #     print("-" * 40)
    #     print(doc.page_content[:500] + "...")  

    # compare answer with no RAG`
    print("\n🔍 Running **Without Retrieval (No-RAG)**...")

    prompt = f"Answer the following question as accurately as possible:\n\n{question}"
    no_rag_result = llm.invoke(prompt)

    print(f"\n🤖 No-RAG Answer: {no_rag_result}")

# === MAIN EXECUTION ===
if __name__ == "__main__":

    with open("./config/default.yaml", "r") as file:
        config = yaml.safe_load(file)

        pdf_path = config["pdf_dir"]

        embedding_model = config["embedding_model"]

        embeddings_path = config["embeddings_path"]

    # load_dotenv()
    # openai.api_key = os.getenv("OPENAI_API_KEY")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config["extract_text"]:
        print("Extracting text from PDFs...")
        documents = extract_text_from_pdfs(pdf_path)
    
    if config["chunk_text"]:
        print("Chunking text...")
        chunks = chunk_text(documents)

    if config["create_embeddings"]:
        print("Creating embeddings and storing in local folder...")
        create_embeddings(chunks, embedding_model, device)

    if config["load_embeddings"]:
        vectordb = load_embeddings(embedding_model, embeddings_path, device)

    print("Creating retrieval chain...")
    retrieval_chain(vectordb)
    


    

