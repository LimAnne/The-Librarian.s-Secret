import fitz  # PyMuPDF for extracting text
from langchain_community.vectorstores import FAISS
import numpy as np
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
import os
import yaml
import re
from prompts.prompt_template import PROMPT
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp

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

# === STEP 3: Create Embeddings and Store in FAISS-GPU ===
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

    ### test if vector DB was loaded correctly
    results = vectordb.similarity_search('creatures', top_k=3)
    print("🔍 Top 3 Most Similar Results:")
    for i, doc in enumerate(results):
        cleaned_text = clean_text(doc.page_content)
        print(f"Result {i+1}: {cleaned_text[:200]}...")

    return vectordb

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = text.strip()  # Remove leading/trailing spaces
    return text

# # === STEP 4: Search in FAISS-GPU ===
# def search_faiss_gpu(query, index, embeddings, chunks, model, top_k=3):
#     query_embedding = np.array([model.encode(query)], dtype="float32")
#     distances, indices = index.search(query_embedding, top_k)
#     return [chunks[i] for i in indices[0]]

def retrieval_chain(vectordb):
    llm = LlamaCpp(
        model_path="./download/mistral-7b-instruct.Q4_K_M.gguf",  
        n_gpu_layers=35,  # Optimize for RTX 3050
        n_batch=512,  # Adjust for performance
        temperature=0.1,  # Lower temp for better factual responses
        max_tokens=512,
        verbose=True  # Set to False in production
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 5, "search_type": "similarity"})
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True,
    verbose=False
)
    
    question = "What does Rincewind like to do when faced with danger?"
    # Perform retrieval and generate the answer
    result = qa_chain.invoke({"query": question})

    # Print the response
    print("\n🤖 Answer:", result["result"])
    
    # If you want to see retrieved documents
    print("\n📚 Retrieved Documents:")
    for doc in result["source_documents"]:
        print("-" * 40)
        print(doc.page_content)

    return result

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

    

