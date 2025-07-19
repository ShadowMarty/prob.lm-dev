import os
import csv 

# Document Loading and Splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Stores and Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Database Path
DB_PATH = "problm_chroma_db" 

# Output CSV Path 
OUTPUT_CSV_PATH = "retrieved_chunks.csv"

def process_single_document(file_path):
    """Loads and splits a single PDF document."""
    if not os.path.isfile(file_path) or not file_path.endswith(".pdf"):
        print("Error: Please provide a valid path to the PDF file.")
        return None

    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    except Exception as e:
        print(f"Error loading PDF: {e}. Make sure the file exists and is a valid PDF.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)
    
    print(f"Successfully processed 1 PDF file, created {len(split_docs)} chunks.")
    return split_docs

def create_vector_store_and_retriever(split_docs):
    """Creates a vector store and returns its retriever for similarity search."""
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("Creating Chroma vector store for semantic search...")
    
    # Check if DB_PATH exists and contains data, if so, load it to save time
    if os.path.exists(DB_PATH) and os.path.isdir(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
        print(f"Loading existing Chroma DB from {DB_PATH}...")
        try:
            vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
            # Test loading by doing a similarity search
            test_query_result = vector_store.similarity_search("test", k=1) 
            if test_query_result:
                print("Existing vector store loaded successfully.")
            else:
                print("Existing vector store found but appears empty or corrupted. Recreating...")
                vector_store = Chroma.from_documents(documents=split_docs, embedding=embeddings, persist_directory=DB_PATH)
        except Exception as e:
            print(f"Error loading existing Chroma DB: {e}. Recreating...")
            vector_store = Chroma.from_documents(documents=split_docs, embedding=embeddings, persist_directory=DB_PATH)
    else:
        # If not, create and persist a new one
        vector_store = Chroma.from_documents(documents=split_docs, embedding=embeddings, persist_directory=DB_PATH)
        print("New vector store created and persisted.")

    # Retrieve relevant chunks using similarity search
    retriever = vector_store.as_retriever(search_kwargs={"k": 8}) # Retrieve 8 relevant documents
    print("Vector Retriever created for similarity search.")
    return retriever

def save_retrieved_docs_to_csv(retrieved_docs, query):
    """Saves the content and metadata of retrieved documents to a CSV file."""
    if not retrieved_docs:
        print("No documents were retrieved to save.")
        return

    # Define CSV header
    csv_headers = ["Query", "Retrieved Content", "Source File", "Page Number"]

    # Prepare rows for CSV
    csv_rows = []
    for doc in retrieved_docs:
        source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
        page_num = doc.metadata.get('page', -1)
        # Add 1 to page number for 1-based indexing display
        page_display = f"Page {page_num + 1}" if page_num != -1 else "N/A"
        
        csv_rows.append({
            "Query": query,
            "Retrieved Content": doc.page_content,
            "Source File": source_name,
            "Page Number": page_display
        })

    try:
        # Write to CSV file
        with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nSuccessfully saved {len(retrieved_docs)} retrieved documents to '{OUTPUT_CSV_PATH}'")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def main():
    """Main function for demonstrating document retrieval."""
    print("------------------------------")
    print("✨ Prob.lm Study Assistant ✨")
    print("------------------------------")

    # Document Loading 
    while True:
        pdf_path = input("\nEnter full path to the PDF file (C:/docs/notes.pdf): ")
        if not pdf_path:
            print("No file provided. Exiting.")
            return
        if os.path.exists(pdf_path) and os.path.isfile(pdf_path) and pdf_path.endswith(".pdf"):
            break
        else:
            print("Invalid path or not a PDF file. Please try again.")

    # Pipeline Setup 
    split_docs = process_single_document(pdf_path)
    if not split_docs:
        print("Failed to process document. Exiting.")
        return
    
    retriever = create_vector_store_and_retriever(split_docs)

    # Interactive Query Loop for Retrieval 
    print("\nReady to retrieve documents! Type 'quit' or 'exit' to stop.")
    while True:
        try:
            query = input("\nEnter your query to retrieve relevant content > ")
            if query.lower() in ['quit', 'exit']:
                break
            if not query.strip():
                continue
            
            print(f"\nPerforming similarity search for query: '{query}'...")
            retrieved_docs = retriever.invoke(query) # Use invoke directly on retriever
            
            if retrieved_docs:
                print(f"Retrieved {len(retrieved_docs)} documents.")
                save_retrieved_docs_to_csv(retrieved_docs, query)
            else:
                print("No documents retrieved for this query.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An unexpected error occurred during the query loop: {e}")

    print("\nThank You!")

if __name__ == "__main__":
    main()