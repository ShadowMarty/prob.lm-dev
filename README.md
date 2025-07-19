## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd tilde4-problm
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - On Windows:
     ```
     venv/Scripts/activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application**
   ```bash
   python main.py
   ```

2. **Follow the on-screen prompts to**
   - Input the path to your PDF document
   - Enter your search query
   - View and analyze the retrieved document chunks

## Project Structure

- `main.py`: Main application script containing the document processing and retrieval logic
- `requirements.txt`: Lists all Python dependencies
- `retrieved_chunks.csv`: Output file containing search results (created after first search)
- `problm_chroma_db/`: Directory containing the vector store (created after first run)

## Dependencies

- `langchain-community`: Core LangChain components
- `langchain`: Framework for developing applications with LLMs
- `langchain-chroma`: ChromaDB integration for LangChain
- `langchain-huggingface`: HuggingFace models integration
- `sentence-transformers`: For generating document embeddings
- `pypdf`: PDF processing library
- `chromadb`: Vector database for efficient similarity search
- `huggingface-hub`: For downloading HuggingFace models

## How It Works

1. **Document Processing**
   - Loads PDF documents using PyPDF
   - Splits documents into manageable chunks with overlap
   - Generates vector embeddings for each chunk

2. **Vector Storage**
   - Stores document chunks and their embeddings in ChromaDB
   - Persists the vector store for future use

3. **Query Processing**
   - Accepts natural language queries
   - Retrieves the most relevant document chunks based on semantic similarity
   - Displays results and optionally saves them to a CSV file

