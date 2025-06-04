import os
import sys
import logging
from typing import Optional, List

import pandas as pd
import torch
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Constants ---
DATASET_PATH = os.path.join("E:", os.sep, "Movie-Search-System", "imdb_dataset_final.csv")
CHROMA_DB_DIR = "chroma_db"

# --- Environment & Model Initialization ---
def initialize_environment() -> None:
    """Load environment variables and initialize models."""
    logger.info("Loading environment variables...")
    load_dotenv()

def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize and return the HuggingFace embeddings model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Initializing HuggingFace embeddings on device: {device}")
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )

# --- Data Preparation ---
def load_imdb_dataset(path: str) -> Optional[pd.DataFrame]:
    """Load the IMDb dataset CSV."""
    try:
        logger.info(f"Loading IMDb dataset from {path} ...")
        df = pd.read_csv(path, dtype=str)
        logger.info(f"Loaded {len(df)} rows from IMDb dataset.")
        return df
    except Exception as e:
        logger.error(f"Failed to load IMDb dataset: {e}")
        return None

def filter_and_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for movies and drop incomplete rows."""
    logger.info("Filtering for movies and dropping incomplete rows...")
    # Keep only rows where titleType is 'movie'
    df = df[df['titleType'] == 'movie']
    # Drop rows with missing essential fields
    df = df.dropna(subset=['primaryTitle', 'genres', 'averageRating'])
    logger.info(f"Filtered dataset contains {len(df)} movies.")
    return df

def create_documents_from_imdb(df: pd.DataFrame) -> List[Document]:
    """Convert IMDb DataFrame rows to LangChain Documents."""
    logger.info("Converting DataFrame rows to LangChain Documents...")

    def create_doc(row):
        # Build a text block for each movie, including all relevant fields
        return (
            f"Title: {row['primaryTitle']}\n"
            f"Original Title: {row['originalTitle']}\n"
            f"Year: {row['startYear']}\n"
            f"Genres: {row['genres']}\n"
            f"Rating: {row['averageRating']} ({row['numVotes']} votes)\n"
            f"Directors: {row.get('director_names', row.get('directors', 'N/A'))}\n"
            f"Writers: {row.get('writer_names', row.get('writers', 'N/A'))}\n"
            f"Region: {row.get('region', 'N/A')} | Language: {row.get('language', 'N/A')}\n"
            f"Alternative Title: {row.get('altTitle', row.get('title', 'N/A'))}\n"
        )

    # Apply the create_doc function to each row to generate the document text
    df['document'] = df.apply(create_doc, axis=1)
    # Create LangChain Document objects with metadata for each row
    docs = [
        Document(
            page_content=row['document'],
            metadata={
                "tconst": row["tconst"],
                "title": row["primaryTitle"],
                "genres": row["genres"],
                "year": row["startYear"],
                "rating": row["averageRating"],
                "votes": row["numVotes"],
            }
        )
        for _, row in df.iterrows()
    ]
    logger.info(f"Created {len(docs)} documents.")
    return docs

# --- Vector Store Operations ---
def load_existing_vectorstore(embeddings: HuggingFaceEmbeddings) -> Optional[Chroma]:
    """Attempt to load an existing Chroma vector store."""
    try:
        logger.info("Checking for existing Chroma DB...")
        # Try to load the persisted Chroma vector store
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
        count = vector_store._collection.count()
        logger.info(f"Loaded existing vector store with {count} documents.")
        return vector_store
    except Exception as e:
        logger.warning(f"Failed to load existing vector store: {e}")
        return None

def create_vectorstore_from_imdb(
    df: pd.DataFrame,
    embeddings: HuggingFaceEmbeddings
) -> Optional[Chroma]:
    """Create a new Chroma vector store from IMDb data."""
    try:
        # Convert DataFrame rows to LangChain Documents
        documents = create_documents_from_imdb(df)
        logger.info(f"Inserting {len(documents)} documents into Chroma DB...")
        # Create and persist the Chroma vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        logger.info(f"Created Chroma vector store with {vector_store._collection.count()} documents.")
        logger.info("Chroma vector store created and persisted successfully.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to create Chroma vector store: {e}")
        return None

def query_embedding(vector_store: Chroma, query: str = "What are some good sci-fi movies from the 1980s?", k: int = 4) -> None:
    """Query the vector store and print top results."""
    logger.info(f"Querying vector store: {query}")
    try:
        # Perform a similarity search in the vector store
        results = vector_store.similarity_search(query, k=k)
        logger.info("Top results:")
        print("-" * 40)
        for i, doc in enumerate(results, 1):
            print(f"[Result {i}]")
            print(doc.page_content)
            print("-" * 30)
    except Exception as e:
        logger.error(f"Failed to query vector store: {e}")

# --- Main Entrypoint ---
def main():
    # Load environment variables (e.g., API keys)
    initialize_environment()
    # Initialize the embeddings model
    embeddings = get_embeddings()

    # Try to load an existing vector store, if available
    vector_store = load_existing_vectorstore(embeddings)
    if vector_store and vector_store._collection.count() > 0:
        logger.info("Using existing vector store.")
        query_embedding(vector_store)
        return

    # Load the IMDb dataset from CSV
    df = load_imdb_dataset(DATASET_PATH)
    if df is None:
        logger.critical("IMDb dataset could not be loaded. Exiting.")
        sys.exit(1)

    # Filter and prepare the DataFrame for vectorization
    df = filter_and_prepare_dataframe(df)
    # Create a new vector store from the IMDb data
    vector_store = create_vectorstore_from_imdb(df, embeddings)
    if vector_store:
        query_embedding(vector_store)
    else:
        logger.critical("Failed to create vector store. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()
