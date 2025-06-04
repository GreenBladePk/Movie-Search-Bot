# Movie Search System

A powerful movie search and recommendation system built with LangChain, Chroma DB, and Chainlit. This system allows users to search and get information about movies using natural language queries, with support for filtering by year, genre, and ratings.

## Features

- Natural language movie search and recommendations
- Filter movies by year, genre, and ratings
- Interactive chat interface
- Vector-based semantic search
- Support for complex queries and conversations
- Persistent vector database for efficient retrieval

## Prerequisites

- Python 3.8 or higher
- IMDB dataset files (downloaded from [IMDB Datasets](https://datasets.imdbws.com/))
- Groq API key
- HuggingFace API key

## Dataset Setup

1. Download the following TSV files from [IMDB Datasets](https://datasets.imdbws.com/):
   - `name.basics.tsv` - Contains basic information about people
   - `title.akas.tsv` - Contains alternative titles for movies
   - `title.basics.tsv` - Contains basic information about titles
   - `title.crew.tsv` - Contains director and writer information
   - `title.principals.tsv` - Contains principal cast/crew information
   - `title.ratings.tsv` - Contains rating and vote information

2. Place all downloaded TSV files in the `Dataset` directory of the project.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/GreenBladePk/Movie-Search-Bot.git
cd Movie-Search-Bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

## Project Structure

- `ingest_data.py`: Script for processing and merging IMDB datasets,creating and managing the vector database
- `app.py`: Main application file with the Chainlit interface
- `chroma_db/`: Directory for the persistent vector database
- `Dataset/`: Directory for IMDB dataset files

## Setup and Usage

1. Process the IMDB dataset and Create the vector database:
```bash
python ingest_data.py
```

3. Start the chat interface:
```bash
chainlit run app.py
```

4. Open your browser and navigate to `http://localhost:8000`

## Usage Examples

You can interact with the system using natural language queries like:

- "What are some good sci-fi movies from the 1980s?"
- "Show me action movies with rating above 8"
- "What are the best comedy movies from 2010?"
- "Find me horror movies from the 90s"

The system supports various filters:
- Year (e.g., "movies from 1995")
- Genre (e.g., "action", "comedy", "drama", "sci-fi", "thriller", "romance", "horror", "animation", "documentary")
- Rating (e.g., "movies with rating above 7.5")

## Technical Details

The system uses:
- LangChain for the LLM integration
- Chroma DB for vector storage
- HuggingFace embeddings for semantic search
- Chainlit for the chat interface
- Groq's LLama model for natural language processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Docker Setup

If you want to run the application using Docker:

1. Make sure you have Docker installed on your system.

2. Build the Docker image:
```bash
docker build -t movie-search-bot .
```

3. Run the container:
```bash
docker run -p 8000:8000 --env-file .env movie-search-bot
```

4. Access the application at `http://localhost:8000`

Note: Make sure your `.env` file contains the required API keys before running the container.


