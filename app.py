import asyncio
import os
import json
import warnings
from dotenv import load_dotenv
import streamlit as st
import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import time
import nest_asyncio

# Allow nested event loops for compatibility with Streamlit and async libraries
nest_asyncio.apply()

# Suppress specific PyTorch warnings that are not relevant to app functionality
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'", module="torch")

# Load environment variables from .env file
load_dotenv()

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="🎬 Movie Recommendation Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: white;
        background-clip: text;
        color: white !important;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: black !important;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: black !important;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        color: black !important;
    }
    .filter-info {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: black !important;
    }
    .stAlert > div {
        border-radius: 10px;
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# Model for structured movie filters
class MovieFilters(BaseModel):
    language: Optional[str] = None
    director: Optional[str] = None
    year_exact: Optional[int] = None
    year_after: Optional[int] = None
    year_before: Optional[int] = None
    year_between: Optional[List[int]] = None
    genres: Optional[List[str]] = None
    rating_min: Optional[float] = None
    rating_max: Optional[float] = None
    actor: Optional[str] = None
    keywords: Optional[List[str]] = None
    is_greeting: bool = False
    is_continuation: bool = False

# Initialize Streamlit session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = {'filters': {}, 'history': []}
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'response_llm' not in st.session_state:
    st.session_state.response_llm = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

@st.cache_resource
def initialize_components():
    """Initialize and cache embeddings, LLMs, and vectorstore."""
    logger.info("Starting component initialization...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logger.info("Embeddings initialized.")

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        llm = ChatGroq(
            model="llama3-8b-8192",
            api_key=groq_api_key,
            temperature=0.1,
            max_tokens=512
        )
        logger.info("Filter extraction LLM initialized.")

        response_llm = ChatGroq(
            model="llama3-8b-8192",
            api_key=groq_api_key,
            temperature=0.7,
            max_tokens=512
        )
        logger.info("Response generation LLM initialized.")

        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
        logger.info("Vectorstore initialized.")

        collection_count = vectorstore._collection.count()
        logger.info(f"Chroma collection count: {collection_count}")

        logger.info("Component initialization successful.")
        return vectorstore, llm, response_llm, embeddings
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}", exc_info=True)
        st.error(f"Error initializing components: {str(e)}")
        return None, None, None, None

async def extract_filters_with_llm(query: str, conversation_context: List[str] = None) -> Dict[str, Any]:
    """Extract structured filters from user query using LLM, considering conversation context."""
    context_str = ""
    if conversation_context:
        context_str = f"\nPrevious conversation context:\n" + "\n".join(conversation_context[-6:])

    filter_extraction_prompt = ChatPromptTemplate.from_template("""
You are a movie database query parser. Extract structured filters from the user's natural language query, considering the previous conversation context if available.

IMPORTANT: Return ONLY a valid JSON object with the following structure. Do not include any other text or explanation. If a filter is not mentioned and not implied by context, set it to null. If the query is a continuation of a previous one, set "is_continuation" to true. If the query is a simple greeting, set "is_greeting" to true.

{{
    "language": null or "Tamil" or "English" or "Hindi" or "Telugu" etc.,
    "director": null or director name (e.g., "Shankar", "Christopher Nolan"),
    "year_exact": null or specific year as integer,
    "year_after": null or year as integer (for "after 2010", "since 2015"),
    "year_before": null or year as integer (for "before 2000", "until 2020"),
    "year_between": null or [start_year, end_year] as array (e.g., [2000, 2010] for "between 2000 and 2010"),
    "genres": null or array of genres like ["romance", "comedy", "action", "drama", "sci-fi", "thriller", "horror", "animation", "documentary", "adventure", "crime", "fantasy", "mystery", "war", "western", "musical", "biography", "history"],
    "rating_min": null or minimum rating as float (e.g., 7.0),
    "rating_max": null or maximum rating as float,
    "actor": null or actor name,
    "keywords": null or array of important keywords for search (excluding explicit filter values already captured, e.g., if director is captured, don't put director name in keywords),
    "is_greeting": false,
    "is_continuation": false
}}

Examples:
Query: "I want a tamil movie of director Shankar"
Output: {{"language": "Tamil", "director": "Shankar", "keywords": [], "is_greeting": false, "is_continuation": false}}

Query: "I am having a good mood today, can you recommend me some rom-com released after 2010"
Output: {{"genres": ["romance", "comedy"], "year_after": 2010, "keywords": ["good mood", "recommend"], "is_greeting": false, "is_continuation": false}}

Query: "hi"
Output: {{"language": null, "director": null, "year_exact": null, "year_after": null, "year_before": null, "year_between": null, "genres": null, "rating_min": null, "rating_max": null, "actor": null, "keywords": null, "is_greeting": true, "is_continuation": false}}

Query: "just english" (assuming previous query was about movies)
Output: {{"language": "English", "keywords": [], "is_greeting": false, "is_continuation": true}}

{context_str}

User Query: "{query}"

JSON Output:""")

    try:
        logger.info(f"Attempting to extract filters for query: {query}")
        response = await (filter_extraction_prompt | st.session_state.llm | StrOutputParser()).ainvoke({
            "query": query,
            "context_str": context_str
        })

        response = response.strip()
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end != -1 and json_end > json_start:
            response = response[json_start:json_end]
        else:
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]

        filters = json.loads(response.strip())

        # Clean and validate filters
        valid_filter_keys = MovieFilters.model_fields.keys()
        cleaned_filters = {}
        for key, value in filters.items():
            if key in valid_filter_keys and value is not None and value != "" and value != []:
                if key in ["year_exact", "year_after", "year_before"] and isinstance(value, (str, float)):
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert {key} value '{value}' to int. Skipping filter.")
                        continue
                elif key in ["rating_min", "rating_max"] and isinstance(value, str):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert {key} value '{value}' to float. Skipping filter.")
                        continue
                elif key in ["genres", "keywords"] and isinstance(value, str):
                    value = [v.strip() for v in value.split(',') if v.strip()]
                elif key == "year_between" and isinstance(value, list):
                    if len(value) == 2:
                        try:
                            value = [int(value[0]), int(value[1])]
                            if value[0] > value[1]:
                                value = [value[1], value[0]]
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert year_between values '{value}' to integers. Skipping filter.")
                            continue
                    else:
                        logger.warning(f"Invalid format for year_between: {value}. Skipping filter.")
                        continue
                cleaned_filters[key] = value

        logger.info(f"LLM extracted and cleaned filters: {cleaned_filters}")
        return cleaned_filters

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}, Response: {response}", exc_info=True)
        return {"keywords": query.lower().split(), "is_greeting": False, "is_continuation": False}
    except Exception as e:
        logger.error(f"Filter extraction error: {e}", exc_info=True)
        return {"keywords": query.lower().split(), "is_greeting": False, "is_continuation": False}

def convert_to_chroma_filters(llm_filters: Dict[str, Any]) -> Dict[str, Any]:
    """Convert LLM-extracted filters to Chroma-compatible format for metadata filtering."""
    where_filters = {}

    if "language" in llm_filters:
        where_filters["language"] = {"$contains": llm_filters['language']}
    if "director" in llm_filters:
        where_filters["director"] = {"$contains": llm_filters['director']}
    if "actor" in llm_filters:
        where_filters["$or"] = [
            {"actors": {"$contains": llm_filters['actor']}},
            {"cast": {"$contains": llm_filters['actor']}}
        ]
    if "year_exact" in llm_filters:
        where_filters["year"] = {"$eq": llm_filters["year_exact"]}
    elif "year_after" in llm_filters:
        where_filters["year"] = {"$gte": llm_filters["year_after"]}
    elif "year_before" in llm_filters:
        where_filters["year"] = {"$lte": llm_filters["year_before"]}
    elif "year_between" in llm_filters and len(llm_filters["year_between"]) == 2:
        start_year, end_year = sorted(llm_filters["year_between"])
        if "$and" not in where_filters: where_filters["$and"] = []
        where_filters["$and"].append({"year": {"$gte": start_year}})
        where_filters["$and"].append({"year": {"$lte": end_year}})
    if "genres" in llm_filters:
        genres = llm_filters["genres"]
        if len(genres) == 1:
            where_filters["genres"] = {"$contains": genres[0]}
        else:
            if "$or" not in where_filters: where_filters["$or"] = []
            genre_clauses = [{"genres": {"$contains": genre}} for genre in genres]
            if "$or" in where_filters:
                existing_or = where_filters.pop("$or")
                where_filters["$and"] = [{"$or": existing_or}, {"$or": genre_clauses}]
            else:
                where_filters["$or"] = genre_clauses
    if "rating_min" in llm_filters:
        if "$and" not in where_filters: where_filters["$and"] = []
        where_filters["$and"].append({"imdb_rating": {"$gte": llm_filters["rating_min"]}})
    if "rating_max" in llm_filters:
        if "$and" not in where_filters: where_filters["$and"] = []
        where_filters["$and"].append({"imdb_rating": {"$lte": llm_filters["rating_max"]}})

    # Combine $and/$or if both exist at the top level
    combined_filters = {}
    if "$and" in where_filters or "$or" in where_filters:
        combined_clauses = []
        if "$and" in where_filters:
            combined_clauses.extend(where_filters.pop("$and"))
        if "$or" in where_filters:
            combined_clauses.append({"$or": where_filters.pop("$or")})
        for key, value in where_filters.items():
            combined_clauses.append({key: value})
        if len(combined_clauses) == 1:
            combined_filters = combined_clauses[0]
        elif combined_clauses:
            combined_filters["$and"] = combined_clauses
    else:
        combined_filters = where_filters

    return combined_filters

async def enhanced_retrieval_with_llm(vectorstore, query: str, llm_filters: Dict[str, Any], k: int = 5):
    """Retrieve documents from vectorstore using LLM-extracted filters and enhanced query."""
    try:
        if llm_filters.get("is_greeting", False):
            logger.info("Skipping retrieval for greeting.")
            return []

        chroma_metadata_filters = convert_to_chroma_filters(llm_filters)
        search_query_parts = [query]
        if llm_filters.get("keywords"):
            search_query_parts.extend(llm_filters["keywords"])
        if llm_filters.get("language") and "language" not in chroma_metadata_filters:
            search_query_parts.append(llm_filters["language"])
        if llm_filters.get("director") and "director" not in chroma_metadata_filters:
            search_query_parts.append(llm_filters["director"])
        if llm_filters.get("actor"):
            search_query_parts.append(llm_filters["actor"])
        if llm_filters.get("genres") and "$or" not in chroma_metadata_filters.values() and not any(isinstance(val, dict) and "$or" in val for val in chroma_metadata_filters.values()):
            search_query_parts.extend(llm_filters["genres"])

        enhanced_query = " ".join(list(set(search_query_parts)))
        logger.info(f"Enhanced search query: {enhanced_query}")

        search_kwargs = {"k": k}
        if chroma_metadata_filters:
            logger.info(f"Applying Chroma metadata filters: {chroma_metadata_filters}")
            search_kwargs["where"] = chroma_metadata_filters

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

        docs = await retriever.ainvoke(enhanced_query)
        logger.info(f"Retrieved {len(docs)} documents.")
        return docs

    except Exception as e:
        logger.error(f"Enhanced retrieval error: {e}", exc_info=True)
        logger.warning("Falling back to basic similarity search due to error.")
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            return await retriever.ainvoke(query)
        except Exception as final_error:
            logger.error(f"Final fallback error: {final_error}", exc_info=True)
            return []

def format_docs(docs):
    """Format retrieved documents for display in the chat."""
    if not docs:
        return "No relevant movie information found in the database."

    formatted_docs = []
    for doc in docs:
        content = doc.page_content
        metadata = getattr(doc, 'metadata', {})
        details = []
        if 'title' in metadata and metadata['title']:
            details.append(f"Title: {metadata['title']}")
        if 'year' in metadata and metadata['year']:
            details.append(f"Year: {metadata['year']}")
        if 'director' in metadata and metadata['director']:
            details.append(f"Director: {metadata['director']}")
        if 'language' in metadata and metadata['language']:
            details.append(f"Language: {metadata['language']}")
        if 'genres' in metadata and metadata['genres']:
            genres_str = ", ".join(metadata['genres']) if isinstance(metadata['genres'], list) else metadata['genres']
            if genres_str:
                details.append(f"Genres: {genres_str}")
        if 'imdb_rating' in metadata and metadata['imdb_rating']:
            details.append(f"IMDb Rating: {metadata['imdb_rating']}")
        if 'actors' in metadata and metadata['actors']:
            actors_str = ", ".join(metadata['actors'][:3]) + ('...' if len(metadata['actors']) > 3 else '') if isinstance(metadata['actors'], list) else metadata['actors']
            if actors_str:
                details.append(f"Actors: {actors_str}")

        metadata_str = ", ".join(details) if details else "Details: N/A"
        formatted_docs.append(f"{content}\n{metadata_str}")

    return "\n\n---\n\n".join(formatted_docs)

def update_conversation_context(new_filters: Dict[str, Any]):
    """Update conversation context with new filters, handling continuation logic."""
    if new_filters.get("is_continuation", False):
        for key, value in new_filters.items():
            if key not in ["is_continuation", "is_greeting"] and value is not None:
                if key in ["genres", "keywords"] and isinstance(value, list) and isinstance(st.session_state.conversation_context['filters'].get(key), list):
                    st.session_state.conversation_context['filters'][key].extend(value)
                    st.session_state.conversation_context['filters'][key] = list(set(st.session_state.conversation_context['filters'][key]))
                else:
                    st.session_state.conversation_context['filters'][key] = value
    else:
        st.session_state.conversation_context['filters'] = {
            k: v for k, v in new_filters.items()
            if k not in ["is_continuation", "is_greeting"] and v is not None
        }
    logger.info(f"Updated conversation filters: {st.session_state.conversation_context['filters']}")

def add_to_conversation_history(user_msg: str, assistant_msg: str):
    """Add user and assistant messages to conversation history for LLM context."""
    st.session_state.conversation_context['history'].append(f"User: {user_msg}")
    st.session_state.conversation_context['history'].append(f"Assistant: {assistant_msg}")
    history_length = 8  # Keep last 8 messages (4 turns) for LLM context
    if len(st.session_state.conversation_context['history']) > history_length:
        st.session_state.conversation_context['history'] = st.session_state.conversation_context['history'][-history_length:]
    logger.info(f"Conversation history updated. Current length: {len(st.session_state.conversation_context['history'])}")

async def process_message(user_message: str):
    """Process user message, extract filters, retrieve docs, and generate response."""
    logger.info(f"Processing user message: {user_message}")
    try:
        context_for_llm = st.session_state.conversation_context['history']
        llm_filters = await extract_filters_with_llm(user_message, context_for_llm)
        update_conversation_context(llm_filters)

        if llm_filters.get("is_greeting", False):
            logger.info("Detected greeting.")
            greeting_responses = [
                "Hi there! What kind of movies are you interested in today?",
                "Hello! I'm ready to help you discover some great movies. What are you in the mood for?",
                "Hey! Tell me what you'd like to watch and I'll find some recommendations for you."
            ]
            response = greeting_responses[0]
            return response, llm_filters, []

        current_accumulated_filters = st.session_state.conversation_context['filters']
        logger.info(f"Using accumulated filters for retrieval: {current_accumulated_filters}")

        docs = await enhanced_retrieval_with_llm(st.session_state.vectorstore, user_message, current_accumulated_filters)
        logger.info(f"Retrieval completed. Found {len(docs)} documents.")

        prompt_template = ChatPromptTemplate.from_template("""
You are a helpful and knowledgeable movie assistant. Use the provided movie information to answer the user's question and recommend movies naturally and helpfully.

User's current preferences/filters derived from conversation:
{filters_info}

Movie Database Results:
{context}

User Question: {question}

IMPORTANT GUIDELINES:
1. Recommend specific movies from the provided "Movie Database Results" section when available.
2. If relevant movies are found, provide details like year, rating, genre, director, or actors from the database results.
3. Do NOT hallucinate movie details or recommend movies not present in the "Movie Database Results".
4. If no relevant movies are found in the provided context, state that you couldn't find movies matching the request in the database.
5. Maintain a conversational and friendly tone. Acknowledge the user's request and the filters applied.
6. If filters were applied, mention them briefly (e.g., "Based on your interest in Tamil movies by Shankar...").

Answer:""")

        context_content = format_docs(docs)
        filters_info_str = "No specific filters applied yet."
        if current_accumulated_filters:
            filters_info_str = json.dumps(current_accumulated_filters, indent=2)

        logger.info("Invoking response generation LLM.")
        response = await (prompt_template | st.session_state.response_llm | StrOutputParser()).ainvoke({
            "filters_info": filters_info_str,
            "context": context_content,
            "question": user_message
        })
        logger.info("Response generation complete.")

        return response, llm_filters, docs

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        error_msg = f"I encountered an error while processing your request: {str(e)}"
        return error_msg, {}, []

def run_async_process_message(user_message: str):
    """Synchronous wrapper to run the async process_message function."""
    logger.info(f"Running async process_message synchronously via asyncio.run for user input: {user_message}")
    try:
        response, filters, docs = asyncio.run(process_message(user_message))
        logger.info("Async process_message completed successfully.")
        return response, filters, docs
    except Exception as e:
        logger.error(f"Error during asyncio.run for message: {e}", exc_info=True)
        return f"An unexpected error occurred: {str(e)}", {}, []

def main():
    """Main Streamlit application entry point."""

    st.markdown('<h1 class="main-header">🎬 Movie Recommendation Assistant</h1>', unsafe_allow_html=True)

    # Initialize components if not already done
    if not st.session_state.initialized:
        with st.spinner("🚀 Initializing Movie Assistant..."):
            vectorstore, llm, response_llm, embeddings = initialize_components()
            if vectorstore is not None and llm is not None and response_llm is not None and embeddings is not None:
                st.session_state.vectorstore = vectorstore
                st.session_state.llm = llm
                st.session_state.response_llm = response_llm
                st.session_state.embeddings = embeddings
                st.session_state.initialized = True
                try:
                    collection_count = vectorstore._collection.count()
                    st.success(f"✅ Movie database loaded with {collection_count} movies!")
                    logger.info(f"Database initialized with {collection_count} movies.")
                except Exception as e:
                    st.warning(f"✅ Movie components initialized, but could not count database size: {e}")
                    logger.warning(f"Could not count database size: {e}")
                    st.session_state.collection_count = "N/A"
            else:
                st.error("❌ Failed to initialize the movie assistant. Please check your environment variables (like GROQ_API_KEY) and database setup.")
                logger.error("Initialization failed.")
                return

    # Sidebar: Show current filters and database info
    with st.sidebar:
        st.header("🎯 Current Filters")
        filters = st.session_state.conversation_context['filters']
        if filters:
            st.markdown("<div class='filter-info'>", unsafe_allow_html=True)
            for key, value in filters.items():
                if value is not None and value != "" and value != []:
                    if isinstance(value, list):
                        st.write(f"**{key.replace('_', ' ').title()}:** {', '.join(map(str, value))}")
                    elif isinstance(value, dict):
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No active filters. Ask me about movies!")

        st.markdown("---")
        st.header("📊 Database Info")
        if st.session_state.vectorstore:
            try:
                count = st.session_state.vectorstore._collection.count()
                st.metric("Total Movies", count)
            except Exception as e:
                st.warning(f"Could not get database count: {e}")
                st.metric("Total Movies", "N/A")

        st.markdown("---")
        if st.button("🔄 Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.session_state.conversation_context = {'filters': {}, 'history': []}
            logger.info("Chat history and context cleared.")
            st.rerun()

    # Chat interface
    st.header("💬 Chat with Movie Assistant")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                if "filters" in message and message["filters"]:
                    filter_display = {}
                    for k, v in message["filters"].items():
                        if v is not None and v != "" and v != []:
                            filter_display[k] = v
                    if filter_display:
                        with st.expander("🔍 Filters Detected for this turn"):
                            st.json(filter_display)
                # Uncomment below to show retrieved docs for debugging
                # if "docs" in message and message["docs"]:
                #     with st.expander(f"📚 Retrieved Documents ({len(message['docs'])})"):
                #         for j, doc in enumerate(message["docs"]):
                #             st.write(f"**Doc {j+1}:**")
                #             st.write(f"Content: {doc.page_content[:200]}...")
                #             st.json(doc.metadata)

    # Input area for user queries
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input(
                "Ask me about movies...",
                placeholder="e.g., 'I want Tamil movies by Shankar' or 'Recommend some good sci-fi movies after 2010'",
                key="user_input_text"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("Send 🚀", type="primary")

    # Handle user input and update chat
    if submit_button and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("🧠 Processing your request..."):
            response, filters, docs = run_async_process_message(user_input)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "filters": filters,
                "docs": [d.dict() if hasattr(d, 'dict') else {"page_content": d.page_content, "metadata": d.metadata} for d in docs]
            })
            add_to_conversation_history(user_input, response)
        st.rerun()

    # Show welcome message and example queries for new users
    if not st.session_state.chat_history and st.session_state.initialized:
        st.info("👋 Welcome! I'm your AI movie assistant. I can help you find movies based on your preferences like genre, director, year, language, and more. Just tell me what you're looking for!")
        st.markdown("### 💡 Try these examples:")
        example_queries = [
            "I want Tamil movies by director Shankar",
            "Recommend some good romantic comedies after 2010",
            "Show me Christopher Nolan sci-fi movies",
            "I'm looking for action movies with high ratings",
            "What are some recent Hindi movies?"
        ]
        cols = st.columns(2)
        for i, query in enumerate(example_queries):
            with cols[i % 2]:
                with st.form(key=f"example_form_{i}", clear_on_submit=True):
                    st.write(query)
                    example_submit = st.form_submit_button("Try it!", type="secondary", use_container_width=True)
                    if example_submit:
                        st.session_state.chat_history.append({"role": "user", "content": query})
                        with st.spinner("🧠 Processing example request..."):
                            response, filters, docs = run_async_process_message(query)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response,
                                "filters": filters,
                                "docs": [d.dict() if hasattr(d, 'dict') else {"page_content": d.page_content, "metadata": d.metadata} for d in docs]
                            })
                        st.rerun()

if __name__ == "__main__":
    # Only relevant for direct script execution, not for Streamlit runtime
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            logger.warning("Existing event loop detected. Streamlit main function will run synchronously.")
        else:
            pass
    except RuntimeError:
        pass

    main()