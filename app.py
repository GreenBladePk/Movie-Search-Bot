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

# Apply nest_asyncio to allow nested event loops if necessary within libraries
# (Less critical with correct asyncio.run usage, but doesn't hurt)
nest_asyncio.apply()

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'", module="torch")
# Note: Setting PYTORCH_ENABLE_MPS_FALLBACK might be relevant for Apple Silicon users,
# but the primary issue seems related to event loop handling, not MPS itself.
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # Keep if needed for environment

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="🎬 Movie Recommendation Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: white;  /* Changed to white */
        background-clip: text;
        color: white !important;         /* Ensure white font */
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: black !important;         /* Ensure black font */
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: black !important;         /* Ensure black font */
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        color: black !important;         /* Ensure black font */
    }
    .filter-info {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: black !important;         /* Ensure black font */
    }
    .stAlert > div {
        border-radius: 10px;
        color: black !important;         /* Ensure black font */
    }
</style>
""", unsafe_allow_html=True)

# Movie Filters Model
class MovieFilters(BaseModel):
    """Structured filters for movie search"""
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

# Initialize session state
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
    st.session_state.embeddings = None # Store embeddings in session state too if needed later


@st.cache_resource
def initialize_components():
    """Initialize and cache the components"""
    logger.info("Starting component initialization...")
    try:
        # Initialize embeddings (synchronous call)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logger.info("Embeddings initialized.")

        # Initialize LLMs (synchronous calls)
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

        # Initialize vectorstore (synchronous call)
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
    """Use LLM to extract structured filters from natural language query (async)"""
    
    # Build context from conversation history
    context_str = ""
    if conversation_context:
        # Use recent history for context
        context_str = f"\nPrevious conversation context:\n" + "\n".join(conversation_context[-6:]) # Increased history length
    
    # Filter extraction prompt
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
Output: {{"language": "Tamil", "director": "Shankar", "keywords": [], "is_greeting": false, "is_continuation": false}} # Keywords are empty as director and language are specific filters

Query: "I am having a good mood today, can you recommend me some rom-com released after 2010"
Output: {{"genres": ["romance", "comedy"], "year_after": 2010, "keywords": ["good mood", "recommend"], "is_greeting": false, "is_continuation": false}}

Query: "hi"
Output: {{"language": null, "director": null, "year_exact": null, "year_after": null, "year_before": null, "year_between": null, "genres": null, "rating_min": null, "rating_max": null, "actor": null, "keywords": null, "is_greeting": true, "is_continuation": false}} # Return full structure even for greeting

Query: "just english" (assuming previous query was about movies)
Output: {{"language": "English", "keywords": [], "is_greeting": false, "is_continuation": true}} # Note is_continuation=true

{context_str}

User Query: "{query}"

JSON Output:""")

    try:
        logger.info(f"Attempting to extract filters for query: {query}")
        # Use the filter extraction LLM
        response = await (filter_extraction_prompt | st.session_state.llm | StrOutputParser()).ainvoke({
            "query": query,
            "context_str": context_str
        })

        # Clean the response and parse JSON
        response = response.strip()
        # Robustly find JSON block
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end != -1 and json_end > json_start:
            response = response[json_start:json_end]
        else:
             # If no JSON block found, try cleaning common code block markers
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]

        filters = json.loads(response.strip())

        # Validate and clean filters - ensure only valid fields from MovieFilters are kept
        valid_filter_keys = MovieFilters.model_fields.keys()
        cleaned_filters = {}
        for key, value in filters.items():
            if key in valid_filter_keys and value is not None and value != "" and value != []:
                # Basic type validation/correction if needed (e.g., ensure year is int)
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
                     # If LLM returns a single string instead of list for genres/keywords
                     value = [v.strip() for v in value.split(',') if v.strip()]
                elif key == "year_between" and isinstance(value, list):
                    if len(value) == 2:
                        try:
                           value = [int(value[0]), int(value[1])]
                           if value[0] > value[1]: # Swap if start > end year
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
        # Fallback: treat as a basic keyword search if parsing fails
        return {"keywords": query.lower().split(), "is_greeting": False, "is_continuation": False}
    except Exception as e:
        logger.error(f"Filter extraction error: {e}", exc_info=True)
        # Fallback: treat as a basic keyword search if LLM invocation fails
        return {"keywords": query.lower().split(), "is_greeting": False, "is_continuation": False}

def convert_to_chroma_filters(llm_filters: Dict[str, Any]) -> Dict[str, Any]:
    """Convert LLM-extracted filters to Chroma-compatible format"""
    chroma_filters = {}

    # Chroma's filtering syntax can be sensitive.
    # Basic $eq, $gt, $lt, $gte, $lte, $ne, $in, $nin operators work well for numeric/exact matches.
    # Text containment ($contains) works for string lists.
    # Complex $and/$or with different field conditions can be tricky.

    where_filters = {}
    where_document_filters = {} # Filters applied to document content

    # Language filtering - usually a metadata field
    if "language" in llm_filters:
        # Assuming 'language' is stored as a list in metadata
        where_filters["language"] = {"$contains": llm_filters['language']}

    # Director filtering - usually a metadata field
    if "director" in llm_filters:
         # Assuming 'director' is stored as a string or list in metadata
        where_filters["director"] = {"$contains": llm_filters['director']}


    # Actor filtering - usually a metadata field ('actors' or 'cast')
    if "actor" in llm_filters:
        # Use $or for 'actors' or 'cast' fields
        where_filters["$or"] = [
             {"actors": {"$contains": llm_filters['actor']}}, # Assuming actors is a list
             {"cast": {"$contains": llm_filters['actor']}}    # Assuming cast is a list
         ]

    # Year filtering (assuming 'year' is stored as an integer in metadata)
    if "year_exact" in llm_filters:
        where_filters["year"] = {"$eq": llm_filters["year_exact"]}
    elif "year_after" in llm_filters:
        where_filters["year"] = {"$gte": llm_filters["year_after"]} # Use $gte for "after" meaning year >= year_after
    elif "year_before" in llm_filters:
        where_filters["year"] = {"$lte": llm_filters["year_before"]} # Use $lte for "before" meaning year <= year_before
    elif "year_between" in llm_filters and len(llm_filters["year_between"]) == 2:
        start_year, end_year = sorted(llm_filters["year_between"]) # Ensure correct order
        if "$and" not in where_filters: where_filters["$and"] = []
        where_filters["$and"].append({"year": {"$gte": start_year}})
        where_filters["$and"].append({"year": {"$lte": end_year}})

    # Genre filtering (assuming 'genres' is stored as a list in metadata)
    if "genres" in llm_filters:
        genres = llm_filters["genres"]
        if len(genres) == 1:
            where_filters["genres"] = {"$contains": genres[0]}
        else:
            # For multiple genres, require ANY of the listed genres
            # This requires an $or within the metadata filter
            if "$or" not in where_filters: where_filters["$or"] = []
            genre_clauses = [{"genres": {"$contains": genre}} for genre in genres]
            # If there's already an $or clause (e.g., for actors), we need to combine them
            # This can become complex. A simpler approach is to add genre clauses
            # and handle potential conflicts with other $or/ $and later or in post-filtering.
            # Let's try combining with existing $or if it exists, otherwise add it.
            if "$or" in where_filters:
                # Combine existing $or with new genre $or
                existing_or = where_filters.pop("$or")
                where_filters["$and"] = [{"$or": existing_or}, {"$or": genre_clauses}]
            else:
                 where_filters["$or"] = genre_clauses

    # Rating filtering (assuming 'imdb_rating' is stored as a float in metadata)
    if "rating_min" in llm_filters:
        if "$and" not in where_filters: where_filters["$and"] = []
        where_filters["$and"].append({"imdb_rating": {"$gte": llm_filters["rating_min"]}})
    if "rating_max" in llm_filters:
        if "$and" not in where_filters: where_filters["$and"] = []
        where_filters["$and"].append({"imdb_rating": {"$lte": llm_filters["rating_max"]}})

    # Handle complex $and/$or structure
    # If both "$and" and "$or" exist at the top level, this structure is invalid in Chroma
    # You generally need one top-level operator ($and or $or) or simple key-value pairs.
    # Let's simplify: if both exist, combine everything under a single $and.
    # This assumes an AND logic between all conditions, which might not always be desired
    # but is safer than an invalid filter structure.
    combined_filters = {}
    if "$and" in where_filters or "$or" in where_filters:
        combined_clauses = []
        if "$and" in where_filters:
            combined_clauses.extend(where_filters.pop("$and"))
        if "$or" in where_filters:
             # Wrap the top-level $or clauses in an $or condition within the $and
             # This assumes we want (cond1 AND cond2 ...) AND (or_cond_A OR or_cond_B ...)
             combined_clauses.append({"$or": where_filters.pop("$or")})

        # Add remaining simple conditions (those not in $and or $or)
        for key, value in where_filters.items():
             combined_clauses.append({key: value})

        if len(combined_clauses) == 1:
            # If only one clause resulted, don't wrap in $and
            combined_filters = combined_clauses[0]
        elif combined_clauses:
            combined_filters["$and"] = combined_clauses

    else:
         # No complex operators, just simple key-value conditions
         combined_filters = where_filters

    # Keywords can be used for 'where_document' filtering or in the query text
    if "keywords" in llm_filters:
        # For simplicity and robustness with similarity search,
        # we will primarily use keywords by including them in the search query text.
        # Adding them to where_document can sometimes be too restrictive.
        pass

    # Chroma filter structure should be { "$and": [...], "$or": [...], "field": {...}, ... }
    # Let's return the final combined_filters for the 'where' clause
    return combined_filters

async def enhanced_retrieval_with_llm(vectorstore, query: str, llm_filters: Dict[str, Any], k: int = 5):
    """Enhanced retrieval using LLM-extracted filters (async)"""
    try:
        # Handle greetings - although handled in process_message, good to have check here
        if llm_filters.get("is_greeting", False):
            logger.info("Skipping retrieval for greeting.")
            return []

        # Convert to Chroma filters
        chroma_metadata_filters = convert_to_chroma_filters(llm_filters)
        # No where_document filters implemented in convert_to_chroma_filters currently

        # Build the search query text
        # Include keywords and potentially other filter values in the query text
        # to guide the embedding similarity search, *in addition* to metadata filtering.
        search_query_parts = [query]
        if llm_filters.get("keywords"):
             search_query_parts.extend(llm_filters["keywords"])
        if llm_filters.get("language") and "language" not in chroma_metadata_filters: # Only add if not strictly filtered
             search_query_parts.append(llm_filters["language"])
        if llm_filters.get("director") and "director" not in chroma_metadata_filters: # Only add if not strictly filtered
             search_query_parts.append(llm_filters["director"])
        if llm_filters.get("actor"): # Actor filter is complex, always add to query
             search_query_parts.append(llm_filters["actor"])
        if llm_filters.get("genres") and "$or" not in chroma_metadata_filters.values() and not any(isinstance(val, dict) and "$or" in val for val in chroma_metadata_filters.values()):
             # Only add genres to query if not using complex $or filter for genres
             search_query_parts.extend(llm_filters["genres"])

        enhanced_query = " ".join(list(set(search_query_parts))) # Use set to deduplicate
        logger.info(f"Enhanced search query: {enhanced_query}")


        search_kwargs = {"k": k}
        if chroma_metadata_filters:
             logger.info(f"Applying Chroma metadata filters: {chroma_metadata_filters}")
             search_kwargs["where"] = chroma_metadata_filters
             # If you had where_document filters, add them here:
             # if where_document_filters:
             #      search_kwargs["where_document"] = where_document_filters


        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

        # Perform async retrieval
        docs = await retriever.ainvoke(enhanced_query)

        logger.info(f"Retrieved {len(docs)} documents.")
        return docs

    except Exception as e:
        logger.error(f"Enhanced retrieval error: {e}", exc_info=True)
        # Fallback: Basic similarity search if filtering fails
        logger.warning("Falling back to basic similarity search due to error.")
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            return await retriever.ainvoke(query)
        except Exception as final_error:
            logger.error(f"Final fallback error: {final_error}", exc_info=True)
            return []


def format_docs(docs):
    """Format documents with better structure"""
    if not docs:
        return "No relevant movie information found in the database."

    formatted_docs = []
    for doc in docs:
        content = doc.page_content
        metadata = getattr(doc, 'metadata', {})
        details = []
        # Add specific metadata fields if they exist and are relevant
        if 'title' in metadata and metadata['title']:
            details.append(f"Title: {metadata['title']}")
        if 'year' in metadata and metadata['year']:
            details.append(f"Year: {metadata['year']}")
        if 'director' in metadata and metadata['director']:
            details.append(f"Director: {metadata['director']}")
        if 'language' in metadata and metadata['language']:
            details.append(f"Language: {metadata['language']}")
        if 'genres' in metadata and metadata['genres']:
            # Join genres if it's a list
            genres_str = ", ".join(metadata['genres']) if isinstance(metadata['genres'], list) else metadata['genres']
            if genres_str:
                 details.append(f"Genres: {genres_str}")
        if 'imdb_rating' in metadata and metadata['imdb_rating']:
             details.append(f"IMDb Rating: {metadata['imdb_rating']}")
        if 'actors' in metadata and metadata['actors']:
             # Join actors if it's a list, show first few
            actors_str = ", ".join(metadata['actors'][:3]) + ('...' if len(metadata['actors']) > 3 else '') if isinstance(metadata['actors'], list) else metadata['actors']
            if actors_str:
                 details.append(f"Actors: {actors_str}")


        metadata_str = ", ".join(details) if details else "Details: N/A"
        formatted_docs.append(f"{content}\n{metadata_str}")

    return "\n\n---\n\n".join(formatted_docs)

def update_conversation_context(new_filters: Dict[str, Any]):
    """Update conversation context with new filters"""
    # Always reset filters unless it's explicitly a continuation
    if new_filters.get("is_continuation", False):
        # Merge new filters into existing ones for continuation
        for key, value in new_filters.items():
             # Do not merge is_greeting or is_continuation flags themselves
            if key not in ["is_continuation", "is_greeting"] and value is not None:
                # For list types like genres, extend the list
                if key in ["genres", "keywords"] and isinstance(value, list) and isinstance(st.session_state.conversation_context['filters'].get(key), list):
                     st.session_state.conversation_context['filters'][key].extend(value)
                     st.session_state.conversation_context['filters'][key] = list(set(st.session_state.conversation_context['filters'][key])) # Deduplicate
                else:
                    # For other types, overwrite or set
                    st.session_state.conversation_context['filters'][key] = value
    else:
        # Replace filters completely for a new search query
        st.session_state.conversation_context['filters'] = {
            k: v for k, v in new_filters.items()
            if k not in ["is_continuation", "is_greeting"] and v is not None
        }
    logger.info(f"Updated conversation filters: {st.session_state.conversation_context['filters']}")


def add_to_conversation_history(user_msg: str, assistant_msg: str):
    """Add conversation to history"""
    st.session_state.conversation_context['history'].append(f"User: {user_msg}")
    st.session_state.conversation_context['history'].append(f"Assistant: {assistant_msg}")
    # Keep only last N messages for context (adjust N as needed)
    history_length = 8 # Keep last 8 messages (4 turns) for LLM context
    if len(st.session_state.conversation_context['history']) > history_length:
        st.session_state.conversation_context['history'] = st.session_state.conversation_context['history'][-history_length:]
    logger.info(f"Conversation history updated. Current length: {len(st.session_state.conversation_context['history'])}")


async def process_message(user_message: str):
    """Process user message and generate response (async function)"""
    logger.info(f"Processing user message: {user_message}")
    try:
        # Extract filters using LLM
        context_for_llm = st.session_state.conversation_context['history']
        # Note: This is an async call, which is fine *within* this async function
        llm_filters = await extract_filters_with_llm(user_message, context_for_llm)

        # Update conversation context (sync function call)
        update_conversation_context(llm_filters)

        # Handle greetings
        if llm_filters.get("is_greeting", False):
            logger.info("Detected greeting.")
            greeting_responses = [
                "Hi there! What kind of movies are you interested in today?",
                "Hello! I'm ready to help you discover some great movies. What are you in the mood for?",
                "Hey! Tell me what you'd like to watch and I'll find some recommendations for you."
            ]
            response = greeting_responses[0] # Or random.choice(greeting_responses)
            # History update happens after determining final response
            return response, llm_filters, [] # Return empty docs for greeting

        # Use current conversation filters for retrieval, combined with current query filters
        # For now, the retrieval function uses the LLM filters directly, which are derived from the query
        # and potentially merge with conversation context via update_conversation_context.
        # Need to decide: does retrieval use *only* the filters extracted from the *current* query,
        # or the *accumulated* filters in st.session_state.conversation_context['filters']?
        # The current design of update_conversation_context suggests accumulated filters.
        # Let's pass the *current* query and the *current* state of conversation_context['filters'] to retrieval.

        # Use the *current* query text for semantic search, combined with *current* accumulated filters.
        current_accumulated_filters = st.session_state.conversation_context['filters']
        logger.info(f"Using accumulated filters for retrieval: {current_accumulated_filters}")

        # Enhanced retrieval with ACCUMULATED filters and CURRENT query text
        docs = await enhanced_retrieval_with_llm(st.session_state.vectorstore, user_message, current_accumulated_filters)
        logger.info(f"Retrieval completed. Found {len(docs)} documents.")

        # Generate response
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

        # Format filters info for the response LLM prompt
        filters_info_str = "No specific filters applied yet."
        if current_accumulated_filters:
            filters_info_str = json.dumps(current_accumulated_filters, indent=2)

        logger.info("Invoking response generation LLM.")
        # Use the response generation LLM
        response = await (prompt_template | st.session_state.response_llm | StrOutputParser()).ainvoke({
            "filters_info": filters_info_str,
            "context": context_content,
            "question": user_message
        })
        logger.info("Response generation complete.")

        # History update happens after determining final response
        # add_to_conversation_history(user_message, response) # Handled in main now

        return response, llm_filters, docs

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        error_msg = f"I encountered an error while processing your request: {str(e)}"
        # History update happens after determining final response
        # add_to_conversation_history(user_message, error_msg) # Handled in main now
        return error_msg, {}, [] # Return empty filters and docs on error

# --- New Synchronous Wrapper for Async Function ---
def run_async_process_message(user_message: str):
    """Synchronous wrapper to run the async process_message function using asyncio.run."""
    logger.info(f"Running async process_message synchronously via asyncio.run for user input: {user_message}")
    try:
        # asyncio.run() is the recommended way to run the top-level async function
        # from a synchronous context. It handles creating, running, and closing
        # the event loop for this specific execution.
        response, filters, docs = asyncio.run(process_message(user_message))
        logger.info("Async process_message completed successfully.")
        return response, filters, docs
    except Exception as e:
        logger.error(f"Error during asyncio.run for message: {e}", exc_info=True)
        # Propagate error or return a specific error response
        return f"An unexpected error occurred: {str(e)}", {}, []


def main():
    """Main Streamlit application (synchronous)"""

    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation Assistant</h1>', unsafe_allow_html=True)

    # Initialize components
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
                return # Stop execution if initialization fails

    # Sidebar with current filters
    with st.sidebar:
        st.header("🎯 Current Filters")
        # Display filters in a more readable way
        filters = st.session_state.conversation_context['filters']
        if filters:
            st.markdown("<div class='filter-info'>", unsafe_allow_html=True)
            for key, value in filters.items():
                 if value is not None and value != "" and value != []:
                    if isinstance(value, list):
                         st.write(f"**{key.replace('_', ' ').title()}:** {', '.join(map(str, value))}")
                    elif isinstance(value, dict): # Handle complex filters like year_between
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

                # Show filters if available
                if "filters" in message and message["filters"]:
                    # Format filters for display
                    filter_display = {}
                    for k, v in message["filters"].items():
                         if v is not None and v != "" and v != []:
                            filter_display[k] = v

                    if filter_display:
                        with st.expander("🔍 Filters Detected for this turn"):
                            st.json(filter_display)
                # Optional: Show retrieved documents for debugging/insight
                # if "docs" in message and message["docs"]:
                #      with st.expander(f"📚 Retrieved Documents ({len(message['docs'])})"):
                #          for j, doc in enumerate(message["docs"]):
                #             st.write(f"**Doc {j+1}:**")
                #             st.write(f"Content: {doc.page_content[:200]}...") # Show snippet
                #             st.json(doc.metadata)

    # Input area
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])

        with col1:
            user_input = st.text_input(
                "Ask me about movies...",
                placeholder="e.g., 'I want Tamil movies by Shankar' or 'Recommend some good sci-fi movies after 2010'",
                key="user_input_text" # Add a key to manage state
            )

        with col2:
            # Add some space above the button to align it better
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("Send 🚀", type="primary")

    # Process input
    if submit_button and user_input.strip():
        # Add user message to chat history immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Show processing status
        with st.spinner("🧠 Processing your request..."):
            # Use the synchronous wrapper to call the async process_message
            response, filters, docs = run_async_process_message(user_input)

            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "filters": filters, # Store filters for display if needed
                "docs": [d.dict() if hasattr(d, 'dict') else {"page_content": d.page_content, "metadata": d.metadata} for d in docs] # Store doc info (serialize if necessary)
            })

            # Update conversation history text for the LLM context
            # This happens after the response is generated
            add_to_conversation_history(user_input, response)

        # Rerun to update the chat display and sidebar
        st.rerun()

    # Welcome message for new users
    if not st.session_state.chat_history and st.session_state.initialized: # Only show if initialized
        st.info("👋 Welcome! I'm your AI movie assistant. I can help you find movies based on your preferences like genre, director, year, language, and more. Just tell me what you're looking for!")

        # Example queries
        st.markdown("### 💡 Try these examples:")
        example_queries = [
            "I want Tamil movies by director Shankar",
            "Recommend some good romantic comedies after 2010",
            "Show me Christopher Nolan sci-fi movies",
            "I'm looking for action movies with high ratings",
            "What are some recent Hindi movies?"
        ]

        # Use columns to arrange buttons
        cols = st.columns(2)
        for i, query in enumerate(example_queries):
            # Using form for each button ensures it triggers a single submit event
            # This avoids potential issues with multiple buttons trying to modify state simultaneously
            with cols[i % 2]:
                 with st.form(key=f"example_form_{i}", clear_on_submit=True):
                     st.write(query) # Display the query text
                     example_submit = st.form_submit_button("Try it!", type="secondary", use_container_width=True)
                     if example_submit:
                         st.session_state.chat_history.append({"role": "user", "content": query})
                         with st.spinner("🧠 Processing example request..."):
                              # Use the synchronous wrapper for example queries too
                              response, filters, docs = run_async_process_message(query)
                              st.session_state.chat_history.append({
                                   "role": "assistant",
                                   "content": response,
                                   "filters": filters,
                                   "docs": [d.dict() if hasattr(d, 'dict') else {"page_content": d.page_content, "metadata": d.metadata} for d in docs]
                               })
                         st.rerun()


if __name__ == "__main__":
    # Ensure we are not already running an event loop if this script is executed directly
    # (This check is more for general Python execution than Streamlit, but good practice)
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            logger.warning("Existing event loop detected. Streamlit main function will run synchronously.")
        else:
             # No running loop, safe to proceed with synchronous main
             pass # Streamlit will run main()
    except RuntimeError:
         # No running loop, which is the expected state for Streamlit's main execution thread
         pass # Streamlit will run main()

    main() # Execute the main Streamlit function