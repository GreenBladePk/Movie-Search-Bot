import asyncio
import os
import re
from dotenv import load_dotenv
import chainlit as cl
import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize LLM
llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
    max_tokens=512
)

# --- Query Filter Parser ---
def parse_query_filters(query: str) -> dict:
    filters = {}

    year_match = re.search(r'\b(19|20)\d{2}\b', query)
    if year_match:
        filters["year"] = int(year_match.group())

    genre_match = re.findall(r'\b(action|comedy|drama|sci-fi|thriller|romance|horror|animation|documentary)\b', query, re.IGNORECASE)
    if genre_match:
        filters["genres"] = {"$in": [g.lower() for g in genre_match]}

    rating_match = re.search(r'(?:rating|score)\s*(?:above|over|greater than|>=|more than)?\s*(\d+(\.\d+)?)', query, re.IGNORECASE)
    if rating_match:
        filters["rating"] = {"$gte": float(rating_match.group(1))}

    return filters

# --- Format Chat History + Context ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def combine_input(inputs, memory):
    chat_history = memory.load_memory_variables({})["chat_history"]
    history_str = ""
    for msg in chat_history:
        if msg.__class__.__name__ == "HumanMessage":
            history_str += f"User: {msg.content}\n"
        else:
            history_str += f"Assistant: {msg.content}\n"
    return {
        "chat_history": history_str,
        "context": format_docs(inputs["docs"]),
        "question": inputs["question"]
    }

# --- Chat Start Handler ---
@cl.on_chat_start
async def on_chat_start():
    logger.info("Starting new chat session...")

    try:
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
        logger.info(f"Loaded Chroma vector store with {vectorstore._collection.count()} documents")

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="output"
        )

        prompt_template = ChatPromptTemplate.from_template("""
You are a friendly and knowledgeable movie assistant. Using the provided movie information and conversation history, answer the user's question in a concise, natural, and conversational way. Summarize key details, avoid repeating raw data verbatim, and ensure consistency with prior messages in this session. If the input is vague (e.g., "hi"), respond with a friendly prompt to clarify the user's request.

Conversation History:
{chat_history}

Movie Information:
{context}

Question:
{question}

Answer:
""")

        session_id = cl.user_session.get("id") or "default_session"
        cl.user_session.set(f"vectorstore_{session_id}", vectorstore)
        cl.user_session.set(f"retriever_{session_id}", retriever)
        cl.user_session.set(f"memory_{session_id}", memory)
        cl.user_session.set("prompt_template", prompt_template)

        await cl.Message(
            content="🎬 Welcome to the Movie Assistant! I can help you find information about movies from the IMDB database. What would you like to know?"
        ).send()

    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        await cl.Message(content=f"❌ Error loading the movie database: {str(e)}").send()

# --- Chat Message Handler ---
@cl.on_message
async def on_message(message: cl.Message):
    logger.info(f'Processing message: {message.content}')

    session_id = cl.user_session.get("id") or "default_session"
    vectorstore = cl.user_session.get(f"vectorstore_{session_id}")
    memory = cl.user_session.get(f"memory_{session_id}")
    prompt_template = cl.user_session.get("prompt_template")

    if not vectorstore or not memory or not prompt_template:
        await cl.Message(content="❌ Chat system not properly initialized. Please refresh the page.").send()
        return

    msg = cl.Message(content="")

    try:
        async with cl.Step(type="retrieval", name="🔍 Searching movie database"):
            await msg.stream_token("🔍 Searching for relevant movie information...\n")

            filters = parse_query_filters(message.content)
            logger.info(f"Extracted filters: {filters}")

            search_kwargs = {"k": 5}
            if filters:
                search_kwargs["filter"] = filters

            filtered_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )


            docs = await filtered_retriever.ainvoke(message.content)

            if docs:
                await cl.Message(content=f"✅ Found {len(docs)} relevant movie documents").send()
            else:
                await cl.Message(content="⚠️ No relevant documents found for your filters.").send()

        async with cl.Step(type="run", name="🤖 Generating response"):
            await msg.stream_token("🎬 ")

            if message.content.strip().lower() in ["hi", "hello", "hey"]:
                response = "Hi there! I'm ready to help with any movie questions you have. What's on your mind?"
                await msg.stream_token(response)
            else:
                combined_inputs = combine_input({"docs": docs, "question": message.content}, memory)
                response = await (prompt_template | llm | StrOutputParser()).ainvoke(combined_inputs)
                await msg.stream_token(response)

            memory.save_context({"question": message.content}, {"output": response})

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        response = f"❌ I encountered an error: {str(e)}"
        await msg.stream_token(response)
        memory.save_context({"question": message.content}, {"output": response})

    await msg.send()

# --- Test Script for Local Use ---
async def test_retrieval(query: str = "sci-fi movies from the 1980s with rating above 8"):
    try:
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

        filters = parse_query_filters(query)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": filters})
        docs = retriever.get_relevant_documents(query)

        logger.info(f"Test query: {query}")
        logger.info(f"Filters applied: {filters}")
        logger.info(f"Found {len(docs)} relevant documents")
        for i, doc in enumerate(docs, 1):
            logger.info(f"Document {i}: {doc.page_content[:200]}...")

        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_retrieval())


