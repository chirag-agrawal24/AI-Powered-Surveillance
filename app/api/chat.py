import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from supabase import create_client, Client
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="CCTV Captioning Chatbot API")

# Set Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize embeddings and LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.9, google_api_key=GOOGLE_API_KEY)

# Pydantic model for POST request body
class QueryRequest(BaseModel):
    query: str

def clean_caption_text(caption_text: str) -> list:
    """Extract relevant descriptions from caption_text."""
    segments = [segment.strip() for segment in caption_text.split(";")]
    descriptions = []
    for segment in segments:
        if not segment.lower().startswith(("describe the actions", "the recognized person", "relevant objects")):
            if segment:
                descriptions.append(segment)
    return descriptions

def get_chunks(query: str) -> list:
    """Fetch top 5 chunks from Supabase."""
    try:
        query_embedding = embeddings.embed_query(query)
        response = supabase.rpc("match_caption_chunks", {
            "query_embedding": query_embedding,
            "match_threshold": 0.6,
            "match_count": 5
        }).execute()
        chunks = response.data
        if not chunks:
            response = supabase.table("caption_chunks").select("*").order("created_at", desc=True).limit(5).execute()
            chunks = response.data
        return chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chunks: {str(e)}")

def format_context(chunks: list) -> str:
    """Format chunks into a simple context string."""
    context = ""
    for chunk in chunks:
        cleaned_captions = clean_caption_text(chunk["caption_text"])
        if not cleaned_captions:
            continue
        start_time = chunk["start_timestamp"]
        end_time = chunk["end_timestamp"]
        objects = ", ".join([f"{count} {label}" for label, count in chunk["objects"].items()]) if chunk["objects"] else "no objects"
        persons = ", ".join(chunk["persons"]) if chunk["persons"] else "no persons"
        actions = "; ".join(cleaned_captions)
        context += f"[{start_time} to {end_time}] Objects: {objects}, Persons: {persons}, Actions: {actions}\n"
    return context if context else "No relevant information found."

def chatbot(query: str) -> str:
    """Process a user query and return the answer."""
    try:
        # Get and format chunks
        chunks = get_chunks(query)
        context = format_context(chunks)

        # Create prompt
        prompt = f"""Answer the question based on the CCTV captioning data below. If the data doesn't have enough information, say so.

Data:
{context}

Question: {query}

Answer:"""

        # Generate response
        response = llm.invoke(prompt).content
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# GET endpoint
@app.get("/query")
async def get_query(query: str):
    """Handle GET request to process a query."""
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")
    response = chatbot(query)
    return {"query": query, "response": response}

# POST endpoint
@app.post("/query")
async def post_query(request: QueryRequest):
    """Handle POST request to process a query."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query field is required.")
    response = chatbot(request.query)
    return {"query": request.query, "response": response}

# Run the server
if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
