import os
import shutil
import glob
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from backend.rag_core import VectorStore
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FRONTEND_DIR, exist_ok=True)

vector_store = VectorStore(data_dir=DATA_DIR)

API_KEY_PATH = os.path.join(BASE_DIR, "api_key.txt")
try:
    with open(API_KEY_PATH, "r") as f:
        api_key_content = f.read().strip()
except FileNotFoundError:
    api_key_content = "your-sk-api-key-here"

client = OpenAI(api_key=api_key_content or os.getenv("OPENAI_API_KEY", "your-sk-api-key-here"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    vector_store.load_or_build()
    yield

app = FastAPI(title="NumPy RAG Prototype", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    model_provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    api_key: str = ""
    smart_search: bool = False

class QueryResponse(BaseModel):
    answer: str
    sources: list
    expanded_query: str = ""

def _get_llm_client(provider: str, api_key: str = ""):
    if provider == "ollama":
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    else:
        effective_key = api_key.strip() if api_key.strip() else (api_key_content or os.getenv("OPENAI_API_KEY", ""))
        return OpenAI(api_key=effective_key)

def _expand_query(llm_client, model_name: str, user_query: str) -> str:
    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": (
                    "You are a search query optimizer. Given a user's question, generate an expanded search query "
                    "that would help find the most relevant chunks in a document database.\n\n"
                    "Rules:\n"
                    "- Output ONLY the expanded search keywords/phrases, nothing else.\n"
                    "- Include synonyms, related terms, and alternative phrasings.\n"
                    "- Keep it concise (under 50 words).\n"
                    "- Do NOT answer the question, just produce search terms.\n\n"
                    "Example:\n"
                    "User: 'who are team members'\n"
                    "Output: 'team members names contributors authors people involved participants group members list'"
                )},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0,
            max_tokens=80
        )
        expanded = response.choices[0].message.content.strip()
        return expanded if expanded else user_query
    except Exception as e:
        print(f"Query expansion failed, using original query: {e}")
        return user_query

@app.post("/api/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    llm_client = _get_llm_client(req.model_provider, req.api_key)
    
    search_query = req.query
    expanded_query = ""
    if req.smart_search:
        expanded_query = _expand_query(llm_client, req.model_name, req.query)
        search_query = expanded_query
        print(f"[Smart Search] Original: '{req.query}' -> Expanded: '{expanded_query}'")
    
    top_chunks = vector_store.search(search_query, top_k=5)
    
    if not top_chunks:
        return QueryResponse(
            answer="No context available. Please add some files to the data/ directory and restart.",
            sources=[],
            expanded_query=expanded_query
        )

    context_text = "\n\n---\n\n".join([item["chunk"] for item in top_chunks])
    
    system_prompt = (
        "You are a knowledgeable and articulate assistant. Your job is to answer the user's question "
        "by carefully reading the retrieved context below and synthesizing a clear, well-structured answer "
        "IN YOUR OWN WORDS. Do NOT simply copy-paste chunks from the context.\n\n"
        "Guidelines:\n"
        "- Explain concepts clearly as if you were a helpful expert.\n"
        "- If the context contains enough information, provide a thorough and detailed answer.\n"
        "- If the context is insufficient, say so honestly rather than making things up.\n"
        "- Use Markdown formatting in your response: headings, bold, bullet points, code blocks, etc. where appropriate.\n"
        "- If the context contains code, include and explain it properly using fenced code blocks with the correct language tag.\n"
        "- Do NOT hallucinate information that is not present in the context.\n\n"
        "RETRIEVED CONTEXT:\n"
        f"{context_text}"
    )

    try:
        response = llm_client.chat.completions.create(
            model=req.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.query}
            ],
            temperature=0.3
        )
        answer = response.choices[0].message.content
        
    except Exception as e:
        if req.model_provider == "ollama":
            answer = f"Error calling Ollama (Is the server running at localhost:11434?): {str(e)}\n\nHowever, I did retrieve the following relevant chunks. You can view them below."
        else:
            answer = f"Error calling LLM (Please verify your OPENAI_API_KEY): {str(e)}\n\nHowever, I did retrieve the following relevant chunks. You can view them below."
        
    return QueryResponse(
        answer=answer,
        sources=top_chunks,
        expanded_query=expanded_query
    )

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".txt", ".md"]:
        raise HTTPException(status_code=400, detail="Only .txt and .md files are allowed")
    
    file_path = os.path.join(DATA_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    finally:
        file.file.close()
        
    vector_store.build_index()
    return {"message": f"Successfully uploaded and indexed {file.filename}"}

@app.get("/api/files")
async def list_files():
    try:
        files = []
        for ext in ["*.txt", "*.md"]:
            files.extend(glob.glob(os.path.join(DATA_DIR, ext)))
        
        file_list = []
        for f in files:
            basename = os.path.basename(f)
            stat = os.stat(f)
            file_list.append({
                "filename": basename,
                "size": stat.st_size
            })
        return file_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    if os.path.dirname(filename) or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        os.remove(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")
        
    vector_store.build_index()
    return {"message": f"Successfully deleted and re-indexed {filename}"}

class ApiKeyRequest(BaseModel):
    api_key: str

@app.post("/api/settings/apikey")
async def update_api_key(req: ApiKeyRequest):
    global api_key_content, client
    new_key = req.api_key.strip()
    if not new_key:
        raise HTTPException(status_code=400, detail="API key cannot be empty")
    api_key_content = new_key
    client = OpenAI(api_key=new_key)
    try:
        with open(API_KEY_PATH, "w") as f:
            f.write(new_key)
    except Exception:
        pass
    return {"message": "API key updated successfully"}

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_FAMILIES = {"nomic-bert", "bert", "e5"}

@app.get("/api/ollama/models")
async def list_ollama_models():
    try:
        async with httpx.AsyncClient(timeout=5.0) as http_client:
            resp = await http_client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            data = resp.json()

        models = []
        for m in data.get("models", []):
            family = m.get("details", {}).get("family", "")
            name = m.get("name", "")
            if family in EMBEDDING_FAMILIES:
                continue
            if "base" in name:
                continue
            params = m.get("details", {}).get("parameter_size", "")
            label = f"{name} ({params})" if params else name
            models.append({"name": name, "label": label})

        return {"models": models}
    except httpx.ConnectError:
        return {"models": [], "error": "Ollama is not running on localhost:11434"}
    except Exception as e:
        return {"models": [], "error": str(e)}

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
