import importlib, json, os, tempfile, time, uuid
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tools import rag

import os
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# ragwire fastapi reference
# https://kgptalkie.com/ragwire-api-reference/

## Windows
#$env:AGENT="02_langgraph_self_correcting_agent"; python main.py

## Linux/MacOS
# AGENT=02_langgraph_self_correcting_agent python main.py


# agent = importlib.import_module(f"agents.{os.getenv('AGENT', '01_langchain_agent')}")
agent = importlib.import_module(f"agents.{os.getenv('AGENT', '02_langgraph_self_correcting_agent')}")

router = APIRouter()

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = agent.MODEL_ID
    messages: List[Message]


def chunk(cid, ts, content="", finish_reason=None):
    delta = {"content": content} if content else ({"role": "assistant", "content": ""} if finish_reason is None else {})
    return f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': ts, 'model': agent.MODEL_ID, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}]})}\n\n"


API_KEY = os.getenv("API_KEY")
bearer = HTTPBearer(auto_error=False)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(bearer)):
    if API_KEY and (not credentials or credentials.credentials != API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    
@router.get("/health")
async def health():
    return {"status": "ok"}

# @router.get("/v1/models")
@router.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    return {"object": "list", "data": [{"id": agent.MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "ragwire"}]}


# @router.get("/v1/models/{model_id}")
@router.get("/v1/models/{model_id}", dependencies=[Depends(verify_api_key)])
async def get_model(model_id: str):
    return {"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "ragwire"}


# @router.post("/v1/chat/completions")
@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(req: ChatRequest):
    messages = [m.model_dump() for m in req.messages]
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    cid, ts = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())

    async def stream():
        yield chunk(cid, ts)
        try:
            async for text in agent.stream(messages):
                yield chunk(cid, ts, content=text)
        except Exception as exc:
            yield chunk(cid, ts, content=f"\n[Error: {exc}]")
        yield chunk(cid, ts, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# @router.post("/upload")
@router.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_documents(files: List[UploadFile] = File(...)):
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in files:
            with open(os.path.join(tmpdir, f.filename), "wb") as out:
                out.write(await f.read())
        stats = rag.ingest_directory(tmpdir)
    return {"message": f"Ingested {stats['chunks_created']} chunks from {stats['processed']} file(s) ({stats['skipped']} skipped).", "stats": stats}
