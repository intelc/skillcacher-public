"""Tiny FastAPI echo backend that mimics vLLM's OpenAI-compatible endpoints
for offline proxy testing. Responds to /v1/chat/completions only."""
import argparse
import json
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn


app = FastAPI()


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    last = body["messages"][-1]
    text = last.get("content") or "echo"
    if isinstance(text, list):
        text = " ".join(b.get("text", "") for b in text if isinstance(b, dict) and b.get("type") == "text")
    if body.get("stream"):
        async def gen():
            cid = "chatcmpl-" + uuid.uuid4().hex[:8]
            for tok in [text[:1], text[1:]]:
                ev = {"id": cid, "choices": [{"delta": {"content": tok}, "finish_reason": None}]}
                yield f"data: {json.dumps(ev)}\n\n".encode()
            ev = {"id": cid, "choices": [{"delta": {}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": len(text)}}
            yield f"data: {json.dumps(ev)}\n\n".encode()
            yield b"data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return JSONResponse({
        "id": "chatcmpl-" + uuid.uuid4().hex[:8],
        "model": body.get("model", "echo"),
        "choices": [{"message": {"role": "assistant", "content": f"echo: {text}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": len(text) + 6},
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")


if __name__ == "__main__":
    main()
