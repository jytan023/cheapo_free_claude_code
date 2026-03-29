import os, json, httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import re

load_dotenv()

app = FastAPI()

# --- Provider config ---
PROVIDER = os.getenv("PROVIDER", "nvidia_nim")
MODEL = os.getenv("MODEL", "mistralai/devstral-2-123b-instruct-2512")

PROVIDERS = {
    "nvidia_nim": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": os.getenv("NVIDIA_API_KEY"),
        "extra_headers": {},
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "extra_headers": {
            "HTTP-Referer": "http://localhost:8082",
            "X-Title": "claude-code-proxy"
        },
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.getenv("GROQ_API_KEY"),
        "extra_headers": {},
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "extra_headers": {},
    },
    "zai": {
        "base_url": "https://api.z.ai/api/paas/v4",
        "api_key": os.getenv("ZAI_API_KEY"),
        "extra_headers": {},
    },
}

provider = PROVIDERS.get(PROVIDER)
if not provider:
    raise ValueError(f"Unknown provider: {PROVIDER}. Choose from: {list(PROVIDERS.keys())}")
if not provider["api_key"]:
    raise ValueError(f"Missing API key for provider: {PROVIDER}")

BASE_URL = provider["base_url"].rstrip("/")  # ensure no trailing slash
API_KEY = provider["api_key"]
EXTRA_HEADERS = provider["extra_headers"]

print(f"✅ Provider: {PROVIDER}")
print(f"✅ Model: {MODEL}")


# --- Helpers ---
def clean_delta(text):
    text = re.sub(r'<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|tool_call_begin\|>.*?<\|tool_call_end\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|[^|]+\|>', '', text)
    return text

def anthropic_to_openai(body):
    messages = []
    if body.get("system"):
        system = body["system"]
        if isinstance(system, list):
            system = " ".join(b.get("text", "") for b in system if b.get("type") == "text")
        messages.append({"role": "system", "content": system})
    for m in body.get("messages", []):
        content = m["content"]
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
        messages.append({"role": m["role"], "content": content})
    return {
        "model": MODEL,
        "messages": messages,
        "max_tokens": body.get("max_tokens", 8192),
        "stream": body.get("stream", False),
    }


# --- Routes ---
@app.get("/v1/models")
async def models():
    return JSONResponse({"data": [{"id": MODEL, "object": "model"}]})

@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    oai_payload = anthropic_to_openai(body)
    headers = {"Authorization": f"Bearer {API_KEY}", **EXTRA_HEADERS}
    
    # Debug
    print(f"🔵 Provider: {PROVIDER}")
    print(f"🔵 Model: {MODEL}")
    print(f"🔵 Base URL: {BASE_URL}")
    print(f"🔵 Streaming: {oai_payload['stream']}")
    print(f"🔵 API Key starts with: {API_KEY[:10] if API_KEY else 'NONE'}")

    # Non-streaming
    if not oai_payload["stream"]:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{BASE_URL}/chat/completions", json=oai_payload, headers=headers)
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            text = clean_delta(text)
            return {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "model": body.get("model", MODEL),
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }

    # Streaming
    async def stream():
        yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':'msg_1','type':'message','role':'assistant','content':[],'model':MODEL,'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"
        yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"
        yield f"event: ping\ndata: {json.dumps({'type':'ping'})}\n\n"
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", f"{BASE_URL}/chat/completions", json=oai_payload, headers=headers) as r:
                    async for line in r.aiter_lines():
                        print(f"📥 Raw line: {line}")  # add this
                        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                            continue
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk["choices"][0].get("delta", {}).get("content") or ""
                            delta = clean_delta(delta)
                            if delta:
                                yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':delta}})}\n\n"
                        except Exception:
                            continue
        except Exception as e:
            yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':f'[proxy error: {str(e)}]'}})}\n\n"
        yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
        yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':0}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")