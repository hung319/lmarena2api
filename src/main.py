import asyncio
import json
import re
import uuid
import time
import secrets
import base64
import mimetypes
import os
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timezone

import uvicorn
import httpx
from dotenv import load_dotenv
from camoufox.async_api import AsyncCamoufox
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from starlette.responses import StreamingResponse

# ============================================================
# CONFIGURATION & ENV
# ============================================================

# Xác định đường dẫn file .env dựa trên vị trí file main.py
# main.py đang ở /src, nên .env sẽ ở ../.env (thư mục cha)
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
MODELS_FILE = BASE_DIR / "models.json"  # Lưu models.json ra ngoài root luôn cho gọn

# Load .env từ đường dẫn cụ thể
load_dotenv(dotenv_path=ENV_PATH)

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
PORT = int(os.getenv("PORT", 8000))
PROXY_URL = os.getenv("PROXY_URL")
MASTER_API_KEY = os.getenv("API_KEY")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

# Global State
# Lưu session chat: { "conversation_id": { ... } }
chat_sessions: Dict[str, dict] = {}
cf_clearance_token = ""

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def uuid7():
    """Generate UUIDv7 compliant with browser implementation."""
    timestamp_ms = int(time.time() * 1000)
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)
    uuid_int = timestamp_ms << 80
    uuid_int |= (0x7000 | rand_a) << 64
    uuid_int |= (0x8000000000000000 | rand_b)
    hex_str = f"{uuid_int:032x}"
    return f"{hex_str[0:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"

def get_models():
    try:
        if MODELS_FILE.exists():
            with open(MODELS_FILE, "r") as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return []

def save_models(models):
    with open(MODELS_FILE, "w") as f:
        json.dump(models, f, indent=2)

def get_request_headers():
    if not AUTH_TOKEN:
        debug_print("❌ AUTH_TOKEN not set in .env")
    
    headers = {
        "Content-Type": "application/json",
        "Cookie": f"cf_clearance={cf_clearance_token}; arena-auth-prod-v1={AUTH_TOKEN}",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    return headers

# --- Auth Middleware ---
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def verify_api_key(key: str = Depends(api_key_header)):
    """Validates the Bearer token against MASTER_API_KEY in .env"""
    if not MASTER_API_KEY:
        return True
        
    if not key or not key.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = key.replace("Bearer ", "").strip()
    if token != MASTER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

# ============================================================
# IMAGE HANDLING
# ============================================================

async def upload_image_to_lmarena(image_data: bytes, mime_type: str, filename: str) -> Optional[tuple]:
    if not image_data: return None
    
    debug_print(f"📤 Uploading image: {filename}")
    request_headers = get_request_headers()
    request_headers.update({
        "Accept": "text/x-component",
        "Content-Type": "text/plain;charset=UTF-8",
        "Next-Action": "70cb393626e05a5f0ce7dcb46977c36c139fa85f91",
        "Referer": "https://lmarena.ai/?mode=direct",
    })

    transport = httpx.AsyncHTTPTransport(proxy=PROXY_URL) if PROXY_URL else None

    async with httpx.AsyncClient(transport=transport, timeout=60.0) as client:
        try:
            # Step 1: Request Upload URL
            resp = await client.post(
                "https://lmarena.ai/?mode=direct",
                headers=request_headers,
                content=json.dumps([filename, mime_type])
            )
            resp.raise_for_status()
            
            upload_url = None
            key = None
            for line in resp.text.strip().split('\n'):
                if line.startswith('1:'):
                    data = json.loads(line[2:])
                    upload_url = data['data']['uploadUrl']
                    key = data['data']['key']
                    break
            
            if not upload_url: return None

            # Step 2: Put to R2
            await client.put(
                upload_url,
                content=image_data,
                headers={"Content-Type": mime_type}
            )

            # Step 3: Get Signed URL
            request_headers["Next-Action"] = "6064c365792a3eaf40a60a874b327fe031ea6f22d7"
            resp = await client.post(
                "https://lmarena.ai/?mode=direct",
                headers=request_headers,
                content=json.dumps([key])
            )
            
            download_url = None
            for line in resp.text.strip().split('\n'):
                if line.startswith('1:'):
                    data = json.loads(line[2:])
                    download_url = data['data']['url']
                    break
            
            return (key, download_url) if download_url else None

        except Exception as e:
            debug_print(f"❌ Image upload failed: {e}")
            return None

async def process_message_content(content, model_capabilities: dict) -> tuple[str, List[dict]]:
    supports_images = model_capabilities.get('inputCapabilities', {}).get('image', False)
    
    if isinstance(content, str):
        return content, []
    
    if isinstance(content, list):
        text_parts = []
        attachments = []
        
        for part in content:
            if isinstance(part, dict):
                if part.get('type') == 'text':
                    text_parts.append(part.get('text', ''))
                elif part.get('type') == 'image_url' and supports_images:
                    url = part.get('image_url', {}).get('url', '') if isinstance(part.get('image_url'), dict) else part.get('image_url')
                    
                    if url.startswith('data:'):
                        try:
                            header, data = url.split(',', 1)
                            mime_type = header.split(';')[0].split(':')[1]
                            image_data = base64.b64decode(data)
                            ext = mimetypes.guess_extension(mime_type) or '.png'
                            filename = f"upload-{uuid.uuid4()}{ext}"
                            
                            res = await upload_image_to_lmarena(image_data, mime_type, filename)
                            if res:
                                attachments.append({"name": res[0], "contentType": mime_type, "url": res[1]})
                        except Exception as e:
                            debug_print(f"Failed to process base64 image: {e}")

        return '\n'.join(text_parts).strip(), attachments
    return str(content), []

# ============================================================
# APP SETUP & BACKGROUND TASKS
# ============================================================

app = FastAPI(title="LMArena Headless Bridge")

async def get_initial_data():
    """Fetch Cloudflare clearance token and model list."""
    global cf_clearance_token
    print("🔄 Initializing: Fetching models and Cloudflare token...")
    
    try:
        proxy_config = {"server": PROXY_URL} if PROXY_URL else None

        async with AsyncCamoufox(headless=True, proxy=proxy_config) as browser:
            page = await browser.new_page()
            
            print("➡️  Navigating to lmarena.ai...")
            await page.goto("https://lmarena.ai/", wait_until="domcontentloaded")

            try:
                await page.wait_for_function(
                    "() => document.title.indexOf('Just a moment...') === -1", 
                    timeout=60000
                )
            except Exception:
                print("⚠️  Cloudflare challenge timeout/fail.")

            await asyncio.sleep(5)

            cookies = await page.context.cookies()
            cf_cookie = next((c for c in cookies if c["name"] == "cf_clearance"), None)
            
            if cf_cookie:
                cf_clearance_token = cf_cookie["value"]
                print(f"✅ CF Clearance Token acquired: {cf_clearance_token[:10]}...")
            else:
                print("⚠️  cf_clearance cookie not found.")

            try:
                body = await page.content()
                match = re.search(r'{\\"initialModels\\":(\[.*?\]),\\"initialModel[A-Z]Id', body, re.DOTALL)
                if match:
                    models_json = match.group(1).encode().decode('unicode_escape')
                    models = json.loads(models_json)
                    save_models(models)
                    print(f"✅ Cached {len(models)} models.")
            except Exception as e:
                print(f"❌ Error parsing models: {e}")

    except Exception as e:
        print(f"❌ Browser automation error: {e}")

async def periodic_refresh_task():
    while True:
        await asyncio.sleep(1800)
        await get_initial_data()

@app.on_event("startup")
async def startup_event():
    if not AUTH_TOKEN:
        print("⚠️  WARNING: AUTH_TOKEN is not set in .env!")
    if not MASTER_API_KEY:
        print("⚠️  WARNING: API_KEY is not set in .env! API is open to the public.")
        
    asyncio.create_task(get_initial_data())
    asyncio.create_task(periodic_refresh_task())

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "active",
        "proxy": bool(PROXY_URL),
        "auth_token_set": bool(AUTH_TOKEN),
        "cf_token_acquired": bool(cf_clearance_token),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    models = get_models()
    data = []
    for m in models:
        caps = m.get('capabilities', {}).get('outputCapabilities', {})
        if caps.get('text') and m.get('organization'):
             data.append({
                "id": m.get("publicName"),
                "object": "model",
                "created": int(time.time()),
                "owned_by": m.get("organization", "lmarena")
            })
    return {"object": "list", "data": data}

@app.post("/api/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    model_name = body.get("model")
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    if not model_name or not messages:
        raise HTTPException(status_code=400, detail="Missing model or messages")

    models = get_models()
    target_model = next((m for m in models if m.get("publicName") == model_name), None)
    
    if not target_model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    model_id = target_model.get("id")
    capabilities = target_model.get("capabilities", {})

    # Prepare Prompt
    system_prompt = ""
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    if sys_msgs:
        system_prompt = "\n\n".join([m.get("content", "") for m in sys_msgs])

    last_msg = messages[-1].get("content", "")
    prompt, attachments = await process_message_content(last_msg, capabilities)
    
    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"

    # Conversation Hash
    import hashlib
    conv_key_hash = hashlib.sha256(f"{MASTER_API_KEY}_{model_name}_{str(messages[0])[:50]}".encode()).hexdigest()[:16]
    session = chat_sessions.get(conv_key_hash)
    
    if not session:
        session_id = str(uuid7())
        user_msg_id = str(uuid7())
        model_msg_id = str(uuid7())
        is_new = True
        url = "https://lmarena.ai/nextjs-api/stream/create-evaluation"
    else:
        session_id = session["conversation_id"]
        user_msg_id = str(uuid7())
        model_msg_id = str(uuid7())
        is_new = False
        url = f"https://lmarena.ai/nextjs-api/stream/post-to-evaluation/{session_id}"

    payload = {
        "id": session_id,
        "mode": "direct",
        "modelAId": model_id,
        "userMessageId": user_msg_id,
        "modelAMessageId": model_msg_id,
        "userMessage": {
            "content": prompt,
            "experimental_attachments": attachments
        },
        "modality": "chat"
    }

    headers = get_request_headers()
    transport = httpx.AsyncHTTPTransport(proxy=PROXY_URL) if PROXY_URL else None

    async def stream_generator():
        full_text = ""
        chunk_id = f"chatcmpl-{uuid.uuid4()}"
        
        async with httpx.AsyncClient(transport=transport, timeout=120.0) as client:
            try:
                async with client.stream('POST', url, json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        err_txt = await resp.read()
                        yield f"data: {json.dumps({'error': {'message': f'Upstream Error: {resp.status_code}', 'details': str(err_txt)}}})}\n\n"
                        return

                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line: continue
                        
                        content_delta = None
                        finish_reason = None

                        if line.startswith("a0:"):
                            try:
                                content_delta = json.loads(line[3:])
                                full_text += content_delta
                            except: pass
                        elif line.startswith("ad:"):
                            try:
                                meta = json.loads(line[3:])
                                finish_reason = meta.get("finishReason", "stop")
                            except: pass
                        
                        if content_delta:
                            chunk = {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_name,
                                "choices": [{"index": 0, "delta": {"content": content_delta}, "finish_reason": None}]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                        
                        if finish_reason:
                            chunk = {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_name,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                    
                    yield "data: [DONE]\n\n"
                    
                    if is_new:
                        chat_sessions[conv_key_hash] = {"conversation_id": session_id}

            except Exception as e:
                debug_print(f"Stream Error: {e}")
                yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"

    if stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        response_content = ""
        finish_reason = "stop"
        async for chunk_str in stream_generator():
            if chunk_str.startswith("data: [DONE]"): break
            if chunk_str.startswith("data: "):
                try:
                    chunk_json = json.loads(chunk_str[6:])
                    if "error" in chunk_json:
                        raise HTTPException(status_code=500, detail=chunk_json["error"]["message"])
                    delta = chunk_json["choices"][0]["delta"].get("content", "")
                    response_content += delta
                except: pass
        
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_content},
                "finish_reason": finish_reason
            }],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(response_content), "total_tokens": 0}
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
