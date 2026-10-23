import glob
import hashlib
import json
import os
from datetime import datetime, timedelta
from json import JSONDecodeError

from flask import Blueprint, request, jsonify, stream_with_context, Response

from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

from backend.eval.metrics import estimate_cost_usd
from backend.rag.agent import process_input_agent, process_input_agent_sync
from backend.rag.pipeline import process_input, process_input_streaming
from backend.rag.rag_utils import generate_questions
from backend.utils.utils import compute_upload_hash, extract_title_abstract_fulltext

VALID_MODES = ("pipeline", "agent")


def _resolve_mode(raw):
    mode = (raw or "pipeline").strip().lower()
    return mode if mode in VALID_MODES else "pipeline"


def _build_usage(counter: TokenCountingHandler, model: str) -> dict:
    """Pack TokenCountingHandler totals into the SSE-friendly dict."""
    prompt = int(counter.prompt_llm_token_count)
    completion = int(counter.completion_llm_token_count)
    return {
        "model": model,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cost_usd": estimate_cost_usd(model, prompt, completion),
    }

api_bp = Blueprint('api', __name__)

CACHE_DIR = "data/cache"
TEMP_DIR = "data/temp"
CACHE_EXPIRY_DAYS = 20
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


def _parse_final_payload(message):
    """Parse an SSE chunk and return the final payload dict if it carries a summary.

    The streaming pipeline emits ``data: {"update": "..."}`` for progress and
    ``data: {"summary": "...", "profiles": [...]}`` for the final result. We
    detect the final by structural shape (presence of the ``summary`` key)
    rather than substring match, so progress messages that mention "summary"
    don't false-trigger.
    """
    text = message.decode("utf-8", errors="replace") if isinstance(message, (bytes, bytearray)) else message
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        body = line[len("data:"):].strip()
        if not body:
            continue
        try:
            parsed = json.loads(body)
        except JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "summary" in parsed:
            return parsed
    return None

@api_bp.route('/process_input', methods=['POST'])
def process():
    prompt = request.form.get("prompt")
    model = request.form.get("model")
    online = request.form.get("online", False)
    no_cache = request.form.get("no_cache", "false").lower() == "true"
    mode = _resolve_mode(request.form.get("mode"))

    # Collect and hash relevant request data (mode is part of the key so
    # pipeline and agent results don't collide for the same prompt).
    cache_input = {
        "prompt": prompt,
        "model": model,
        "online": str(online),
        "mode": mode,
    }

    # Handle file upload
    paper = {}
    pdf_file = request.files.get("pdf_file")
    if pdf_file:
        file_hash = compute_upload_hash(pdf_file)
        path = f"data/temp/{file_hash}.pdf"
        pdf_file.save(path)

        title, abstract, fulltext = extract_title_abstract_fulltext(path)
        paper = {"title": title, "abstract": abstract, "fulltext": fulltext}
        cache_input["file_hash"] = file_hash
    else:
        cache_input["file_hash"] = None

    # Generate request hash and cache path
    request_hash = hashlib.md5(json.dumps(cache_input, sort_keys=True).encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{request_hash}.json")

    # Try to serve from cache
    if not no_cache and os.path.exists(cache_path):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - cache_time < timedelta(days=CACHE_EXPIRY_DAYS):
            with open(cache_path, "r") as f:
                cached_data = json.load(f)
            return jsonify(cached_data)

    token_counter = TokenCountingHandler()
    saved_cb_manager = Settings.callback_manager
    Settings.callback_manager = CallbackManager([token_counter])
    try:
        if mode == "agent":
            summary, profiles = process_input_agent_sync(prompt, paper, online, model)
        else:
            summary, profiles = process_input(prompt, paper, online, model)
    finally:
        Settings.callback_manager = saved_cb_manager

    usage = _build_usage(token_counter, model)

    output = {
        "summary": (summary or "").strip(),
        "profiles": profiles or [],
        "input": {
            "prompt": prompt,
            "model": model,
            "online": online,
            "mode": mode,
            "file_hash": cache_input["file_hash"],
        },
        "cache_timestamp": None,
        "usage": usage,
    }

    # Save to cache (excluding the timestamp)
    with open(cache_path, "w") as f:
        json.dump({k: v for k, v in output.items() if k != "cache_timestamp"}, f)

    return jsonify(output)

@api_bp.route("/generate_query", methods=['POST'])
def generate_query():
    data = request.get_json() or {}
    prompt = data.get("prompt")
    model = data.get("model")
    online = data.get("online", False)

    if not prompt or not model:
        return jsonify({"error": "Missing required fields"}), 400

    classification = generate_questions(prompt, None, llm_model=model)

    return jsonify({
        "classification": classification,
        "input": {
            "prompt": prompt,
            "model": model,
            "online": online,
        },
    })

@api_bp.route('/process_input_stream', methods=['POST'])
def process_stream():
    prompt = request.form.get("prompt")
    model = request.form.get("model")
    online = request.form.get("online", False)
    no_cache = request.form.get("no_cache", "false").lower() == "true"
    mode = _resolve_mode(request.form.get("mode"))

    # Prepare cache key (mode-aware so pipeline/agent runs don't collide).
    cache_input = {
        "prompt": prompt,
        "model": model,
        "online": str(online),
        "mode": mode,
    }

    # Handle uploaded PDF
    paper = {}
    pdf_file = request.files.get("pdf_file")
    if pdf_file:
        file_hash = compute_upload_hash(pdf_file)
        path = f"data/temp/{file_hash}.pdf"
        pdf_file.save(path)
        title, abstract, fulltext = extract_title_abstract_fulltext(path)
        paper = {"title": title, "abstract": abstract, "fulltext": fulltext}
        cache_input["file_hash"] = file_hash
    else:
        cache_input["file_hash"] = None

    # Caching logic
    request_hash = hashlib.md5(json.dumps(cache_input, sort_keys=True).encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{request_hash}.json")

    def event_stream():
        # Serve from cache if valid
        if not no_cache and os.path.exists(cache_path):
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - cache_time < timedelta(days=CACHE_EXPIRY_DAYS):
                with open(cache_path, "r") as f:
                    cached_data = json.load(f)
                yield f"data: {json.dumps({'cached': True, **cached_data})}\n\n"
                yield "event: done\ndata: {}\n\n"
                return

        def stream_fn(message):
            return [f"data: {json.dumps({'update': message})}\n\n"]

        # Attach a TokenCountingHandler so we can report the run's token
        # usage + estimated cost at the end. Restore the prior callback
        # manager when we're done so the next request gets its own counter.
        token_counter = TokenCountingHandler()
        saved_cb_manager = Settings.callback_manager
        Settings.callback_manager = CallbackManager([token_counter])

        if mode == "agent":
            generator = process_input_agent(
                prompt, paper=paper, online=online, model=model, stream_fn=stream_fn
            )
        else:
            generator = process_input_streaming(
                prompt, paper=paper, online=online, model=model, stream_fn=stream_fn
            )

        summary, profiles = None, None
        try:
            for message in generator:
                final = _parse_final_payload(message)
                if final is not None:
                    summary = final.get("summary")
                    profiles = final.get("profiles")
                yield message  # pass through (str or bytes); Flask handles both
        finally:
            Settings.callback_manager = saved_cb_manager

        usage = _build_usage(token_counter, model)
        yield f"data: {json.dumps({'usage': usage})}\n\n"

        # Save to cache if successful
        if summary is not None:
            output = {
                "summary": summary.strip(),
                "profiles": profiles,
                "input": cache_input,
                "cache_timestamp": datetime.now().isoformat(),
                "usage": usage,
            }
            with open(cache_path, "w") as f:
                json.dump({k: v for k, v in output.items()}, f)

        yield "event: done\ndata: {}\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

@api_bp.route('/feedback', methods=['POST'])
def add_feedback():
    data = request.get_json()

    prompt = data.get("prompt")
    model = data.get("model")
    file_hash = data.get("file_hash")
    rating = data.get("rating")
    comments = data.get("comments")

    if not all([prompt, model]):
        return jsonify({"error": "Missing required fields"}), 400

    # Recompute the cache key
    cache_input = {
        "prompt": prompt,
        "model": model,
        "online": "true",  # assuming online mode is always used
        "file_hash": file_hash,
    }

    request_hash = hashlib.md5(json.dumps(cache_input, sort_keys=True).encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{request_hash}.json")

    if not os.path.exists(cache_path):
        return jsonify({"error": "No cached entry found for this input."}), 404

    # Load cache and update feedback field
    with open(cache_path, "r") as f:
        cached_data = json.load(f)

    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "rating": rating,
        "comments": comments
    }

    if "feedback" not in cached_data:
        cached_data["feedback"] = []

    cached_data["feedback"].append(feedback_entry)

    with open(cache_path, "w") as f:
        json.dump(cached_data, f)

    return jsonify({"message": "Feedback saved successfully."}), 200


@api_bp.route('/recent_caches', methods=['GET'])
def recent_caches():
    cache_files = sorted(
        glob.glob(os.path.join(CACHE_DIR, "*.json")),
        key=os.path.getmtime,
        reverse=True
    )[:15]  # Get up to 5 most recent

    recent = []
    for path in cache_files:
        try:
            with open(path, "r") as f:
                data = json.load(f)
                if data.get("input", {}).get("file_hash") is not None:
                    continue
                recent.append({
                    "prompt": data.get("input", {}).get("prompt", "N/A"),
                    "model": data.get("input", {}).get("model", "N/A"),
                    "file_hash": data.get("input", {}).get("file_hash"),
                    "summary": data.get("summary", "")[:250],  # preview
                })
        except Exception as e:
            continue  # skip broken files

    return jsonify(recent[:5])
