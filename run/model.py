from flask import Flask, request, Response, jsonify
import json
import threading
import time

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import (
    ExLlamaV2DynamicGenerator,
    ExLlamaV2DynamicJob,
    ExLlamaV2Sampler,
)

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_DIR       = "./Zeta-Qwen2.5-Coder-7B-EXL2"
MAX_SEQ_LEN     = 8192
MAX_BATCH_SIZE  = 4        # concurrent requests
CACHE_QUANT     = True     # Q4 cache saves ~60% VRAM
PORT            = 5000
# ----------------------

# ── Load model ──────────────────────────────────────────────────

print("Loading model...")
t0 = time.time()

config = ExLlamaV2Config(MODEL_DIR)
config.max_seq_len = MAX_SEQ_LEN
config.max_batch_size = MAX_BATCH_SIZE

model = ExLlamaV2(config)
model.load()

tokenizer = ExLlamaV2Tokenizer(config)

CacheClass = ExLlamaV2Cache_Q4 if CACHE_QUANT else __import__(
    "exllamav2", fromlist=["ExLlamaV2Cache"]
).ExLlamaV2Cache
cache = CacheClass(model, max_seq_len=MAX_SEQ_LEN, lazy=True)

generator = ExLlamaV2DynamicGenerator(
    model=model,
    cache=cache,
    tokenizer=tokenizer,
)

print(f"Model loaded in {time.time() - t0:.1f}s")

# Lock to serialize job submission (DynamicGenerator handles
# concurrent *execution* but job creation should be serialized)
_gen_lock = threading.Lock()


# ── Helpers ─────────────────────────────────────────────────────

def make_sampler_settings(data: dict) -> ExLlamaV2Sampler.Settings:
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = data.get("temperature", 0.0)
    settings.top_p = data.get("top_p", 0.9)
    settings.top_k = data.get("top_k", 50)
    settings.token_repetition_penalty = data.get("repetition_penalty", 1.0)
    if settings.temperature == 0.0:
        # Greedy — disable sampling entirely for speed
        settings.temperature = 1.0
        settings.top_k = 1
        settings.top_p = 0.0
    return settings


def encode_stop_conditions(stop_list):
    """Return list of token-id and string stop conditions."""
    conditions = [tokenizer.eos_token_id]
    if stop_list:
        for s in stop_list:
            if isinstance(s, str) and s:
                conditions.append(s)
    return conditions


# ── Endpoints ───────────────────────────────────────────────────

@app.route("/v1/completions", methods=["POST"])
def completions():
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = min(data.get("max_tokens", 480), MAX_SEQ_LEN)
    stream = data.get("stream", False)

    settings = make_sampler_settings(data)
    stop_conditions = encode_stop_conditions(data.get("stop"))

    input_ids = tokenizer.encode(prompt, encode_special_tokens=True)
    prompt_tokens = input_ids.shape[-1]

    if stream:
        return Response(_stream(prompt, input_ids, settings,
                                stop_conditions, max_tokens, prompt_tokens),
                        mimetype="text/event-stream")

    # Non-streaming
    output_text, comp_tokens = _generate_full(
        prompt, input_ids, settings, stop_conditions, max_tokens
    )
    return jsonify({
        "choices": [{"text": output_text, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
        },
    })


def _generate_full(prompt, input_ids, settings, stop_conditions, max_tokens):
    """Blocking full generation. Returns (text, token_count)."""
    collected = []
    token_count = 0

    with _gen_lock:
        job = ExLlamaV2DynamicJob(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            stop_conditions=stop_conditions,
            gen_settings=settings,
        )
        generator.enqueue(job)

    while True:
        results = generator.iterate()
        for result in results:
            if result["stage"] == "streaming":
                text = result.get("text", "")
                if text:
                    collected.append(text)
                    token_count += 1
            elif result["stage"] == "eos":
                return "".join(collected), token_count

    return "".join(collected), token_count


def _stream(prompt, input_ids, settings, stop_conditions, max_tokens, prompt_tokens):
    """SSE streaming generator."""
    with _gen_lock:
        job = ExLlamaV2DynamicJob(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            stop_conditions=stop_conditions,
            gen_settings=settings,
        )
        generator.enqueue(job)

    while True:
        results = generator.iterate()
        for result in results:
            if result["stage"] == "streaming":
                text = result.get("text", "")
                if text:
                    payload = json.dumps({"choices": [{"text": text}]})
                    yield f"data: {payload}\n\n"
            elif result["stage"] == "eos":
                yield "data: [DONE]\n\n"
                return


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_DIR, "max_seq_len": MAX_SEQ_LEN})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)