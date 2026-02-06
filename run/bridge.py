import socket
import threading
import time
import sys
import json

# --- CONFIGURATION ---
# Local ExLlamaV2 Flask server
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 5000
# Port Zed connects to (pretend to be Ollama)
PROXY_PORT   = 11434
DEBOUNCE_MS  = 300
BUFFER_SIZE  = 65536
DEBOUNCE_PATHS = {"/api/generate", "/api/chat"}
MODEL_NAME   = "Zeta-Qwen2.5-Coder-7B:latest"
# ---------------------

_req_counter = 0
_req_lock = threading.Lock()

def next_req_id():
    global _req_counter
    with _req_lock:
        _req_counter += 1
        return _req_counter


def log(msg):
    ts = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{ts}] {msg}")


def make_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s


def _send_empty_ok(sock):
    try:
        sock.sendall(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 2\r\n"
            b"Connection: close\r\n\r\n{}"
        )
    except Exception:
        pass
    finally:
        try: sock.close()
        except Exception: pass


def _parse_request_line(raw: bytes):
    try:
        first_line = raw.split(b"\r\n", 1)[0].decode("utf-8", errors="ignore")
        parts = first_line.split()
        if len(parts) >= 2:
            return parts[0], parts[1]
    except Exception:
        pass
    return None, None


def read_full_client_request(client_socket, first_chunk):
    buf = first_chunk
    if b"\r\n\r\n" in buf:
        header_end = buf.index(b"\r\n\r\n") + 4
        header_text = buf[:header_end].decode("utf-8", errors="ignore")
        body_so_far = buf[header_end:]
        content_length = None
        for line in header_text.split("\r\n"):
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":", 1)[1].strip())
                break
        if content_length is not None:
            remaining = content_length - len(body_so_far)
            while remaining > 0:
                chunk = client_socket.recv(min(BUFFER_SIZE, remaining))
                if not chunk:
                    break
                buf += chunk
                remaining -= len(chunk)
    return buf


# ── Ollama → OpenAI translation ────────────────────────────────

def ollama_to_openai(ollama_body: dict) -> dict:
    """Convert Ollama /api/generate request to OpenAI /v1/completions format."""
    openai_req = {
        "prompt": ollama_body.get("prompt", ""),
        "stream": ollama_body.get("stream", False),
    }

    opts = ollama_body.get("options", {})
    openai_req["max_tokens"] = opts.get("num_predict", 480)
    openai_req["temperature"] = opts.get("temperature", 0.0)

    if "stop" in opts:
        openai_req["stop"] = opts["stop"]

    if opts.get("top_p") is not None:
        openai_req["top_p"] = opts["top_p"]

    return openai_req


def build_http_request(host: str, port: int, path: str, body: bytes) -> bytes:
    """Build a raw HTTP POST request."""
    return (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode() + body


# ── Response handlers ───────────────────────────────────────────

def handle_streaming(remote, client_socket, cancel_event, rid, model_name):
    """
    Read OpenAI SSE stream from Flask backend, translate each chunk
    to Ollama NDJSON streaming format, and relay to Zed.
    """
    start = time.time()
    first_byte = None
    total_tokens = 0

    try:
        # Read HTTP headers from backend
        buf = b""
        while b"\r\n\r\n" not in buf:
            if cancel_event.is_set():
                return
            chunk = remote.recv(BUFFER_SIZE)
            if not chunk:
                return
            if first_byte is None:
                first_byte = time.time()
                log(f"  #{rid} TTFT: {(first_byte - start) * 1000:.0f}ms")
            buf += chunk

        header_end = buf.index(b"\r\n\r\n") + 4
        leftover = buf[header_end:]

        # Send Ollama-style chunked response headers to Zed
        client_socket.sendall(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/x-ndjson\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
        )

        def send_chunk(data: bytes):
            """Send one HTTP chunked-encoding frame."""
            client_socket.sendall(f"{len(data):x}\r\n".encode() + data + b"\r\n")

        def send_ollama_token(token: str, done: bool = False):
            nonlocal total_tokens
            obj = {
                "model": model_name,
                "response": token,
                "done": done,
            }
            if done:
                obj["eval_count"] = total_tokens
            send_chunk(json.dumps(obj).encode() + b"\n")
            if not done:
                total_tokens += 1

        # Process SSE stream
        sse_buf = leftover
        while not cancel_event.is_set():
            # Process complete lines
            while b"\n" in sse_buf:
                line, sse_buf = sse_buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                if line.startswith(b"data: "):
                    payload = line[6:].strip()
                    if payload == b"[DONE]":
                        send_ollama_token("", done=True)
                        # Terminal chunk
                        client_socket.sendall(b"0\r\n\r\n")
                        elapsed = time.time() - start
                        log(f"  #{rid} DONE: {total_tokens} tokens in {elapsed:.2f}s")
                        return
                    try:
                        data = json.loads(payload)
                        text = data.get("choices", [{}])[0].get("text", "")
                        if text:
                            send_ollama_token(text)
                    except json.JSONDecodeError:
                        pass

            # Read more from backend
            chunk = remote.recv(BUFFER_SIZE)
            if not chunk:
                break
            sse_buf += chunk

        # If we fell through without [DONE], send final
        if not cancel_event.is_set():
            send_ollama_token("", done=True)
            client_socket.sendall(b"0\r\n\r\n")

    except Exception as e:
        elapsed = time.time() - start
        if cancel_event.is_set():
            log(f"  #{rid} CANCELLED after {elapsed:.2f}s")
        else:
            log(f"  #{rid} ERROR: {e} after {elapsed:.2f}s")


def handle_non_streaming(remote, client_socket, cancel_event, rid, model_name):
    """
    Read OpenAI JSON response from Flask backend, translate to
    Ollama non-streaming format, and send to Zed.
    """
    start = time.time()

    try:
        buf = b""
        while True:
            if cancel_event.is_set():
                return
            chunk = remote.recv(BUFFER_SIZE)
            if not chunk:
                break
            buf += chunk

        if b"\r\n\r\n" not in buf:
            return

        body = buf.split(b"\r\n\r\n", 1)[1]
        data = json.loads(body)
        text = data.get("choices", [{}])[0].get("text", "")

        ollama_resp = {
            "model": model_name,
            "response": text,
            "done": True,
        }

        resp_body = json.dumps(ollama_resp).encode()
        http_resp = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(resp_body)).encode() + b"\r\n"
            b"Connection: close\r\n\r\n" + resp_body
        )
        client_socket.sendall(http_resp)

        elapsed = time.time() - start
        log(f"  #{rid} DONE: {len(text)} chars in {elapsed:.2f}s")

    except Exception as e:
        elapsed = time.time() - start
        if cancel_event.is_set():
            log(f"  #{rid} CANCELLED after {elapsed:.2f}s")
        else:
            log(f"  #{rid} ERROR: {e} after {elapsed:.2f}s")


# ── Fake Ollama endpoints ───────────────────────────────────────

def handle_tags(client_socket):
    """Respond to GET /api/tags with a fake model list so Zed sees the model."""
    body = json.dumps({
        "models": [{
            "name": MODEL_NAME,
            "model": MODEL_NAME,
            "size": 0,
            "digest": "0000000000000000",
            "details": {
                "format": "gguf",
                "family": "qwen2",
                "parameter_size": "7B",
                "quantization_level": "Q4_K_M",
            },
        }]
    }).encode()
    client_socket.sendall(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"Connection: close\r\n\r\n" + body
    )
    client_socket.close()


def handle_show(client_socket, ollama_body):
    """Respond to POST /api/show with minimal model info."""
    body = json.dumps({
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": {
            "format": "gguf",
            "family": "qwen2",
            "parameter_size": "7B",
            "quantization_level": "Q4_K_M",
        },
    }).encode()
    client_socket.sendall(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"Connection: close\r\n\r\n" + body
    )
    client_socket.close()


def handle_version(client_socket):
    """Respond to GET /api/version."""
    body = json.dumps({"version": "0.6.2"}).encode()
    client_socket.sendall(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"Connection: close\r\n\r\n" + body
    )
    client_socket.close()


def handle_root(client_socket):
    """Respond to GET / (Ollama health check)."""
    body = b"Ollama is running"
    client_socket.sendall(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/plain\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"Connection: close\r\n\r\n" + body
    )
    client_socket.close()


# ── Gate: Debounce + In-Flight Cancellation ─────────────────────

class _RequestGate:
    def __init__(self, delay_sec):
        self.delay = delay_sec
        self.lock = threading.Lock()
        self.pending: dict[str, dict] = {}
        self.inflight: dict[str, dict] = {}
        self.seq = 0

    def submit(self, key, client_socket, request_body, stream):
        with self.lock:
            self.seq += 1
            my_seq = self.seq

            # Cancel pending debounce
            prev = self.pending.get(key)
            if prev is not None:
                prev["timer"].cancel()
                _send_empty_ok(prev["client_socket"])
                log(f"  DEBOUNCE: cancelled pending #{prev.get('rid', '?')}")

            # Abort in-flight
            flight = self.inflight.get(key)
            if flight is not None:
                flight["cancel_event"].set()
                rsock = flight.get("remote_socket")
                if rsock:
                    try: rsock.close()
                    except Exception: pass
                    flight["remote_socket"] = None
                _send_empty_ok(flight["client_socket"])
                log(f"  CANCEL: aborted in-flight #{flight.get('rid', '?')}")
                del self.inflight[key]

            rid = next_req_id()
            timer = threading.Timer(
                self.delay, self._fire,
                args=(key, my_seq, rid, client_socket, request_body, stream)
            )
            self.pending[key] = {
                "timer": timer,
                "client_socket": client_socket,
                "seq": my_seq,
                "rid": rid,
            }
            timer.start()

    def _fire(self, key, seq, rid, client_socket, request_body, stream):
        with self.lock:
            entry = self.pending.get(key)
            if entry is None or entry["seq"] != seq:
                return
            del self.pending[key]

            cancel_event = threading.Event()
            self.inflight[key] = {
                "cancel_event": cancel_event,
                "client_socket": client_socket,
                "remote_socket": None,
                "rid": rid,
            }

        log(f"  DEBOUNCE: forwarding #{rid} (waited {self.delay * 1000:.0f}ms)")
        self._forward(key, rid, client_socket, request_body, stream, cancel_event)

    def _forward(self, key, rid, client_socket, request_body, stream, cancel_event):
        if cancel_event.is_set():
            return

        # Connect to Flask backend
        try:
            remote = make_socket()
            remote.settimeout(120)
            remote.connect((BACKEND_HOST, BACKEND_PORT))
        except Exception as e:
            log(f"  #{rid} ERROR: backend connect failed — {e}")
            with self.lock:
                cur = self.inflight.get(key)
                if cur and cur["rid"] == rid:
                    del self.inflight[key]
            client_socket.close()
            return

        if cancel_event.is_set():
            remote.close()
            with self.lock:
                cur = self.inflight.get(key)
                if cur and cur["rid"] == rid:
                    del self.inflight[key]
            return

        with self.lock:
            cur = self.inflight.get(key)
            if cur and cur["rid"] == rid:
                cur["remote_socket"] = remote
            else:
                remote.close()
                return

        # Send translated request to Flask
        http_req = build_http_request(
            BACKEND_HOST, BACKEND_PORT, "/v1/completions", request_body
        )
        log(f"  #{rid} → POST /v1/completions ({len(request_body)}B, stream={stream})")

        try:
            remote.sendall(http_req)
        except Exception as e:
            log(f"  #{rid} send error: {e}")
            remote.close()
            with self.lock:
                cur = self.inflight.get(key)
                if cur and cur["rid"] == rid:
                    del self.inflight[key]
            client_socket.close()
            return

        # Handle response
        if stream:
            handle_streaming(remote, client_socket, cancel_event, rid, MODEL_NAME)
        else:
            handle_non_streaming(remote, client_socket, cancel_event, rid, MODEL_NAME)

        # Cleanup
        with self.lock:
            cur = self.inflight.get(key)
            if cur and cur["rid"] == rid:
                del self.inflight[key]

        try: remote.close()
        except Exception: pass
        try: client_socket.close()
        except Exception: pass


gate = _RequestGate(DEBOUNCE_MS / 1000.0)


# ── Main handler ────────────────────────────────────────────────

def handle_client(client_socket):
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        client_socket.settimeout(5)
        first_chunk = client_socket.recv(BUFFER_SIZE)
        if not first_chunk:
            client_socket.close()
            return
    except Exception:
        client_socket.close()
        return

    method, path = _parse_request_line(first_chunk)
    request_data = read_full_client_request(client_socket, first_chunk)

    log(f"← {method} {path} ({len(request_data)}B)")

    # Handle Ollama meta endpoints locally
    if method == "GET" and path == "/":
        handle_root(client_socket)
        return
    if method == "GET" and path == "/api/tags":
        handle_tags(client_socket)
        return
    if method == "GET" and path == "/api/version":
        handle_version(client_socket)
        return
    if method == "POST" and path == "/api/show":
        try:
            body = json.loads(request_data.split(b"\r\n\r\n", 1)[1])
        except Exception:
            body = {}
        handle_show(client_socket, body)
        return

    # Handle generate/chat: translate and forward
    if path in DEBOUNCE_PATHS and method == "POST":
        try:
            ollama_body = json.loads(request_data.split(b"\r\n\r\n", 1)[1])
        except Exception as e:
            log(f"  ERROR: failed to parse body — {e}")
            _send_empty_ok(client_socket)
            return

        openai_body = ollama_to_openai(ollama_body)
        stream = openai_body.get("stream", False)
        request_body = json.dumps(openai_body).encode()

        log(f"  Translated: prompt={len(openai_body['prompt'])} chars, "
            f"max_tokens={openai_body.get('max_tokens')}, "
            f"temp={openai_body.get('temperature')}, stream={stream}")

        gate.submit(path, client_socket, request_body, stream)
        return

    # Fallback: 404
    body = b'{"error": "not found"}'
    client_socket.sendall(
        b"HTTP/1.1 404 Not Found\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"Connection: close\r\n\r\n" + body
    )
    client_socket.close()


def main():
    server = make_socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(("localhost", PROXY_PORT))
        server.listen(5)
    except OSError:
        print(f"ERROR: Port {PROXY_PORT} in use. Stop any local Ollama instance.")
        sys.exit(1)

    print(f"═══════════════════════════════════════════════════")
    print(f"  Ollama-compat proxy on localhost:{PROXY_PORT}")
    print(f"  Backend: {BACKEND_HOST}:{BACKEND_PORT} (ExLlamaV2)")
    print(f"  Debounce: {DEBOUNCE_MS}ms | Cancellation: ON")
    print(f"  Model alias: {MODEL_NAME}")
    print(f"  Translates: /api/generate → /v1/completions")
    print(f"═══════════════════════════════════════════════════")

    try:
        while True:
            client_sock, addr = server.accept()
            threading.Thread(
                target=handle_client, args=(client_sock,), daemon=True
            ).start()
    except KeyboardInterrupt:
        print("\nStopping proxy...")


if __name__ == "__main__":
    main()