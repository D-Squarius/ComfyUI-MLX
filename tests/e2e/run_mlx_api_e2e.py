from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests


MODEL_REPOS = {
    "lm": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "vlm": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "tts": "mlx-community/Soprano-1.1-80M-bf16",
    "asr": "mlx-community/whisper-tiny-asr-4bit",
    "embeddings": "mlx-community/all-MiniLM-L6-v2-4bit",
}
MANIFESTS = {
    "lm": ("qwen2_5_0_5b_mlx_lm.mlx.json", "mlx_lm"),
    "vlm": ("qwen2_vl_2b_mlx_vlm.mlx.json", "mlx_vlm"),
    "tts": ("soprano_80m_mlx_audio_tts.mlx.json", "mlx_audio_tts"),
    "asr": ("whisper_tiny_4bit_mlx_audio_asr.mlx.json", "mlx_audio_asr"),
    "embeddings": ("minilm_mlx_embeddings.mlx.json", "mlx_embeddings"),
}
WORKFLOW_PATH = Path(__file__).with_name("mlx_api_text_generate_workflow.json")
HF_ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.npz",
    "*.py",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.jinja",
    "tokenizer*",
    "vocab*",
    "merges*",
    "normalizer.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ComfyUI MLX backend API end-to-end checks.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--artifacts-dir", default="")
    parser.add_argument("--model-local-path", default="", help="Override only the LLM model path.")
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--execution-timeout", type=float, default=600.0)
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--torch-cpu", action="store_true", help="Start ComfyUI with --cpu for Torch-side isolation. MLX still uses its own Metal device.")
    return parser.parse_args()


def free_port(host: str) -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def request_json(method: str, url: str, **kwargs) -> Any:
    response = requests.request(method, url, timeout=30, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"{method} {url} failed: {response.status_code} {response.text[:2000]}")
    if response.content:
        return response.json()
    return None


def wait_for_server(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url}/models/text_encoders", timeout=5)
            if response.status_code == 200:
                return
            last_error = RuntimeError(f"status {response.status_code}: {response.text[:200]}")
        except Exception as e:
            last_error = e
        time.sleep(0.5)
    raise TimeoutError(f"ComfyUI did not become ready within {timeout}s: {last_error}")


def wait_for_history(base_url: str, prompt_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = request_json("GET", f"{base_url}/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(0.5)
    raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")


def tail(path: Path, limit: int = 12000) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-limit:]


def repo_cache_dir(repo_root: Path, repo_id: str) -> Path:
    return repo_root / "models" / "mlx" / "huggingface" / f"models--{repo_id.replace('/', '--')}" / "snapshots"


def find_cached_snapshot(repo_root: Path, repo_id: str) -> str:
    local = repo_cache_dir(repo_root, repo_id)
    if not local.exists():
        return ""
    snapshots = sorted(path for path in local.iterdir() if path.is_dir())
    return str(snapshots[-1]) if snapshots else ""


def write_manifest(
    path: Path,
    loader_type: str,
    repo_id: str,
    local_path: str = "",
    generation: dict[str, Any] | None = None,
    audio: dict[str, Any] | None = None,
    video: dict[str, Any] | None = None,
    embeddings: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "backend": loader_type,
        "allow_patterns": HF_ALLOW_PATTERNS,
        "ignore_patterns": ["*.md"],
    }
    if local_path:
        payload["local_path"] = local_path
    else:
        payload["repo_id"] = repo_id
        payload["revision"] = "main"
        payload["local_files_only"] = False
    if generation:
        payload["generation"] = generation
    if audio:
        payload["audio"] = audio
    if video:
        payload["video"] = video
    if embeddings:
        payload["embeddings"] = embeddings
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_workflow(manifest_name: str, loader_type: str, prompt: str, max_length: int) -> dict[str, Any]:
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    workflow["1"]["inputs"]["clip_name"] = manifest_name
    workflow["1"]["inputs"]["type"] = loader_type
    workflow["2"]["inputs"]["prompt"] = prompt
    workflow["2"]["inputs"]["max_length"] = max_length
    return workflow


def submit_workflow(base_url: str, workflow: dict[str, Any], client_id: str) -> str:
    response = request_json("POST", f"{base_url}/prompt", json={"prompt": workflow, "client_id": client_id})
    prompt_id = response.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"/prompt response did not include prompt_id: {response}")
    return str(prompt_id)


def history_status(history_item: dict[str, Any]) -> str:
    return str(history_item.get("status", {}).get("status_str", ""))


def history_text(history_item: dict[str, Any], node_id: str = "3") -> str:
    output = history_item.get("outputs", {}).get(node_id, {})
    values = output.get("text") or ()
    if not values:
        return ""
    return str(values[0])


def history_audio(history_item: dict[str, Any], node_id: str = "4") -> list[dict[str, Any]]:
    output = history_item.get("outputs", {}).get(node_id, {})
    return list(output.get("audio") or [])


def assert_status_success(history_item: dict[str, Any]) -> None:
    status = history_status(history_item)
    if status != "success":
        raise AssertionError(f"expected success status, got {status}: {json.dumps(history_item, indent=2)[:4000]}")


def assert_text_contains(history_item: dict[str, Any], expected: str, node_id: str = "3") -> None:
    assert_status_success(history_item)
    text = history_text(history_item, node_id)
    if expected.lower() not in text.lower():
        raise AssertionError(f"expected generated text to contain {expected!r}, got {text!r}")


def assert_nonempty_text(history_item: dict[str, Any], node_id: str = "3") -> None:
    assert_status_success(history_item)
    text = history_text(history_item, node_id)
    if not text.strip():
        raise AssertionError(f"expected non-empty generated text in node {node_id}: {json.dumps(history_item, indent=2)[:4000]}")


def assert_llm_success(history_item: dict[str, Any]) -> None:
    assert_text_contains(history_item, "OK")
    audio_preview = history_text(history_item, "4")
    if audio_preview != "None":
        raise AssertionError(f"expected optional audio output to preview as None for LLM, got {audio_preview!r}")


def assert_embeddings_success(history_item: dict[str, Any]) -> dict[str, Any]:
    assert_status_success(history_item)
    text = history_text(history_item)
    payload = json.loads(text)
    if payload.get("shape") != [2, 384]:
        raise AssertionError(f"expected MiniLM embedding shape [2, 384], got {payload.get('shape')}: {text[:1000]}")
    return payload


def create_fixture_image(input_dir: Path) -> str:
    from PIL import Image, ImageDraw

    input_dir.mkdir(parents=True, exist_ok=True)
    image_name = "mlx_vlm_red_square.png"
    image_path = input_dir / image_name
    image = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([32, 32, 96, 96], fill="red")
    image.save(image_path)
    return image_name


def output_file_path(base_dir: Path, item: dict[str, Any]) -> Path:
    root = {
        "output": base_dir / "output",
        "temp": base_dir / "temp",
        "input": base_dir / "input",
    }.get(str(item.get("type")), base_dir / "output")
    subfolder = str(item.get("subfolder") or "")
    return root / subfolder / str(item["filename"])


def start_server(args: argparse.Namespace, base_dir: Path, artifacts_dir: Path) -> tuple[subprocess.Popen, Path, Path]:
    stdout_path = artifacts_dir / "server.stdout.log"
    stderr_path = artifacts_dir / "server.stderr.log"
    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")
    command = [
        sys.executable,
        "main.py",
        "--listen",
        args.host,
        "--port",
        str(args.port),
        "--base-directory",
        str(base_dir),
        "--database-url",
        f"sqlite:///{artifacts_dir / 'comfy-e2e.sqlite3'}",
        "--disable-all-custom-nodes",
        "--dont-print-server",
        "--log-stdout",
    ]
    if args.torch_cpu:
        command.append("--cpu")
    process = subprocess.Popen(
        command,
        cwd=args.repo_root,
        stdout=stdout,
        stderr=stderr,
        text=True,
    )
    return process, stdout_path, stderr_path


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def free_models(base_url: str) -> None:
    response = requests.post(f"{base_url}/free", json={"unload_models": True, "free_memory": True}, timeout=30)
    if response.status_code != 200:
        raise AssertionError(f"/free failed: {response.status_code} {response.text[:1000]}")


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    args.repo_root = str(repo_root)
    args.port = args.port or free_port(args.host)
    base_url = f"http://{args.host}:{args.port}"

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    artifacts_dir = Path(args.artifacts_dir or repo_root / "temp" / f"mlx-api-e2e-{timestamp}").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    base_dir = artifacts_dir / "base"
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    text_encoders = base_dir / "models" / "text_encoders"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_encoders.mkdir(parents=True, exist_ok=True)
    (base_dir / "models" / "mlx").mkdir(parents=True, exist_ok=True)

    image_name = create_fixture_image(input_dir)

    local_paths = {name: find_cached_snapshot(repo_root, repo) for name, repo in MODEL_REPOS.items()}
    if args.model_local_path:
        local_paths["lm"] = args.model_local_path

    write_manifest(
        text_encoders / MANIFESTS["lm"][0],
        MANIFESTS["lm"][1],
        MODEL_REPOS["lm"],
        local_path=local_paths["lm"],
    )
    write_manifest(
        text_encoders / MANIFESTS["vlm"][0],
        MANIFESTS["vlm"][1],
        MODEL_REPOS["vlm"],
        local_path=local_paths["vlm"],
        generation={"verbose": False},
        video={"frame_stride": 24, "max_frames": 4},
    )
    write_manifest(
        text_encoders / MANIFESTS["tts"][0],
        MANIFESTS["tts"][1],
        MODEL_REPOS["tts"],
        local_path=local_paths["tts"],
        generation={"max_tokens": 32, "temperature": 0.0, "verbose": False},
    )
    write_manifest(
        text_encoders / MANIFESTS["asr"][0],
        MANIFESTS["asr"][1],
        MODEL_REPOS["asr"],
        local_path=local_paths["asr"],
        generation={"language": "en", "return_timestamps": False, "chunk_duration": 30.0, "verbose": False},
    )
    write_manifest(
        text_encoders / MANIFESTS["embeddings"][0],
        MANIFESTS["embeddings"][1],
        MODEL_REPOS["embeddings"],
        local_path=local_paths["embeddings"],
        embeddings={"max_length": 64, "max_output_json_bytes": 262144},
    )

    shutil.copy2(WORKFLOW_PATH, artifacts_dir / WORKFLOW_PATH.name)

    process, stdout_path, stderr_path = start_server(args, base_dir, artifacts_dir)
    try:
        wait_for_server(base_url, args.startup_timeout)
        if process.poll() is not None:
            raise RuntimeError(f"server exited early with {process.returncode}\n{tail(stdout_path)}\n{tail(stderr_path)}")

        object_info = request_json("GET", f"{base_url}/object_info")
        for node in ("CLIPLoader", "TextGenerate", "PreviewAny", "LoadImage", "LoadAudio", "SaveAudio"):
            if node not in object_info:
                raise AssertionError(f"{node} missing from /object_info")
        clip_types = object_info["CLIPLoader"]["input"]["required"]["type"][0]
        for _, loader_type in MANIFESTS.values():
            if loader_type not in clip_types:
                raise AssertionError(f"{loader_type} missing from CLIPLoader type options")
        if len(object_info["TextGenerate"].get("output", ())) != 2:
            raise AssertionError("TextGenerate should expose text and optional audio outputs")

        models = request_json("GET", f"{base_url}/models/text_encoders")
        for manifest_name, _ in MANIFESTS.values():
            if manifest_name not in models:
                raise AssertionError(f"{manifest_name} missing from /models/text_encoders: {models}")

        queue_before = request_json("GET", f"{base_url}/queue")
        if queue_before.get("queue_running") or queue_before.get("queue_pending"):
            raise AssertionError(f"expected empty queue before test: {queue_before}")

        lm_workflow = load_workflow(MANIFESTS["lm"][0], MANIFESTS["lm"][1], "Reply with exactly OK.", 8)
        lm_prompt_id = submit_workflow(base_url, lm_workflow, "mlx-api-e2e-lm")
        lm_history = wait_for_history(base_url, lm_prompt_id, args.execution_timeout)
        assert_llm_success(lm_history)
        free_models(base_url)

        vlm_workflow = load_workflow(
            MANIFESTS["vlm"][0],
            MANIFESTS["vlm"][1],
            "Describe this image in one short sentence.",
            32,
        )
        vlm_workflow["5"] = {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
            "_meta": {"title": "Load VLM Test Image"},
        }
        vlm_workflow["2"]["inputs"]["image"] = ["5", 0]
        vlm_prompt_id = submit_workflow(base_url, vlm_workflow, "mlx-api-e2e-vlm")
        vlm_history = wait_for_history(base_url, vlm_prompt_id, args.execution_timeout)
        assert_nonempty_text(vlm_history)
        free_models(base_url)

        tts_workflow = load_workflow(MANIFESTS["tts"][0], MANIFESTS["tts"][1], "hello world", 32)
        tts_workflow["4"] = {
            "class_type": "SaveAudio",
            "inputs": {
                "audio": ["2", 1],
                "filename_prefix": "mlx_e2e/tts",
            },
            "_meta": {"title": "Save MLX TTS Audio"},
        }
        tts_prompt_id = submit_workflow(base_url, tts_workflow, "mlx-api-e2e-tts")
        tts_history = wait_for_history(base_url, tts_prompt_id, args.execution_timeout)
        assert_nonempty_text(tts_history)
        saved_audio = history_audio(tts_history)
        if not saved_audio:
            raise AssertionError(f"TTS workflow did not save audio: {json.dumps(tts_history, indent=2)[:4000]}")
        tts_audio_path = output_file_path(base_dir, saved_audio[0])
        if not tts_audio_path.exists():
            raise AssertionError(f"TTS output audio file missing: {tts_audio_path}")
        asr_input_name = "mlx_tts_for_asr.flac"
        shutil.copy2(tts_audio_path, input_dir / asr_input_name)
        free_models(base_url)

        asr_workflow = load_workflow(MANIFESTS["asr"][0], MANIFESTS["asr"][1], "", 64)
        asr_workflow["5"] = {
            "class_type": "LoadAudio",
            "inputs": {"audio": asr_input_name},
            "_meta": {"title": "Load TTS Audio For ASR"},
        }
        asr_workflow["2"]["inputs"]["audio"] = ["5", 0]
        asr_prompt_id = submit_workflow(base_url, asr_workflow, "mlx-api-e2e-asr")
        asr_history = wait_for_history(base_url, asr_prompt_id, args.execution_timeout)
        assert_nonempty_text(asr_history)
        free_models(base_url)

        embeddings_workflow = load_workflow(
            MANIFESTS["embeddings"][0],
            MANIFESTS["embeddings"][1],
            json.dumps([{"text": "cat"}, {"text": "dog"}]),
            8,
        )
        embeddings_prompt_id = submit_workflow(base_url, embeddings_workflow, "mlx-api-e2e-embeddings")
        embeddings_history = wait_for_history(base_url, embeddings_prompt_id, args.execution_timeout)
        embeddings_payload = assert_embeddings_success(embeddings_history)
        free_models(base_url)

        queue_after = request_json("GET", f"{base_url}/queue")
        if queue_after.get("queue_running") or queue_after.get("queue_pending"):
            raise AssertionError(f"expected empty queue after test: {queue_after}")

        summary = {
            "base_url": base_url,
            "artifacts_dir": str(artifacts_dir),
            "prompt_ids": {
                "lm": lm_prompt_id,
                "vlm": vlm_prompt_id,
                "tts": tts_prompt_id,
                "asr": asr_prompt_id,
                "embeddings": embeddings_prompt_id,
            },
            "generated_text": {
                "lm": history_text(lm_history),
                "vlm": history_text(vlm_history),
                "tts": history_text(tts_history),
                "asr": history_text(asr_history),
                "embeddings_shape": embeddings_payload["shape"],
            },
            "saved_audio": str(tts_audio_path),
            "model_sources": {
                name: local_paths[name] or MODEL_REPOS[name]
                for name in MODEL_REPOS
            },
        }
        (artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
        return 0
    except Exception:
        sys.stderr.write("=== server stdout tail ===\n")
        sys.stderr.write(tail(stdout_path) + "\n")
        sys.stderr.write("=== server stderr tail ===\n")
        sys.stderr.write(tail(stderr_path) + "\n")
        raise
    finally:
        if not args.keep_server:
            stop_server(process)


if __name__ == "__main__":
    raise SystemExit(main())
