"""RunPod provisioner (REST API). Idempotent: resumes existing pod by name or creates a new one.
Reads env: RUNPOD_API_KEY (required), POD_NAME, GPU_TYPE_ID, IMAGE, VOLUME_ID, CLOUD_TYPE,
TAILSCALE_AUTH_KEY, HF_TOKEN.
Prints `POD_ID=<id>` on stdout for the bash wrapper to capture.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://rest.runpod.io/v1"


def http(method: str, path: str, api_key: str, body: dict | None = None) -> dict | None:
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"[provision] HTTP {e.code} {method} {path}: {e.read().decode(errors='replace')}\n")
        sys.exit(1)


def find_existing(api_key: str, pod_name: str) -> str | None:
    qs = urllib.parse.urlencode({"name": pod_name})
    data = http("GET", f"/pods?{qs}", api_key)
    if not data:
        return None
    pods = data if isinstance(data, list) else data.get("pods", [])
    for p in pods:
        if p.get("name") == pod_name:
            return p["id"]
    return None


def get_status(api_key: str, pod_id: str) -> dict:
    return http("GET", f"/pods/{pod_id}", api_key) or {}


def resume(api_key: str, pod_id: str) -> None:
    http("POST", f"/pods/{pod_id}/start", api_key, body={})


def create(api_key: str, args: dict) -> str:
    data = http("POST", "/pods", api_key, body=args)
    return data["id"]


def wait_running(api_key: str, pod_id: str, timeout_s: int = 600) -> str | None:
    """Wait for the pod to expose a public proxy URL. Returns the URL.

    RunPod's `runtime.uptimeInSeconds` stays null on the
    vllm-latest template even after the container is up. Use the proxy
    URL pattern + an HTTP /health probe instead."""
    deadline = time.time() + timeout_s
    proxy_url = f"https://{pod_id}-8000.proxy.runpod.net"
    last_status = ""
    while time.time() < deadline:
        d = get_status(api_key, pod_id)
        status = d.get("desiredStatus") or ""
        if status != last_status:
            sys.stderr.write(f"[provision] status={status}\n")
            last_status = status
        if status == "RUNNING":
            try:
                # Cloudflare 403's urllib's default User-Agent on RunPod
                # proxy URLs (`*.proxy.runpod.net`); explicit UA reaches
                # the container. Foundation followup #4.
                req = urllib.request.Request(
                    f"{proxy_url}/health",
                    method="GET",
                    headers={"User-Agent": "skillcacher-provision/0.1"},
                )
                with urllib.request.urlopen(req, timeout=5) as r:
                    if r.status == 200:
                        sys.stderr.write(f"[provision] /health OK at {proxy_url}\n")
                        return proxy_url
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
                pass  # not yet up; keep polling
        time.sleep(5)
    sys.stderr.write(f"[provision] pod {pod_id} not reachable within {timeout_s}s\n")
    return None


def main() -> None:
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        sys.stderr.write("[provision] RUNPOD_API_KEY required\n")
        sys.exit(1)
    pod_name = os.environ.get("POD_NAME", "skillcacher-bench")
    gpu_type_id = os.environ.get("GPU_TYPE_ID", "NVIDIA H100 80GB HBM3")
    # Default to RunPod's official `vllm-latest` template. Templates are pre-pulled
    # on most machines (much faster boot than custom images) and include RunPod's
    # SSH key auto-wiring so we don't need dockerStartCmd hacks.
    # bootstrap will install lmcache on top via bootstrap.sh — small wheel, ~1min.
    # Set IMAGE=... to fall back to a custom image (disables templateId path).
    template_id = os.environ.get("TEMPLATE_ID", "pvcdqlwm9r")
    image = os.environ.get("IMAGE", "").strip() or None
    volume_id = os.environ.get("VOLUME_ID", "").strip() or None
    cloud_type = os.environ.get("CLOUD_TYPE", "SECURE")
    tailscale_key = os.environ.get("TAILSCALE_AUTH_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    # Required by the vllm-latest template — the container's CMD is `vllm serve $MODEL_NAME`.
    # Default to Qwen2.5-7B (ungated, ~15GB) for a smoke run.
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

    # Public key that the pod startup script will write into authorized_keys.
    # SSH_PUBLIC_KEY env wins; otherwise prefer ~/.ssh/runpod_ed25519.pub (the key we
    # actually SSH with), falling back to ~/.ssh/id_ed25519.pub.
    public_key = os.environ.get("SSH_PUBLIC_KEY", "").strip()
    if not public_key:
        for pub_path in ("~/.ssh/runpod_ed25519.pub", "~/.ssh/id_ed25519.pub"):
            full = os.path.expanduser(pub_path)
            if os.path.exists(full):
                public_key = open(full).read().strip()
                sys.stderr.write(f"[provision] using public key from {pub_path}\n")
                break
    if not public_key:
        sys.stderr.write("[provision] WARNING: no public key found; SSH access will be unavailable\n")

    sys.stderr.write(f"[provision] checking for existing pod '{pod_name}'...\n")
    existing = find_existing(api_key, pod_name)
    if existing:
        sys.stderr.write(f"[provision] pod exists ({existing}); resuming...\n")
        resume(api_key, existing)
        pod_id = existing
    else:
        sys.stderr.write("[provision] creating new pod...\n")
        create_args: dict = {
            "name": pod_name,
            "computeType": "GPU",
            "gpuTypeIds": [gpu_type_id],
            "gpuCount": 1,
            "containerDiskInGb": 100,
            "volumeInGb": 0,
            "volumeMountPath": "/models",
            "cloudType": cloud_type,
            "interruptible": False,
            "ports": ["22/tcp", "8000/http"],
            "env": {
                "TAILSCALE_AUTH_KEY": tailscale_key,
                "HF_TOKEN": hf_token,
                "POD_NAME": pod_name,
                "PUBLIC_KEY": public_key,
                "MODEL_NAME": model_name,
                "CONDITION": os.environ.get("CONDITION", "prefix_cache"),
                "LMCACHE_SHIM_API_KEY": os.environ.get("LMCACHE_SHIM_API_KEY", ""),
            },
        }
        if image:
            # Custom-image fallback: must override both entrypoint and CMD because
            # most vllm images set ENTRYPOINT to the server, which would swallow our
            # bash script as args and crash-loop the container.
            create_args["imageName"] = image
            create_args["dockerEntrypoint"] = ["/bin/bash", "-c"]
            create_args["dockerStartCmd"] = [
                'mkdir -p /root/.ssh && '
                'echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys && '
                'chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys && '
                '(service ssh start 2>/dev/null || /usr/sbin/sshd) && '
                'sleep infinity',
            ]
            sys.stderr.write(f"[provision] using custom image {image}\n")
        else:
            create_args["templateId"] = template_id
            sys.stderr.write(f"[provision] using template {template_id}\n")
        if volume_id:
            create_args["networkVolumeId"] = volume_id
        pod_id = create(api_key, create_args)
        sys.stderr.write(f"[provision] created pod {pod_id}\n")

    sys.stderr.write("[provision] waiting for pod to come up...\n")
    proxy_url = wait_running(api_key, pod_id)
    if not proxy_url:
        sys.exit(1)
    sys.stderr.write(f"[provision] pod ready: {pod_id} ({proxy_url})\n")
    print(f"POD_ID={pod_id}")
    print(f"PROXY_URL={proxy_url}")


if __name__ == "__main__":
    main()
