from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SKILLCACHER_", extra="ignore")

    backend_url: str = Field(default="http://localhost:8000")
    backend_model: str = Field(default="meta-llama/Llama-3.3-70B-Instruct")
    backend_api_key: str = Field(default="")
    # Hard ceiling on max_tokens forwarded to the backend. Claude Code
    # requests 32K, but the dev pod runs Qwen3-8B at max_model_len=16384 —
    # vLLM rejects max_tokens > max_model_len. Default 8192 leaves prompt
    # headroom on a 16K-context backend; bump to match larger-context pods.
    backend_max_completion_tokens: int = Field(default=8192)
    proxy_port: int = Field(default=4000)
    proxy_host: str = Field(default="127.0.0.1")
    trace_dir: Path = Field(default=Path("./benchmark/traces"))
    tokenizer_name: str = Field(default="meta-llama/Llama-3.3-70B-Instruct")
    request_timeout_s: float = Field(default=600.0)

    # additions
    skill_dirs: str = Field(default="~/.claude/skills:.claude/skills")
    span_registry_path: Path = Field(default=Path("./benchmark/span_registry.sqlite"))
    lmcache_shim_url: str = Field(default="")
    lmcache_shim_api_key: str = Field(default="")
    runpod_api_key: str = Field(default="")
    runpod_pod_id: str = Field(default="")
    enable_pre_seed: bool = Field(default=True)
    enable_lookup: bool = Field(default=True)
    enable_stdout_tail: bool = Field(default=True)

    # optional path to a JSONL of statistically-mined spans.
    # When set + non-empty, the proxy startup pre-seeds these alongside
    # the structural skill prefixes. Empty/missing = "structural-only"
    # (the §3 ablation baseline arm).
    statistical_spans_file: str = Field(default="")

    def parsed_skill_dirs(self) -> list[Path]:
        return [Path(p).expanduser() for p in self.skill_dirs.split(":") if p]
