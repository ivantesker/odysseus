import httpx
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ModelDiscovery:
    def __init__(self, default_host: str):
        self.default_host = default_host
        self.openai_compat_path = "/v1/chat/completions"
        # Custom ports from env vars, merged into the scan list by discover_models.
        self._extra_ports: set = set()

    def _get_hosts(self) -> list[str]:
        """Get all hosts to scan, using env override, Tailscale, or default."""
        self._extra_ports = set()

        def _append_host(out: list[str], host: str) -> None:
            host = (host or "").strip()
            if not host or host in out:
                return
            out.append(host)

        def _append_env_hosts(out: list[str]) -> None:
            """Add hosts (and any custom ports) from provider-specific env vars."""
            for env_name in ("OLLAMA_BASE_URL", "OLLAMA_URL", "LM_STUDIO_URL"):
                raw = os.getenv(env_name, "").strip()
                if not raw:
                    continue
                try:
                    parsed = urlparse(raw if "://" in raw else "http://" + raw)
                    _append_host(out, parsed.hostname or "")
                    if parsed.port:
                        self._extra_ports.add(parsed.port)
                except Exception:
                    pass

        # Manual override takes priority
        extra = os.getenv("LLM_HOSTS", "").strip()
        if extra:
            hosts = [h.strip() for h in extra.split(",") if h.strip()]
            # Always include the default host too
            if self.default_host not in hosts:
                hosts.insert(0, self.default_host)
            _append_host(hosts, "host.docker.internal")
            _append_env_hosts(hosts)
            return hosts

        hosts = [self.default_host]
        # Docker desktop/Linux compose maps this to the host machine. That is
        # the common "I started Ollama normally on this computer" case.
        _append_host(hosts, "host.docker.internal")
        _append_env_hosts(hosts)
        return hosts

    def _fingerprint_provider(self, host: str, port: int) -> str | None:
        """Identify the server software via its native API, independent of port."""
        try:
            r = httpx.get(f"http://{host}:{port}/api/v1/models", timeout=1.5)
            if r.is_success:
                models = (r.json() or {}).get("models")
                if (isinstance(models, list) and models
                        and isinstance(models[0], dict)
                        and "key" in models[0] and "architecture" in models[0]):
                    return "lmstudio"
        except Exception:
            pass
        return None

    def _check_port(self, host: str, port: int) -> dict[str, Any] | None:
        """Check a single host:port for models."""
        base = f"http://{host}:{port}/v1"
        try:
            r = httpx.get(f"{base}/models", timeout=3)
            if not r.is_success:
                return None
            data = r.json() or {}
            ids = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
            if ids:
                return {
                    "host": host,
                    "port": port,
                    "url": f"http://{host}:{port}{self.openai_compat_path}",
                    "models": ids,
                    "models_display": [i.lstrip("/") for i in ids],
                    "provider": self._fingerprint_provider(host, port),
                }
        except Exception:
            pass
        return None

    def discover_models(self) -> dict[str, list[dict[str, Any]]]:
        """Discover available models from all reachable hosts."""
        hosts = self._get_hosts()
        items = []

        logger.info(f"Scanning {len(hosts)} hosts for models: {hosts}")

        # Well-known ports: 8000-8020 (vLLM, llama.cpp, SGLang, Cookbook),
        # 1234 (LM Studio), 11434 (Ollama)
        ports = list(range(8000, 8021)) + [1234, 11434]
        ports += [p for p in sorted(self._extra_ports) if p not in ports]
        targets = [(h, p) for h in hosts for p in ports]

        seen_models = set()  # dedupe by (port, model_ids) to avoid same machine via different IPs

        with ThreadPoolExecutor(max_workers=50) as pool:
            futures = {pool.submit(self._check_port, h, p): (h, p) for h, p in targets}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    key = (result["port"], tuple(sorted(result["models"])))
                    if key not in seen_models:
                        seen_models.add(key)
                        items.append(result)

        # Sort by host then port for consistent ordering
        items.sort(key=lambda x: (x["host"], x["port"]))

        logger.info(f"Discovered {len(items)} model endpoints across {len(hosts)} hosts")
        return {"hosts": hosts, "items": items}

    def get_providers(self) -> dict[str, Any]:
        """Get all discovered local providers."""
        discovery = self.discover_models()
        items = discovery["items"]
        providers = [{"provider": "local", "hosts": discovery["hosts"], "items": items}]
        return {"providers": providers}
