"""
setup/verify_setup.py
=====================

End-to-end sanity check for the course setup.

Runs a battery of independent checks and prints a PASS/FAIL line for
each one. Exit code is 0 iff every check passes.

Usage::

    python setup/verify_setup.py
    python setup/verify_setup.py --model qwen3.5:9b
    python setup/verify_setup.py --ollama-url http://localhost:11434
    python setup/verify_setup.py --quick        # skip the MCP server check

Each check is fully isolated — a failure in one check does not abort
the others. That way a student can see everything that is broken in a
single run instead of fixing one thing at a time.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# Packages that MUST import for the course to work. Tuples are
# (import_name, human_name). The human_name lets us distinguish e.g.
# the `mcp` PyPI package from the Python builtin `os`.
REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("requests", "requests"),
    ("smolagents", "smolagents"),
    ("litellm", "litellm"),
    ("crewai", "crewai"),
    ("chromadb", "chromadb"),
    ("sentence_transformers", "sentence-transformers"),
    ("pypdf", "pypdf"),
    ("mcp", "mcp"),
    ("arxiv", "arxiv"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
]

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("ISING_OLLAMA_MODEL", "qwen3.5:4b")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _result(name: str, passed: bool, detail: str = "") -> CheckResult:
    return CheckResult(name=name, passed=passed, detail=detail)


# ======================================================================
# Individual checks
# ======================================================================
def check_python_version() -> CheckResult:
    v = sys.version_info
    ok = v >= (3, 11)
    msg = f"Python {v.major}.{v.minor}.{v.micro}"
    if not ok:
        msg += "  (course targets 3.11+)"
    return _result("Python version", ok, msg)


def check_packages() -> list[CheckResult]:
    results: list[CheckResult] = []
    for mod_name, display in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            results.append(_result(f"import {display}", True, f"version {ver}"))
        except Exception as exc:  # noqa: BLE001
            results.append(
                _result(
                    f"import {display}",
                    False,
                    f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def check_ising_simulator() -> CheckResult:
    try:
        # Make sure we can find the repo root on sys.path.
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from mcp_server.ising_simulator import run_ising_simulation, T_C_EXACT

        r = run_ising_simulation(
            lattice_size=16,
            temperature=T_C_EXACT,
            num_steps=500,
            algorithm="metropolis",
            seed=0,
        )
        m = r["magnetization_mean"]
        t = r["elapsed_seconds"]
        ok = 0.0 <= m <= 1.0 and t < 10.0
        return _result(
            "ising_simulator end-to-end",
            ok,
            f"<|m|>={m:.3f}  t={t:.2f}s",
        )
    except Exception as exc:  # noqa: BLE001
        return _result(
            "ising_simulator end-to-end",
            False,
            f"{type(exc).__name__}: {exc}",
        )


def check_ollama_daemon(url: str) -> CheckResult:
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:
        return _result("Ollama daemon", False, f"requests not available: {exc}")

    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        resp.raise_for_status()
        n_models = len(resp.json().get("models", []))
        return _result("Ollama daemon", True, f"{url}  ({n_models} models installed)")
    except Exception as exc:  # noqa: BLE001
        return _result(
            "Ollama daemon",
            False,
            f"cannot reach {url}: {type(exc).__name__}: {exc}. "
            "Start it with `ollama serve` or open the Ollama app.",
        )


def check_ollama_model(url: str, model: str) -> CheckResult:
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:
        return _result(f"Ollama model {model}", False, f"requests not available: {exc}")

    # 1. Is the model listed?
    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        resp.raise_for_status()
        tags = resp.json().get("models", [])
        names = {m.get("name", "") for m in tags}
        # Ollama's tag listing sometimes includes ":latest" suffixes; match both.
        if model not in names and f"{model}:latest" not in names:
            return _result(
                f"Ollama model {model}",
                False,
                f"not installed. Pull with: ollama pull {model}",
            )
    except Exception as exc:  # noqa: BLE001
        return _result(
            f"Ollama model {model}",
            False,
            f"listing models failed: {type(exc).__name__}: {exc}",
        )

    # 2. Does it actually respond to a tiny chat request?
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with just 'ok'."}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 8},
        }
        resp = requests.post(f"{url}/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "").strip()
        return _result(
            f"Ollama model {model}",
            bool(content),
            f"responded: {content[:40]!r}",
        )
    except Exception as exc:  # noqa: BLE001
        return _result(
            f"Ollama model {model}",
            False,
            f"chat request failed: {type(exc).__name__}: {exc}",
        )


def check_paper_corpus() -> CheckResult:
    papers_dir = Path("data/papers")
    pdfs = sorted(papers_dir.glob("*.pdf")) if papers_dir.is_dir() else []
    meta_path = papers_dir / "metadata.json"

    if not papers_dir.is_dir():
        return _result(
            "Paper corpus",
            False,
            f"{papers_dir} does not exist. Run: python setup/download_papers.py",
        )
    if len(pdfs) < 5:
        return _result(
            "Paper corpus",
            False,
            f"only {len(pdfs)} PDFs in {papers_dir} (need >=5). "
            "Run: python setup/download_papers.py",
        )
    if not meta_path.is_file():
        return _result(
            "Paper corpus",
            False,
            f"{meta_path} missing. Run: python setup/download_papers.py",
        )

    try:
        entries = json.loads(meta_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return _result("Paper corpus", False, f"metadata.json unreadable: {exc}")

    return _result(
        "Paper corpus",
        True,
        f"{len(pdfs)} PDFs, {len(entries)} metadata entries",
    )


def check_mcp_server() -> CheckResult:
    """Spawn the MCP server over stdio and do a real tools/list + call."""
    try:
        from mcp import ClientSession, StdioServerParameters  # noqa: PLC0415
        from mcp.client.stdio import stdio_client  # noqa: PLC0415
    except ImportError as exc:
        return _result("MCP server", False, f"mcp package not available: {exc}")

    async def _drive() -> tuple[bool, str]:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.physics_tools_server"],
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                expected = {"search_arxiv", "run_ising_simulation"}
                if not expected.issubset(names):
                    missing = expected - names
                    return False, f"missing tools: {sorted(missing)}"

                result = await session.call_tool(
                    "run_ising_simulation",
                    {
                        "lattice_size": 8,
                        "temperature": 2.269,
                        "num_steps": 200,
                        "algorithm": "metropolis",
                        "seed": 0,
                    },
                )
                for c in result.content:
                    if getattr(c, "type", None) == "text":
                        payload = json.loads(c.text)
                        m = payload["magnetization_mean"]
                        return True, (
                            f"tools={sorted(names)}  "
                            f"run_ising(8, 2.269, 200) -> <|m|>={m:.2f}"
                        )
                return False, "no text content in tool response"

    try:
        # Ensure the repo root is the cwd so `python -m mcp_server...` works.
        repo_root = Path(__file__).resolve().parent.parent
        old_cwd = Path.cwd()
        os.chdir(repo_root)
        try:
            ok, msg = asyncio.run(_drive())
        finally:
            os.chdir(old_cwd)
        return _result("MCP server (stdio)", ok, msg)
    except Exception as exc:  # noqa: BLE001
        return _result(
            "MCP server (stdio)",
            False,
            f"{type(exc).__name__}: {exc}",
        )


def check_ollama_binary() -> CheckResult:
    path = shutil.which("ollama")
    if not path:
        return _result(
            "ollama CLI",
            False,
            "not on PATH. Install from https://ollama.com/download",
        )
    # Don't require this to run — just knowing the binary exists is enough;
    # daemon reachability is a separate check.
    try:
        out = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        ver = (out.stdout or out.stderr).strip().splitlines()[0] if out.stdout or out.stderr else ""
        return _result("ollama CLI", True, f"{path}  ({ver})" if ver else path)
    except Exception as exc:  # noqa: BLE001
        return _result("ollama CLI", True, f"{path} (version check failed: {exc})")


# ======================================================================
# Runner
# ======================================================================
def _run_check(label: str, fn: Callable[[], CheckResult | list[CheckResult]]) -> list[CheckResult]:
    """Wrap each check so a crash becomes a FAIL, not a traceback."""
    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exception_only(type(exc), exc)[-1].strip()
        return [_result(label, False, f"crashed: {tb}")]
    if isinstance(out, CheckResult):
        return [out]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the course setup.")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip the MCP server sub-process check (faster).",
    )
    args = parser.parse_args(argv)

    print("=" * 72)
    print(" Course setup verification".center(72))
    print("=" * 72)

    all_results: list[CheckResult] = []
    all_results += _run_check("Python version", check_python_version)
    all_results += _run_check("imports", check_packages)
    all_results += _run_check("ising_simulator", check_ising_simulator)
    all_results += _run_check("ollama binary", check_ollama_binary)
    all_results += _run_check("ollama daemon", lambda: check_ollama_daemon(args.ollama_url))
    all_results += _run_check(
        "ollama model", lambda: check_ollama_model(args.ollama_url, args.model)
    )
    all_results += _run_check("paper corpus", check_paper_corpus)
    if not args.quick:
        all_results += _run_check("mcp server", check_mcp_server)

    # ---- pretty output ----
    name_col = max(len(r.name) for r in all_results)
    n_pass = sum(1 for r in all_results if r.passed)
    for r in all_results:
        tick = "\033[1;32mPASS\033[0m" if r.passed else "\033[1;31mFAIL\033[0m"
        print(f"  [{tick}] {r.name.ljust(name_col)}  {r.detail}")

    print("-" * 72)
    print(f"  {n_pass}/{len(all_results)} checks passed.")
    print("=" * 72)

    return 0 if n_pass == len(all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
