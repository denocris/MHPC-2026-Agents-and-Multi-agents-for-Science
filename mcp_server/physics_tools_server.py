"""
mcp_server/physics_tools_server.py
==================================

Model Context Protocol (MCP) server exposing the physics tools that the
Day 1 agent uses in Module 1.3.

Two tools are exposed:

* ``search_arxiv(query, max_results=5)``
      Live search of the public arXiv API. Returns a list of
      ``{title, authors, abstract, arxiv_id, url, published}`` dicts.

* ``run_ising_simulation(lattice_size, temperature, num_steps, algorithm,
                          thermalization_steps=None, seed=None)``
      Thin wrapper around ``mcp_server.ising_simulator.run_ising_simulation``.
      Returns the full observables dict (see that module).

Transport
---------
Uses **stdio** transport (the default for local MCP servers). The agent
launches this script as a subprocess and talks to it over stdin/stdout
via JSON-RPC. To run it manually for testing::

    python -m mcp_server.physics_tools_server

For a one-shot sanity check that does NOT need an agent, see the
``_selftest`` function below — invoke via::

    python -m mcp_server.physics_tools_server --selftest
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_server.ising_simulator import run_ising_simulation as _run_ising


# ======================================================================
# Server instance
# ======================================================================
# The name shows up in ``tools/list`` responses and in agent UIs.
mcp = FastMCP("physics-tools")


# ======================================================================
# Tool 1: search_arxiv
# ======================================================================
@mcp.tool()
def search_arxiv(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the arXiv preprint server and return the top results.

    Args:
        query: free-text search query (title/abstract/authors), e.g.
            "2D Ising model Monte Carlo" or "Wolff cluster algorithm".
            For better precision use arXiv's field prefixes and quoted
            phrases, e.g. ``ti:"Ising" AND abs:"neural network"``.
        max_results: maximum number of hits to return (1..20). Defaults to 5.

    Returns:
        A dict of the form::

            {
              "query": "...",
              "count": N,
              "results": [
                 {"title": ..., "authors": [...], "abstract": ...,
                  "arxiv_id": ..., "url": ..., "published": "YYYY-MM-DD"},
                 ...
              ]
            }

        On transient failure (e.g. arXiv rate-limiting, HTTP 429), the
        same shape is returned but with ``results == []`` and two extra
        keys, ``"error"`` and ``"hint"``, so the caller can reason about
        the failure in code instead of having to catch an exception.

    Note on return shape: we deliberately wrap the list in a dict rather
    than returning a bare ``list[dict]``. FastMCP serialises a list
    return as *multiple* content blocks (one per element), and some MCP
    clients — including smolagents' ``mcpadapt`` bridge — only read the
    first block. Returning a single top-level dict guarantees the whole
    payload arrives in one content block.
    """
    # Clamp max_results to a sane range so the agent can't DoS the arXiv API.
    max_results = max(1, min(int(max_results), 20))

    # Import lazily so the server can still start and expose the other tool
    # even if the `arxiv` package is missing for some reason.
    try:
        import arxiv
    except ImportError as exc:
        raise RuntimeError(
            "The 'arxiv' Python package is not installed. "
            "Install it with: pip install arxiv"
        ) from exc

    import time

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    # arXiv rate-limits aggressively: 1 request per 3 s per IP, and once
    # tripped the limit can persist for several minutes. We therefore:
    #   (a) space page requests by 5 s (comfortably above arXiv's 3 s),
    #   (b) retry up to 3 times with exponential backoff on HTTP 429, and
    #   (c) if the API is still refusing us, return a graceful error
    #       record *instead of raising* so the agent can fall back to
    #       search_papers on the local corpus or explain to the user.
    def _one_attempt() -> list[dict[str, Any]]:
        client = arxiv.Client(page_size=max_results, delay_seconds=5.0, num_retries=3)
        out: list[dict[str, Any]] = []
        for r in client.results(search):
            out.append(
                {
                    "title": r.title.strip().replace("\n", " "),
                    "authors": [a.name for a in r.authors],
                    "abstract": r.summary.strip().replace("\n", " "),
                    "arxiv_id": r.get_short_id(),
                    "url": r.entry_id,
                    "published": r.published.date().isoformat() if r.published else None,
                }
            )
        return out

    backoff = 10.0  # seconds before the first retry; doubles each time
    last_err: Exception | None = None
    for attempt in range(1, 4):  # up to 3 tries
        try:
            results = _one_attempt()
            return {"query": query, "count": len(results), "results": results}
        except arxiv.HTTPError as exc:
            last_err = exc
            if getattr(exc, "status", None) == 429 and attempt < 3:
                time.sleep(backoff)
                backoff *= 2
                continue
            break
        except Exception as exc:  # noqa: BLE001 -- we genuinely want to swallow
            last_err = exc
            break

    # All retries exhausted -- return a structured error the agent can read.
    return {
        "query": query,
        "count": 0,
        "results": [],
        "max_results": max_results,
        "error": f"arXiv API call failed: {type(last_err).__name__}: {last_err}",
        "hint": (
            "arXiv is likely rate-limiting this IP. Rate limits typically "
            "clear within 15-30 minutes. In the meantime, try the "
            "`search_papers` tool to query the local RAG corpus of "
            "pre-downloaded Ising-model papers, or rephrase the query and "
            "retry in a few minutes."
        ),
    }


# ======================================================================
# Tool 2: run_ising_simulation
# ======================================================================
@mcp.tool()
def run_ising_simulation(
    lattice_size: int,
    temperature: float,
    num_steps: int,
    algorithm: str = "wolff",
    thermalization_steps: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run a 2D Ising-model Monte Carlo simulation and return observables.

    The underlying code is ``mcp_server.ising_simulator.run_ising_simulation``.
    Units: J = 1, k_B = 1. The exact critical temperature is
    T_c = 2 / ln(1 + sqrt(2)) ≈ 2.269.

    Args:
        lattice_size: linear lattice size L (the lattice is L x L with
            periodic boundary conditions). Must be >= 2. Typical
            classroom values: 16, 32, 64.
        temperature: temperature T in units of J/k_B (must be > 0).
        num_steps: number of Monte Carlo measurement steps (after
            thermalisation).
        algorithm: 'metropolis' (single-spin flip, good far from T_c)
            or 'wolff' (single-cluster, much better near T_c). Defaults
            to 'wolff'.
        thermalization_steps: optional number of discarded warm-up steps.
            Defaults to max(100, num_steps // 5) for Metropolis and
            max(50, num_steps // 5) for Wolff.
        seed: optional RNG seed for reproducibility.

    Returns:
        A dict with ``magnetization_mean``, ``magnetization_std``,
        ``energy_mean``, ``energy_std``, ``specific_heat``,
        ``susceptibility``, ``final_configuration`` (L x L list of +/-1),
        plus bookkeeping fields (``algorithm``, ``lattice_size``,
        ``temperature``, ``num_steps``, ``thermalization_steps``,
        ``acceptance_rate``, ``mean_cluster_size``, ``elapsed_seconds``).
    """
    return _run_ising(
        lattice_size=lattice_size,
        temperature=temperature,
        num_steps=num_steps,
        algorithm=algorithm,
        thermalization_steps=thermalization_steps,
        seed=seed,
    )


# ======================================================================
# Self-test (run WITHOUT an agent or MCP client)
# ======================================================================
def _selftest() -> None:
    """Exercise both tools directly, through the FastMCP machinery.

    This does not speak JSON-RPC on stdio; instead it uses FastMCP's
    in-process helpers to (a) list the registered tools and (b) call
    ``run_ising_simulation``. That is sufficient to prove the tools are
    discoverable and wired up correctly end-to-end, which is what the
    course setup verification needs.
    """

    async def _main() -> None:
        tools = await mcp.list_tools()
        names = sorted(t.name for t in tools)
        print(f"tools/list -> {names}")
        assert "search_arxiv" in names, "search_arxiv missing"
        assert "run_ising_simulation" in names, "run_ising_simulation missing"

        # Call run_ising_simulation through the MCP call path
        result = await mcp.call_tool(
            "run_ising_simulation",
            {
                "lattice_size": 16,
                "temperature": 2.269,
                "num_steps": 5000,
                "algorithm": "metropolis",
                "seed": 1,
            },
        )
        # FastMCP returns either (content_list, structured_dict) or just
        # a content list depending on version; normalise both cases.
        if isinstance(result, tuple):
            _, payload = result
        else:
            payload = None
            for item in result:
                text = getattr(item, "text", None)
                if text:
                    try:
                        payload = json.loads(text)
                        break
                    except json.JSONDecodeError:
                        continue
        if payload is None:
            raise RuntimeError(f"Could not parse tool result: {result!r}")

        print(
            "run_ising_simulation(16, 2.269, 5000) ->"
            f" <|m|>={payload['magnetization_mean']:.3f}"
            f" +/- {payload['magnetization_std']:.3f},"
            f" <E>={payload['energy_mean']:.3f},"
            f" C_v={payload['specific_heat']:.3f},"
            f" chi={payload['susceptibility']:.3f},"
            f" t={payload['elapsed_seconds']:.2f}s"
        )
        print("SELFTEST OK")

    asyncio.run(_main())


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
    else:
        # stdio transport is the default for local MCP servers.
        mcp.run()
