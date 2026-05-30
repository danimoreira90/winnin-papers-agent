"""Architecture fitness functions (Neal Ford).

Walks src/papers_agent/ and asserts two invariants that the codebase
cannot drift away from without a deliberate test edit:

  1. No file exceeds SR-20's 200-line limit.
  2. Inter-layer imports respect the directional model
     api -> agents -> tools -> infra -> core.

Audited finding: tools/ imports VectorStoreClient / EmbeddingClient /
LLMClient Protocols from infra/. The rule allows it; the comment on
LAYER_ALLOWED records the deviation for posterity. If a future refactor
moves those Protocols to core/, tighten the rule.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent / "src" / "papers_agent"
LINE_LIMIT = 200

LAYER_DIRS = ("core", "infra", "tools", "agents", "api")

# Tightest dependency set that the current codebase satisfies. Each value
# lists layers a module of that layer is allowed to import from (besides
# its own layer and stdlib / third-party).
LAYER_ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "infra": {"core"},
    "tools": {"core", "infra"},  # imports Protocols (Vector/Embedding/LLMClient)
    "agents": {"core", "tools"},
    "api": {"core", "infra", "tools", "agents"},
    "main": {"core", "infra", "tools", "agents", "api"},
}


def _layer_of_path(path: pathlib.Path) -> str:
    parts = path.relative_to(ROOT).parts
    if len(parts) == 1:
        return "main"
    return parts[0]


def _layer_of_module(module: str) -> str | None:
    parts = module.split(".")
    if not parts or parts[0] != "papers_agent":
        return None
    if len(parts) < 2:
        return "main"
    second = parts[1]
    if second in LAYER_DIRS:
        return second
    return "main"


def _iter_py_files() -> list[pathlib.Path]:
    return sorted(ROOT.rglob("*.py"))


def test_no_file_exceeds_200_lines() -> None:
    over: list[tuple[str, int]] = []
    for path in _iter_py_files():
        n = sum(1 for _ in path.open(encoding="utf-8"))
        if n > LINE_LIMIT:
            over.append((str(path.relative_to(ROOT)), n))
    assert not over, f"SR-20 violation -- files over {LINE_LIMIT} lines: {over}"


def test_dependency_direction() -> None:
    violations: list[tuple[str, str, str, str]] = []
    for path in _iter_py_files():
        own_layer = _layer_of_path(path)
        allowed = LAYER_ALLOWED.get(own_layer, set())
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                imp_layer = _layer_of_module(module)
                if imp_layer is None or imp_layer == own_layer:
                    continue
                if imp_layer not in allowed:
                    violations.append(
                        (
                            str(path.relative_to(ROOT)),
                            own_layer,
                            module,
                            imp_layer,
                        )
                    )
    assert not violations, f"Layer rule violations: {violations}"
