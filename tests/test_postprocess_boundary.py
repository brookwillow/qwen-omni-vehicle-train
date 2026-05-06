import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _module_ast(path: str) -> ast.Module:
    return ast.parse((PROJECT_ROOT / path).read_text(encoding="utf-8"))


def _imports_postprocess_call(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "tool_postprocess":
            return any(alias.name == "postprocess_action_call" for alias in node.names)
    return False


def _calls_postprocess(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "postprocess_action_call":
                return True
    return False


def test_eval_and_cli_do_not_postprocess_model_outputs():
    for path in ("eval.py", "infer_cli_omni.py"):
        tree = _module_ast(path)

        assert not _imports_postprocess_call(tree)
        assert not _calls_postprocess(tree)


def test_serve_keeps_postprocess_for_runtime_tool_calls():
    tree = _module_ast("serve.py")

    assert _imports_postprocess_call(tree)
    assert _calls_postprocess(tree)
