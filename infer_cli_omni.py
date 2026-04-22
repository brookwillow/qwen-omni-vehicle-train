#!/usr/bin/env python3
"""Simple interactive CLI inference for Qwen2.5-Omni (+ optional LoRA)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


DEFAULT_SYSTEM_PROMPT_FALLBACK = (
    "你是车载语音助手。你可以动态决策下一步是：Clarify 或 Action。"
    "如果收到补充信息或 Tool Result，请输出 Final Answer。"
    "如果请求不在工具能力范围内，请直接输出 Reject。"
)

# Default system prompt file – compact version used during training.
# Resolved relative to the script's parent directory (project root).
_PROJECT_DIR = Path(__file__).resolve().parent
_DEFAULT_SP_FILE = _PROJECT_DIR / "data" / "system-prompt.txt"

ACTION_RE = re.compile(r"Action:\s*([A-Za-z0-9_]+)")
ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{[\s\S]*\})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple CLI inference for Qwen2.5-Omni")
    p.add_argument("--model-dir", default="/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B")
    p.add_argument("--lora-dir", default="", help="Optional LoRA adapter directory")
    p.add_argument("--tools-file", default="data/tools.json", help="Tool schema json used for action validation")
    p.add_argument(
        "--system-prompt-file",
        default=str(_DEFAULT_SP_FILE),
        help="Path to system prompt text file (default: data/system-prompt.txt)",
    )
    p.add_argument("--system-prompt", default="", help="Inline system prompt (overrides --system-prompt-file if set)")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.2)
    return p.parse_args()


def load_model(model_dir: str, lora_dir: str):
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto",
    )
    if lora_dir:
        model = PeftModel.from_pretrained(model, lora_dir)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)
    return model, processor


def load_tools(path: str) -> Dict[str, Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    tool_map = {}
    for item in data:
        fn = item.get("function", {})
        name = fn.get("name")
        if name:
            tool_map[name] = fn
    return tool_map


def parse_action(text: str) -> Tuple[str, Dict[str, Any]]:
    m_tool = ACTION_RE.search(text or "")
    if not m_tool:
        return "", {}
    tool = m_tool.group(1).strip()
    m_args = ACTION_INPUT_RE.search(text or "")
    if not m_args:
        return tool, {}
    try:
        args = json.loads(m_args.group(1))
        return tool, args if isinstance(args, dict) else {}
    except Exception:
        return tool, {}


def validate_action(tool_map: Dict[str, Dict[str, Any]], tool: str, args: Dict[str, Any]) -> bool:
    if not tool:
        return True
    fn = tool_map.get(tool)
    if not fn:
        return False
    params = fn.get("parameters", {})
    props = params.get("properties", {})
    required = params.get("required", [])
    for k in required:
        if k not in args:
            return False
    for k, v in args.items():
        if k not in props:
            continue
        spec = props[k]
        if "enum" in spec and v not in spec["enum"]:
            return False
    return True


def generate_text(
    model,
    processor,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
) -> Tuple[str, Dict[str, int], Optional[torch.Tensor]]:
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(
        text=prompt,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(model.device)
    model_dtype = getattr(model, "dtype", None)
    if model_dtype is not None:
        inputs = inputs.to(model_dtype)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            return_audio=False,
        )

    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    gen_ids = out_ids[:, input_len:] if input_len > 0 else out_ids
    text = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    metrics = {
        "prompt_tokens": int(input_len),
        "gen_tokens": int(gen_ids.shape[-1]) if gen_ids.ndim == 2 else 0,
    }
    current_input_ids = inputs["input_ids"].detach().cpu() if "input_ids" in inputs else None
    return (text[0] if text else "").strip(), metrics, current_input_ids


def _prefix_overlap_tokens(prev_ids: Optional[torch.Tensor], cur_ids: Optional[torch.Tensor]) -> int:
    """Longest common prefix tokens between consecutive prompts."""
    if prev_ids is None or cur_ids is None:
        return 0
    if prev_ids.ndim != 2 or cur_ids.ndim != 2 or prev_ids.shape[0] == 0 or cur_ids.shape[0] == 0:
        return 0
    a = prev_ids[0]
    b = cur_ids[0]
    n = min(a.shape[0], b.shape[0])
    i = 0
    while i < n and a[i].item() == b[i].item():
        i += 1
    return i


def resolve_system_prompt(args) -> str:
    """Resolve system prompt: inline arg > file > fallback."""
    if args.system_prompt:
        print(f"[sp] from --system-prompt arg (chars={len(args.system_prompt)})")
        return args.system_prompt
    sp_path = Path(args.system_prompt_file)
    if sp_path.exists():
        text = sp_path.read_text(encoding="utf-8").strip()
        print(f"[sp] from file: {sp_path} (chars={len(text)})")
        return text
    print(f"[sp] WARNING: file not found: {sp_path}, using short fallback")
    return DEFAULT_SYSTEM_PROMPT_FALLBACK


def main() -> None:
    args = parse_args()
    system_prompt = resolve_system_prompt(args)

    model, processor = load_model(args.model_dir, args.lora_dir)
    if args.lora_dir:
        print(f"[lora] loaded: {args.lora_dir}")
    else:
        print("[lora] not loaded (using base model only)")
    tool_map = load_tools(args.tools_file)
    print(f"[tools] loaded: {len(tool_map)} from {args.tools_file}")
    prompt_cache_enabled = False
    print(f"[cache] prompt_cache_enabled={prompt_cache_enabled} (current engine path does full-prefill each turn)")
    messages: List[Dict[str, Any]] = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
    prev_input_ids: Optional[torch.Tensor] = None
    print("CLI ready. Type your input and press Enter. Type 'exit' to quit. Type '/reset' to reset dialog.")
    while True:
        try:
            user_text = input("\nUser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExit.")
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "q"}:
            print("Exit.")
            break
        if user_text.lower() == "/reset":
            messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
            prev_input_ids = None
            print("Assistant> 已重置对话上下文。")
            continue

        messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})

        reply, gen_metrics, current_input_ids = generate_text(
            model,
            processor,
            messages,
            args.max_new_tokens,
            args.temperature,
        )
        potential_reuse = _prefix_overlap_tokens(prev_input_ids, current_input_ids)
        hit_tokens = 0 if not prompt_cache_enabled else potential_reuse
        print(
            "[cache] "
            f"hit_tokens={hit_tokens} "
            f"potential_prefix_reuse={potential_reuse} "
            f"prompt_tokens={gen_metrics['prompt_tokens']} "
            f"gen_tokens={gen_metrics['gen_tokens']}"
        )
        prev_input_ids = current_input_ids
        tool, action_args = parse_action(reply)
        if tool and tool_map and not validate_action(tool_map, tool, action_args):
            print(
                f"[warn] invalid action against tool schema: tool={tool}, args={action_args} "
                "(raw model output kept)"
            )

        messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
        print(f"Assistant> {reply}")


if __name__ == "__main__":
    main()
