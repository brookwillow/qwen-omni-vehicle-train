#!/usr/bin/env python3
"""Gradio UI for local audio capture and remote Qwen-Omni inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from record_remote_infer import DEFAULT_SERVER, build_payload, post_json


def _extract_result(resp: dict[str, Any]) -> tuple[str, str, str]:
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        call = tool_calls[0]
        fn = call.get("function") or {}
        args_raw = fn.get("arguments", "")
        try:
            args_obj = json.loads(args_raw)
            args_text = json.dumps(args_obj, ensure_ascii=False, indent=2)
        except (TypeError, json.JSONDecodeError):
            args_text = str(args_raw)
        summary = f"Tool Call\n\nname: `{fn.get('name', '')}`"
        return summary, args_text, json.dumps(resp, ensure_ascii=False, indent=2)

    content = message.get("content", "")
    summary = "Text Response"
    return summary, str(content), json.dumps(resp, ensure_ascii=False, indent=2)


def infer_audio(
    audio_path: str | None,
    server: str,
    model: str,
    hint_text: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[str, str, str]:
    if not audio_path:
        return "No Audio", "Please record or upload a WAV file first.", "{}"

    path = Path(audio_path).expanduser().resolve()
    if not path.exists():
        return "Audio Missing", f"File not found: {path}", "{}"

    try:
        url = f"{server.rstrip('/')}/v1/chat/completions"
        payload = build_payload(path, model, int(max_tokens), float(temperature), hint_text.strip())
        resp = post_json(url, payload, float(timeout))
        return _extract_result(resp)
    except Exception as exc:
        return "Request Failed", str(exc), "{}"


def build_app(default_server: str, default_model: str):
    import gradio as gr

    with gr.Blocks(title="Qwen-Omni Remote Audio Test") as demo:
        gr.Markdown("# Qwen-Omni Remote Audio Test")
        gr.Markdown("Record or upload local audio, then send it to the remote `serve.py` endpoint.")

        with gr.Row():
            server = gr.Textbox(label="Server", value=default_server)
            model = gr.Textbox(label="Model", value=default_model)

        audio = gr.Audio(
            label="Audio",
            sources=["microphone", "upload"],
            type="filepath",
            format="wav",
        )

        with gr.Row():
            hint_text = gr.Textbox(
                label="Hint Text",
                value="",
                placeholder="Optional. Leave empty for pure audio E2E.",
            )
            max_tokens = gr.Number(label="Max Tokens", value=128, precision=0)
            temperature = gr.Number(label="Temperature", value=0.0)
            timeout = gr.Number(label="Timeout Seconds", value=120)

        submit = gr.Button("Send Audio", variant="primary")

        result_summary = gr.Markdown(label="Summary")
        parsed_output = gr.Code(label="Parsed Output", language="json")
        raw_response = gr.Code(label="Raw Response", language="json")

        submit.click(
            infer_audio,
            inputs=[audio, server, model, hint_text, max_tokens, temperature, timeout],
            outputs=[result_summary, parsed_output, raw_response],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio UI for remote Qwen-Omni audio inference.")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Server base URL, e.g. http://10.95.64.153:8000")
    parser.add_argument("--model", default="qwen2.5-omni")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args.server, args.model)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
