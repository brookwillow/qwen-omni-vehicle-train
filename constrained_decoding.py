#!/usr/bin/env python3
"""触发式 schema 约束解码 (xgrammar)。

首个生成 token 以 { 或 [ 开头 => 判定为工具调用,激活 JSON-schema 文法约束,
每步将违反 schema 的 token logits 置 -inf; 否则(中文追问/Reject)完全不干预。

依赖: xgrammar (仅推理机需要), torch。本模块惰性导入 xgrammar,
GuidedDecoder 在服务启动时编译一次文法,每个请求用 processor() 取新会话。

Usage (server 侧):
    decoder = GuidedDecoder("data/guided_schema.json", tokenizer, vocab_size, eos_token_ids)
    out = model.generate(..., logits_processor=LogitsProcessorList([decoder.processor()]))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("constrained_decoding")

_UNDECIDED, _ACTIVE, _OFF = 0, 1, 2


class TriggeredJsonProcessor:
    """LogitsProcessor 兼容 callable。batch=1。

    session 需实现 accept(token_id)->bool / is_terminated()->bool / mask(scores);
    decode_fn(list[int])->str 用于首 token 触发判定。
    """

    def __init__(self, session, decode_fn, eos_token_ids: list[int]):
        self._session = session
        self._decode = decode_fn
        self._eos_ids = eos_token_ids
        self._start_len: int | None = None
        self._state = _UNDECIDED
        self._fed = 0

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != 1:
            return scores
        if self._start_len is None:
            self._start_len = input_ids.shape[-1]
        gen = input_ids[0, self._start_len:].tolist()

        if self._state == _UNDECIDED:
            if not gen:
                return scores  # 首 token 自由生成,由其内容决定是否激活
            head = self._decode(gen).lstrip()
            if not head:
                return scores
            self._state = _ACTIVE if head[0] in "{[" else _OFF

        if self._state == _OFF:
            return scores

        while self._fed < len(gen):
            tid = gen[self._fed]
            if self._session.is_terminated() or not self._session.accept(tid):
                logger.warning(
                    "[guided] token %d rejected by grammar at pos %d, fallback to unconstrained",
                    tid, self._fed,
                )
                self._state = _OFF
                return scores
            self._fed += 1

        if self._session.is_terminated():
            scores[:] = float("-inf")
            for eid in self._eos_ids:
                scores[0, eid] = 0.0
        else:
            self._session.mask(scores)
        return scores


class _XgrSession:
    def __init__(self, xgr, compiled_grammar, vocab_size: int):
        self._xgr = xgr
        self._matcher = xgr.GrammarMatcher(compiled_grammar)
        self._bitmask = xgr.allocate_token_bitmask(1, vocab_size)

    def accept(self, token_id: int) -> bool:
        return self._matcher.accept_token(token_id)

    def is_terminated(self) -> bool:
        return self._matcher.is_terminated()

    def mask(self, scores) -> None:
        self._matcher.fill_next_token_bitmask(self._bitmask)
        self._xgr.apply_token_bitmask_inplace(
            scores, self._bitmask.to(scores.device)
        )


class GuidedDecoder:
    """启动时编译一次 schema 文法,按请求生成触发式 processor。"""

    def __init__(self, schema_path: str, tokenizer, vocab_size: int,
                 eos_token_ids: list[int]):
        import xgrammar as xgr

        self._xgr = xgr
        self._vocab_size = vocab_size
        self._eos_ids = eos_token_ids
        self._tokenizer = tokenizer

        schema_str = Path(schema_path).read_text(encoding="utf-8")
        json.loads(schema_str)  # fail fast on invalid json
        tok_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
        compiler = xgr.GrammarCompiler(tok_info)
        # 训练输出为无空白紧凑 JSON,文法同样锁死
        self._compiled = compiler.compile_json_schema(
            schema_str, any_whitespace=False, indent=None, separators=(",", ":"),
        )
        logger.info("[guided] compiled schema %s (vocab=%d)", schema_path, vocab_size)

    def processor(self) -> TriggeredJsonProcessor:
        session = _XgrSession(self._xgr, self._compiled, self._vocab_size)
        decode_fn = lambda ids: self._tokenizer.decode(ids, skip_special_tokens=False)
        return TriggeredJsonProcessor(session, decode_fn, self._eos_ids)
