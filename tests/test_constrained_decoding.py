"""TriggeredJsonProcessor 触发/掩码/回退逻辑测试(不依赖 torch/xgrammar)。"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constrained_decoding import TriggeredJsonProcessor


class FakeRow:
    def __init__(self, ids):
        self._ids = ids

    def tolist(self):
        return list(self._ids)


class FakeInputIds:
    def __init__(self, ids):
        self._ids = ids

    @property
    def shape(self):
        return (1, len(self._ids))

    def __getitem__(self, key):
        row, sl = key
        assert row == 0
        return FakeRow(self._ids[sl])


class FakeScores:
    def __init__(self, vocab=8):
        self.data = [0.0] * vocab
        self.filled_neg_inf = False

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self.data = [value] * len(self.data)
            self.filled_neg_inf = value == float("-inf")
        else:
            _, idx = key
            self.data[idx] = value


class FakeSession:
    def __init__(self, reject_at=None, terminate_after=None):
        self.accepted = []
        self.mask_calls = 0
        self._reject_at = reject_at
        self._terminate_after = terminate_after

    def accept(self, tid):
        if self._reject_at is not None and len(self.accepted) == self._reject_at:
            return False
        self.accepted.append(tid)
        return True

    def is_terminated(self):
        return (self._terminate_after is not None
                and len(self.accepted) >= self._terminate_after)

    def mask(self, scores):
        self.mask_calls += 1


TOKEN_TEXT = {1: "好", 2: "{", 3: '"name"', 4: "}", 9: " "}


def decode(ids):
    return "".join(TOKEN_TEXT.get(i, "?") for i in ids)


def make(session=None):
    return TriggeredJsonProcessor(session or FakeSession(), decode, eos_token_ids=[7])


def test_free_text_stays_unconstrained():
    session = FakeSession()
    p = TriggeredJsonProcessor(session, decode, eos_token_ids=[7])
    prompt = [5, 5, 5]
    p(FakeInputIds(prompt), FakeScores())                # 首 token 打分,不约束
    p(FakeInputIds(prompt + [1]), FakeScores())          # 首 token 是中文 -> OFF
    p(FakeInputIds(prompt + [1, 1]), FakeScores())
    assert session.accepted == [] and session.mask_calls == 0


def test_json_start_activates_grammar():
    session = FakeSession()
    p = TriggeredJsonProcessor(session, decode, eos_token_ids=[7])
    prompt = [5]
    p(FakeInputIds(prompt), FakeScores())
    scores = FakeScores()
    p(FakeInputIds(prompt + [2]), scores)                # "{" -> ACTIVE
    assert session.accepted == [2] and session.mask_calls == 1
    p(FakeInputIds(prompt + [2, 3]), FakeScores())
    assert session.accepted == [2, 3] and session.mask_calls == 2


def test_terminated_forces_eos():
    session = FakeSession(terminate_after=2)
    p = TriggeredJsonProcessor(session, decode, eos_token_ids=[7])
    prompt = [5]
    p(FakeInputIds(prompt), FakeScores())
    p(FakeInputIds(prompt + [2]), FakeScores())
    scores = FakeScores()
    p(FakeInputIds(prompt + [2, 4]), scores)             # 文法终止 -> 仅允许 eos
    assert scores.data[7] == 0.0
    assert all(math.isinf(v) and v < 0 for i, v in enumerate(scores.data) if i != 7)


def test_grammar_reject_falls_back_unconstrained():
    session = FakeSession(reject_at=1)
    p = TriggeredJsonProcessor(session, decode, eos_token_ids=[7])
    prompt = [5]
    p(FakeInputIds(prompt), FakeScores())
    p(FakeInputIds(prompt + [2]), FakeScores())          # 接受 "{"
    p(FakeInputIds(prompt + [2, 1]), FakeScores())       # 非法 token -> 回退 OFF
    mask_before = session.mask_calls
    p(FakeInputIds(prompt + [2, 1, 1]), FakeScores())
    assert session.mask_calls == mask_before             # 回退后不再掩码


def test_leading_whitespace_token_defers_decision():
    session = FakeSession()
    p = TriggeredJsonProcessor(session, decode, eos_token_ids=[7])
    prompt = [5]
    p(FakeInputIds(prompt), FakeScores())
    p(FakeInputIds(prompt + [9]), FakeScores())          # 纯空白,继续观望
    scores = FakeScores()
    p(FakeInputIds(prompt + [9, 1]), scores)             # " 好" -> OFF
    assert session.mask_calls == 0 and session.accepted == []
