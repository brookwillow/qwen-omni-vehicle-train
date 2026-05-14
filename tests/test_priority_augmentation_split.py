import json
import re
from pathlib import Path

from scripts.validate_splits import load_schema, validate_sample


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BY_TOOL_DIR = PROJECT_ROOT / "data" / "splits" / "by_tool"
REJECT_PATH = PROJECT_ROOT / "data" / "splits" / "reject.jsonl"
MULTITURN_PATH = PROJECT_ROOT / "data" / "splits" / "multiturn.jsonl"


def _load_samples() -> list[dict]:
    samples = []
    for path in sorted(BY_TOOL_DIR.glob("*.jsonl")):
        samples.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return samples


def test_by_tool_split_exists_with_enough_samples():
    samples = _load_samples()

    assert len(samples) >= 6000


def test_by_tool_split_is_schema_valid():
    schema = load_schema(str(PROJECT_ROOT / "data" / "tools.json"))

    issues = []
    for index, sample in enumerate(_load_samples(), start=1):
        issues.extend(validate_sample(sample, f"by_tool:{index}", schema))

    assert issues == []


def test_by_tool_split_targets_known_error_clusters():
    samples = _load_samples()
    contents = "\n".join(
        msg["content"]
        for sample in samples
        for msg in sample["messages"]
        if msg["role"] == "assistant"
    )

    for tool in ("ClimateControl", "ProfileControl", "WindowControl", "LightControl"):
        assert contents.count(f'"name":"{tool}"') >= 12


def test_phone_split_includes_builtin_contacts():
    phone_path = BY_TOOL_DIR / "PhoneControl.jsonl"
    contacts = []
    for line in phone_path.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        for message in sample["messages"]:
            if message["role"] != "assistant":
                continue
            try:
                tool_call = json.loads(message["content"])
            except json.JSONDecodeError:
                continue
            if tool_call.get("name") == "PhoneControl":
                contacts.append(tool_call.get("arguments", {}).get("contact"))

    for contact in ("小鹏客服", "小鹏救援", "儿童手表"):
        assert contacts.count(contact) >= 5


def test_reject_split_only_contains_reject_final_assistant():
    for index, line in enumerate(REJECT_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        sample = json.loads(line)
        assistants = [
            message["content"].strip()
            for message in sample["messages"]
            if message["role"] == "assistant"
        ]

        assert assistants[-1] == "Reject", f"reject.jsonl line {index} has final assistant {assistants[-1]!r}"


def test_reject_split_stays_conservative_near_vehicle_tools():
    high_risk_queries = {
        "现在用的是什么驾驶模式",
        "现在是什么驾驶模式",
        "充电口有没有插着",
        "鹏翼门关了吗",
        "后备箱是开着的吗",
        "氛围灯开着吗",
        "冰箱开着吗",
        "你好",
    }
    home_markers = ("家里", "家里的", "家中", "卧室", "客厅", "厨房", "阳台", "热水器", "扫地机器人", "台灯")
    final_users = []
    for line in REJECT_PATH.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        users = [
            message["content"].strip()
            for message in sample["messages"]
            if message["role"] == "user"
        ]
        final_users.append(users[-1])

    assert high_risk_queries.isdisjoint(final_users)
    assert sum(any(marker in query for marker in home_markers) for query in final_users) <= 40
    assert sum(query.startswith(("帮我", "给我")) for query in final_users) == 0


def test_multiturn_assistant_avoids_can_help_phrasing():
    banned_phrases = ("我可以帮", "可以帮", "我可以帮助")

    for index, line in enumerate(MULTITURN_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        sample = json.loads(line)
        for message in sample["messages"]:
            if message["role"] != "assistant":
                continue
            assert not any(phrase in message["content"] for phrase in banned_phrases), (
                f"multiturn.jsonl line {index} has weak assistant phrasing {message['content']!r}"
            )


def test_multiturn_assistant_avoids_undecided_action_options():
    banned_phrases = ("或调节空调", "或调整空调", "或调空调")
    option_pattern = re.compile(r"帮您.*或.*。")

    for index, line in enumerate(MULTITURN_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        sample = json.loads(line)
        for message in sample["messages"]:
            if message["role"] != "assistant":
                continue
            content = message["content"]
            assert not any(phrase in content for phrase in banned_phrases), (
                f"multiturn.jsonl line {index} has undecided action wording {content!r}"
            )
            assert option_pattern.search(content) is None, (
                f"multiturn.jsonl line {index} has option wording in action response {content!r}"
            )


def test_multiturn_history_is_text_only_before_final_query():
    for index, line in enumerate(MULTITURN_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        sample = json.loads(line)
        messages = sample["messages"]
        final_user_index = max(i for i, message in enumerate(messages) if message["role"] == "user")

        for message in messages[:final_user_index]:
            content = message["content"].strip()
            assert message["role"] != "tool", f"multiturn.jsonl line {index} has tool role in history"
            assert not content.startswith("工具结果"), (
                f"multiturn.jsonl line {index} has tool result text in history"
            )
            if message["role"] == "assistant":
                try:
                    tool_call = json.loads(content)
                except json.JSONDecodeError:
                    continue
                assert not (
                    isinstance(tool_call, dict)
                    and "name" in tool_call
                    and "arguments" in tool_call
                ), f"multiturn.jsonl line {index} has assistant tool call in history"


def test_multiturn_current_turn_takes_priority_over_history():
    weak_current_turn = re.compile(
        r"^(嗯+|啊+|哦+|好+的*|对+|谢谢|你好|喂喂|算了.*|不用了|没事|等(等|一下)|稍等.*|"
        r"先这样.*|先别.*|我不是.*|那个.*|\d+)$"
    )
    external_context_markers = ("新闻", "天气", "播客", "有声书", "百科", "AIGC", "导航")
    explicit_current_turn_markers = ("打开", "关闭", "调", "切到", "查", "看下", "把")
    context_completion_markers = ("它", "现在", "开着", "关好", "状态", "多少", "也")

    weak_inherited_actions = []
    explicit_current_after_context = 0
    context_completion_queries = 0
    clarification_samples = 0

    for index, line in enumerate(MULTITURN_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        sample = json.loads(line)
        messages = sample["messages"]
        users = [message["content"].strip() for message in messages if message["role"] == "user"]
        final_user = users[-1]
        final_assistant = messages[-1]["content"].strip()

        try:
            tool_call = json.loads(final_assistant)
        except json.JSONDecodeError:
            if "请问" in final_assistant or "请说明" in final_assistant:
                clarification_samples += 1
            continue

        tool_name = tool_call.get("name")
        if weak_current_turn.match(final_user) and tool_name not in {"NoiseDoNotAct", "Reject"}:
            weak_inherited_actions.append((index, final_user, tool_name))

        previous_user_text = "".join(users[:-1])
        if any(marker in previous_user_text for marker in external_context_markers) and any(
            marker in final_user for marker in explicit_current_turn_markers
        ):
            explicit_current_after_context += 1

        if tool_name == "CarUsageSearch" and any(marker in final_user for marker in context_completion_markers):
            context_completion_queries += 1

    assert weak_inherited_actions == []
    assert explicit_current_after_context >= 20
    assert context_completion_queries >= 10
    assert clarification_samples >= 8


def test_multiturn_current_turn_overrides_similar_history():
    expected_final_calls = {
        "现在把主驾关上": {
            "name": "WindowControl",
            "arguments": {"action": "关闭", "device": "车窗", "position": "主驾"},
        },
        "主驾打开一点": {
            "name": "WindowControl",
            "arguments": {"action": "再开", "device": "车窗", "position": "主驾"},
        },
        "只打开主驾": {
            "name": "WindowControl",
            "arguments": {"action": "打开", "device": "车窗", "position": "主驾"},
        },
        "副驾关上": {
            "name": "WindowControl",
            "arguments": {"action": "关闭", "device": "车窗", "position": "副驾"},
        },
        "主驾打开加热": {
            "name": "SeatControl",
            "arguments": {"action": "打开", "device": "座椅", "feature": "制热", "position": "主驾"},
        },
        "副驾打开通风": {
            "name": "SeatControl",
            "arguments": {"action": "打开", "device": "座椅", "feature": "通风", "position": "副驾"},
        },
        "媒体声音调高": {
            "name": "VoiceControl",
            "arguments": {"action": "调高", "feature": "声音"},
        },
        "风量调高一点": {
            "name": "ClimateControl",
            "arguments": {"action": "调高", "device": "空调", "feature": "风"},
        },
        "打开左侧侧滑门": {
            "name": "GateControl",
            "arguments": {"action": "打开", "device": "侧滑门", "position": "左侧"},
        },
        "打开左侧儿童锁": {
            "name": "LockControl",
            "arguments": {"action": "打开", "device": "儿童锁", "position": "左侧"},
        },
        "现在主驾打开": {
            "name": "WindowControl",
            "arguments": {"action": "打开", "device": "车窗", "position": "主驾"},
        },
    }
    observed = {}

    for line in MULTITURN_PATH.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        users = [message["content"].strip() for message in sample["messages"] if message["role"] == "user"]
        final_user = users[-1]
        if final_user not in expected_final_calls:
            continue
        observed[final_user] = json.loads(sample["messages"][-1]["content"])

    assert observed == expected_final_calls


def test_multiturn_generic_done_history_still_uses_tool_for_current_control():
    expected_final_calls = {
        "副驾的窗户也关上吧": {
            "name": "WindowControl",
            "arguments": {"action": "关闭", "device": "车窗", "position": "副驾"},
        },
        "主驾的也关上": {
            "name": "WindowControl",
            "arguments": {"action": "关闭", "device": "车窗", "position": "主驾"},
        },
        "副驾阅读灯打开": {
            "name": "LightControl",
            "arguments": {"action": "打开", "device": "阅读灯", "position": "副驾"},
        },
        "主驾座椅加热打开": {
            "name": "SeatControl",
            "arguments": {"action": "打开", "device": "座椅", "feature": "制热", "position": "主驾"},
        },
        "媒体声音调高一点": {
            "name": "VoiceControl",
            "arguments": {"action": "调高", "feature": "声音"},
        },
        "风量调高一点": {
            "name": "ClimateControl",
            "arguments": {"action": "调高", "device": "空调", "feature": "风"},
        },
        "打开右侧侧滑门": {
            "name": "GateControl",
            "arguments": {"action": "打开", "device": "侧滑门", "position": "右侧"},
        },
        "左侧儿童锁打开": {
            "name": "LockControl",
            "arguments": {"action": "打开", "device": "儿童锁", "position": "左侧"},
        },
    }
    generic_done_phrases = ("搞定", "已处理", "处理好了", "关好了", "已经关上")
    observed = {}

    for line in MULTITURN_PATH.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        messages = sample["messages"]
        users = [message["content"].strip() for message in messages if message["role"] == "user"]
        final_user = users[-1]
        if final_user not in expected_final_calls:
            continue

        history_assistant_text = "\n".join(
            message["content"] for message in messages[:-1] if message["role"] == "assistant"
        )
        if not any(phrase in history_assistant_text for phrase in generic_done_phrases):
            continue
        observed[final_user] = json.loads(messages[-1]["content"])

    assert observed == expected_final_calls


def test_multiturn_ellipsis_inherits_nearest_prior_control_intent():
    expected_final_calls = {
        ("打开车窗", "打开大灯", "关闭吧"): {
            "name": "LightControl",
            "arguments": {"action": "关闭", "device": "大灯"},
        },
        ("打开大灯", "打开车窗", "关掉吧"): {
            "name": "WindowControl",
            "arguments": {"action": "关闭", "device": "车窗"},
        },
        ("把空调打开", "打开香氛", "关了吧"): {
            "name": "PerfumeControl",
            "arguments": {"action": "关闭", "device": "香氛"},
        },
        ("打开香氛", "打开空气净化器", "关闭吧"): {
            "name": "ClimateControl",
            "arguments": {"action": "关闭", "device": "空气净化器"},
        },
        ("打开副驾座椅通风", "打开主驾按摩", "关掉吧"): {
            "name": "SeatControl",
            "arguments": {"action": "关闭", "device": "座椅", "feature": "按摩", "position": "主驾"},
        },
        ("打开主驾按摩", "打开副驾座椅通风", "关闭吧"): {
            "name": "SeatControl",
            "arguments": {"action": "关闭", "device": "座椅", "feature": "通风", "position": "副驾"},
        },
        ("打开右侧儿童锁", "打开左侧侧滑门", "关上吧"): {
            "name": "GateControl",
            "arguments": {"action": "关闭", "device": "侧滑门", "position": "左侧"},
        },
        ("打开左侧侧滑门", "打开右侧儿童锁", "关掉吧"): {
            "name": "LockControl",
            "arguments": {"action": "关闭", "device": "儿童锁", "position": "右侧"},
        },
        ("打开雨刮", "打开娱乐屏", "关掉吧"): {
            "name": "ScreenControl",
            "arguments": {"action": "关闭", "device": "娱乐屏"},
        },
        ("打开娱乐屏", "打开雨刮", "关掉吧"): {
            "name": "WiperControl",
            "arguments": {"action": "关闭", "device": "雨刮", "value": "自动"},
        },
        ("导航声音调高", "媒体声音调高", "调低一点"): {
            "name": "VoiceControl",
            "arguments": {"action": "调低", "feature": "声音"},
        },
        ("媒体声音调高", "导航声音调高", "调低一点"): {
            "name": "VoiceControl",
            "arguments": {"action": "调低", "feature": "导航音量"},
        },
        ("打开车窗", "打开大灯", "导航声音调高", "再低一点"): {
            "name": "VoiceControl",
            "arguments": {"action": "调低", "feature": "导航音量"},
        },
        ("主驾座椅加热打开", "副驾座椅通风打开", "主驾按摩打开", "调成波浪模式"): {
            "name": "SeatControl",
            "arguments": {"action": "调到", "device": "座椅", "feature": "按摩", "position": "主驾", "value": "波浪"},
        },
        ("空调温度调到二十四度", "媒体声音调高", "副驾屏幕亮度调高", "再暗一点"): {
            "name": "ScreenControl",
            "arguments": {"action": "调低", "device": "娱乐屏", "feature": "亮度", "position": "副驾"},
        },
        ("香氛浓度调高", "雨刮速度调高", "后雨刮灵敏度调高", "再低一点"): {
            "name": "WiperControl",
            "arguments": {"action": "调低", "device": "后雨刮", "feature": "灵敏度", "value": "低"},
        },
        ("导航声音调高", "空调风量调高", "香氛浓度调高", "再淡一点"): {
            "name": "PerfumeControl",
            "arguments": {"action": "调低", "device": "香氛", "feature": "浓度"},
        },
        ("氛围灯调成光剑", "屏幕调到黑夜模式", "娱乐屏调到白天模式", "切到黑夜模式"): {
            "name": "ScreenControl",
            "arguments": {"action": "调到", "device": "娱乐屏", "value": "黑夜模式"},
        },
        ("前排空调风量调高", "后排空调温度调到二十六度", "第三排空调打开", "温度调低一点"): {
            "name": "ClimateControl",
            "arguments": {"action": "调低", "device": "空调", "feature": "温度", "position": "第三排"},
        },
        ("主驾车窗打开一点", "副驾车窗打开一点", "第二排右侧车窗打开一点", "再开一点"): {
            "name": "WindowControl",
            "arguments": {"action": "再开", "device": "车窗", "position": "第二排右侧"},
        },
        ("左侧侧滑门打开", "右侧鹏翼门打开", "后备箱打开", "暂停一下"): {
            "name": "GateControl",
            "arguments": {"action": "暂停", "device": "后备箱"},
        },
        ("自动雨刮打开", "后雨刮打开", "雨刮速度调到中档", "调到最高"): {
            "name": "WiperControl",
            "arguments": {"action": "调到", "device": "雨刮", "feature": "速度", "value": "最高"},
        },
        ("媒体声音调低", "导航声音调低", "语音音量调低", "再大一点"): {
            "name": "VoiceControl",
            "arguments": {"action": "调高", "feature": "语音音量"},
        },
        ("打开空气净化器", "打开外循环", "打开前挡风除雾", "关掉吧"): {
            "name": "ClimateControl",
            "arguments": {"action": "关闭", "device": "前挡风", "feature": "除雾"},
        },
    }
    observed = {}

    for line in MULTITURN_PATH.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        users = [message["content"].strip() for message in sample["messages"] if message["role"] == "user"]
        for expected_users in expected_final_calls:
            if tuple(users[-len(expected_users) :]) != expected_users:
                continue
            observed[expected_users] = json.loads(sample["messages"][-1]["content"])

    assert observed == expected_final_calls


def test_multiturn_current_query_overrides_similar_history():
    expected_final_queries = {
        "那主驾车窗呢": "主驾车窗",
        "副驾温度多少": "副驾空调温度",
        "导航播报音量是多少": "导航播报音量",
        "空调开着吗": "用户空调",
        "剩余续航多少": "剩余续航里程",
        "副驾座椅通风开着吗": "副驾座椅通风",
        "前除雾开了吗": "前除雾",
        "车内空气质量怎么样": "车内空气质量",
        "左侧儿童锁呢": "车门左儿童锁状态",
        "近光灯开了吗": "近光灯开关",
    }
    observed = {}

    for line in MULTITURN_PATH.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        users = [message["content"].strip() for message in sample["messages"] if message["role"] == "user"]
        final_user = users[-1]
        if final_user not in expected_final_queries:
            continue

        tool_call = json.loads(sample["messages"][-1]["content"])
        observed[final_user] = tool_call.get("arguments", {}).get("query")
        assert tool_call.get("name") == "CarUsageSearch"

    assert observed == expected_final_queries
