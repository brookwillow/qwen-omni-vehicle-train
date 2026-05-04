"""Post-processing for parsed tool calls."""

from __future__ import annotations

import re
from typing import Any


SEAT_POSITION_TERMS = (
    "主驾",
    "驾驶位",
    "司机位",
    "副驾",
    "副驾驶",
    "前排",
    "后排",
    "第一排",
    "第二排",
    "第三排",
    "左侧",
    "右侧",
    "中间",
)


def _mentions_seat_position(query: str) -> bool:
    return any(term in query for term in SEAT_POSITION_TERMS)


def _infer_window_position(query: str) -> str | None:
    if any(term in query for term in ("所有", "全部", "都")):
        return "全部"
    if "主驾" in query:
        return "主驾"
    if "副驾" in query or "副驾驶" in query:
        return "副驾"
    if "前排" in query:
        return "前排"
    if "第二排左" in query or "后排左" in query:
        return "第二排左侧"
    if "第二排右" in query or "后排右" in query:
        return "第二排右侧"
    if "第三排左" in query:
        return "第三排左侧"
    if "第三排右" in query:
        return "第三排右侧"
    if "左边" in query or "左侧" in query:
        return "左侧"
    if "右边" in query or "右侧" in query:
        return "右侧"
    if "第二排" in query:
        return "第二排"
    if "第三排" in query:
        return "第三排"
    return None


def _infer_common_position(query: str) -> str | None:
    if any(term in query for term in ("所有", "全部", "全车", "都")):
        return "全部"
    if any(term in query for term in ("我这边", "主驾", "驾驶位", "司机位")):
        return "主驾"
    if any(term in query for term in ("副驾", "副驾驶")):
        return "副驾"
    if "前排" in query:
        return "前排"
    if "后排左" in query or "第二排左" in query or "左后" in query:
        return "第二排左侧"
    if "后排右" in query or "第二排右" in query or "右后" in query:
        return "第二排右侧"
    if "后排" in query or "后座" in query or "后面" in query:
        return "第二排"
    if "第三排左" in query:
        return "第三排左侧"
    if "第三排右" in query:
        return "第三排右侧"
    if "第三排" in query:
        return "第三排"
    if "左边" in query or "左侧" in query:
        return "左侧"
    if "右边" in query or "右侧" in query:
        return "右侧"
    return None


def _query_has_small_delta(query: str) -> bool:
    return any(term in query for term in ("一点", "一点点", "一些", "稍微", "一抬", "一下高度", "轻一点", "大点", "小点"))


def _query_mentions_close(query: str) -> bool:
    return any(term in query for term in ("关闭", "关掉", "关上", "关了", "关掉", "灭了", "合上", "盖上"))


def _query_mentions_open(query: str) -> bool:
    return any(term in query for term in ("打开", "开启", "开一下", "开了", "开开", "开起来"))


def _query_mentions_no_need(query: str) -> bool:
    return any(term in query for term in ("不需要", "不用", "别", "不要"))


def _first_command(query: str) -> str:
    return re.split(r"[，,。；;]|顺便|然后|再把|也", query, maxsplit=1)[0]


def _infer_percent_value(query: str) -> str | None:
    percent_match = re.search(r"(\d{1,3})\s*%", query)
    if percent_match:
        return f"{percent_match.group(1)}%"
    chinese_percent = {
        "百分之三十": "30%",
        "百分之五十": "50%",
        "百分之六十": "60%",
        "百分之八十": "80%",
    }
    for text, value in chinese_percent.items():
        if text in query:
            return value
    if "一半" in query or "半" in query:
        return "50%"
    return None


def _infer_empty_window_args(query: str) -> dict[str, Any]:
    if not any(term in query for term in ("车窗", "窗户", "窗")):
        return {}

    first_part = re.split(r"[，,。；;]|顺便|然后|再把|也", query, maxsplit=1)[0]
    fixed: dict[str, Any] = {"device": "车窗"}
    position = _infer_window_position(first_part)
    if position:
        fixed["position"] = position

    percent_match = re.search(r"(\d{1,3})\s*%", first_part)
    if percent_match:
        fixed["action"] = "开到" if any(term in first_part for term in ("开", "打开", "摇下")) else "关到"
        fixed["value"] = f"{percent_match.group(1)}%"
    elif "一半" in first_part or "半" in first_part:
        fixed["action"] = "开到" if any(term in first_part for term in ("开", "打开", "摇下")) else "关到"
        fixed["value"] = "50%"
    elif "一条缝" in first_part:
        fixed["action"] = "开到"
        fixed["value"] = "10%"
    elif any(term in first_part for term in ("关闭", "关上", "关了", "关掉")):
        fixed["action"] = "关闭"
    elif any(term in first_part for term in ("打开", "开窗", "开一下", "开了", "摇下来", "通风")):
        fixed["action"] = "打开"

    if "action" not in fixed:
        return {}
    return fixed


def _extract_light_color(query: str) -> str | None:
    colors = ("暖黄色", "橙色", "红色", "蓝色", "紫色", "绿色", "白色")
    for color in colors:
        if color in query:
            return color
    short_colors = {"红": "红色", "蓝": "蓝色", "紫": "紫色", "绿": "绿色", "白": "白色"}
    for short, color in short_colors.items():
        if short in query:
            return color
    return None


def _driving_mode_value(query: str) -> str | None:
    if "标准" in query or "默认" in query or "正常驾驶" in query:
        return "标准模式"
    if "节能" in query or "省点电" in query or "撑过去" in query or "电量不" in query:
        return "节能模式"
    if "运动" in query or "推背" in query or "动力响应" in query:
        return "运动模式"
    if "舒适" in query or "平稳" in query or "悠着点" in query:
        return "舒适模式"
    if "脱困" in query or "泥地" in query or "被困" in query or "出不来" in query:
        return "脱困模式"
    if "弹射" in query or "第一个冲出去" in query:
        return "弹射模式"
    if "自定义" in query or "自己调" in query:
        return "自定义"
    return None


def _steering_args_from_query(query: str) -> dict[str, Any] | None:
    if not any(term in query for term in ("方向盘", "转向")):
        return None
    fixed: dict[str, Any] = {"device": "方向盘"}
    if "助力" in query or "转向" in query or any(term in query for term in ("太沉", "太重", "太轻", "调轻", "轻一些", "最轻")):
        fixed["feature"] = "助力"
    else:
        fixed["feature"] = "制热"
    if _query_mentions_close(query) or ("加热" in query and _query_mentions_no_need(query)) or "太烫" in query:
        fixed["action"] = "关闭"
    elif any(term in query for term in ("打开", "开启", "热起来")):
        fixed["action"] = "打开"
    elif any(term in query for term in ("切换到", "切", "调到", "设成", "设为", "设置为", "开到")):
        fixed["action"] = "调到"
    elif any(term in query for term in ("调大", "加大", "不够暖", "沉一点", "太轻")):
        fixed["action"] = "调大"
    elif any(term in query for term in ("调小", "小一些", "轻一点", "减轻", "太重", "太沉", "烫手")):
        fixed["action"] = "调小"
    else:
        fixed["action"] = "打开"

    mode_map = {
        "舒适模式": "舒适模式",
        "适中模式": "适中模式",
        "运动模式": "运动模式",
        "轻盈模式": "轻盈模式",
        "标准模式": "标准模式",
        "稳重模式": "稳重模式",
        "最轻": "最小",
        "最小": "最小",
        "最大": "最大",
        "最猛": "最大",
        "中档": "中",
        "三档": "3",
        "3档": "3",
        "大档": "大",
        "小档": "小",
        "最低档": "最小",
        "中间那个档": "中",
        "不轻不重": "适中模式",
        "默认": "标准模式",
        "驾驶感": "运动模式",
    }
    for text, value in mode_map.items():
        if text in query:
            fixed["value"] = value
            break
    if "调小一点" in query or "调大一点" in query or "加大一点" in query:
        fixed["value"] = "一点"
    if fixed["action"] in {"打开", "关闭"} and fixed.get("value") in {"一点", "中"}:
        fixed.pop("value", None)
    return fixed


def _infer_camera_position(query: str) -> str | None:
    if any(term in query for term in ("前方", "前面", "前侧")):
        return "前侧"
    if "左" in query:
        return "左侧"
    if "右" in query:
        return "右侧"
    if any(term in query for term in ("车尾", "后方", "后面", "后侧")):
        return "后侧"
    if any(term in query for term in ("所有", "全部", "都", "车外面", "车周围", "周围")):
        return "全部"
    return None


def postprocess_action_call(
    query: str,
    tool: str | None,
    args: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Fix high-confidence tool confusions before normal argument cleanup."""
    query = query or ""
    first_part = _first_command(query)
    fixed_tool = tool
    fixed_args = dict(args) if isinstance(args, dict) else args

    if any(term in first_part for term in ("车窗锁", "窗户锁", "乱按车窗", "窗户不要让孩子碰", "窗户锁上", "自己控制窗户")):
        fixed_tool = "LockControl"
        fixed_args = {"action": "关闭" if any(term in first_part for term in ("可以自己控制", "自己控制")) else "打开", "device": "车窗锁"}
    elif "车窗" in first_part or "窗户" in first_part:
        fixed_tool = "WindowControl"
        if "暂停" in first_part:
            action = "暂停"
        else:
            action = "关闭" if _query_mentions_close(first_part) else "打开"
        fixed_args = {"action": action, "device": "车窗"}
        position = _infer_window_position(first_part)
        if position:
            fixed_args["position"] = position
    elif "离方向盘太远" in first_part:
        fixed_tool = "SeatControl"
        fixed_args = {"action": "调前", "device": "座椅", "feature": "位置"}
    elif first_part.startswith(("音量", "声音", "通话音量", "导航的声音", "导航声音")) or any(
        term in first_part for term in ("全车安静", "别出声", "车外的喇叭", "后面的声音")
    ) or ("声音" in first_part and _query_mentions_close(first_part)):
        fixed_tool = "VoiceControl"
        fixed_args = fixed_args if isinstance(fixed_args, dict) else {}
        if "通话音量" in first_part:
            fixed_args = {"action": "调到" if "调到" in first_part else "调小", "feature": "通话音量"}
        elif "导航" in first_part:
            fixed_args = {"action": "关闭", "feature": "导航音量"}
        elif "关" in first_part or "安静" in first_part or "别出声" in first_part:
            fixed_args = {"action": "关闭", "feature": "声音" if "全车" in first_part or "别出声" in first_part else "音量"}
        else:
            fixed_args = {"action": "调小" if "小" in first_part or "低" in first_part else "调到", "feature": "音量"}
        if "最大" in first_part:
            fixed_args["value"] = "最大"
        if "百分之二十" in first_part or "20%" in first_part:
            fixed_args["value"] = "20%"
    elif first_part.startswith("屏幕") or "屏幕亮度" in first_part:
        fixed_tool = "ScreenControl"
        screen_modes = ("清洁模式", "白天模式", "等人模式", "自动模式", "黑夜模式")
        mode = next((item for item in screen_modes if item in first_part), None)
        if mode:
            fixed_args = {"action": "调到", "device": "屏幕", "value": mode}
        else:
            action = "调小" if "调小" in first_part or "太亮" in first_part or "刺眼" in first_part else "调大"
            if "调到" in first_part or "最" in first_part:
                action = "调到"
            fixed_args = {"action": action, "device": "屏幕", "feature": "亮度"}
            if "最大" in first_part:
                fixed_args["value"] = "最大"
            elif "最小" in first_part:
                fixed_args["value"] = "最小"
            elif "一点" in first_part and "然后" not in first_part:
                fixed_args["value"] = "一点"
    elif "阅读灯" in first_part:
        fixed_tool = "LightControl"
        fixed_args = {"action": "关闭" if _query_mentions_close(first_part) else "打开", "device": "阅读灯"}
    elif "冰箱" in first_part:
        fixed_tool = "FridgeControl"
        fixed_args = {"action": "关闭" if _query_mentions_close(first_part) else "打开", "device": "冰箱"}
        if "二排" in first_part or "第二排" in first_part:
            fixed_args["position"] = "第二排"
    elif "方向盘" in first_part or first_part.startswith("转向"):
        steering_args = _steering_args_from_query(query)
        if steering_args:
            fixed_tool = "SteeringwheelControl"
            fixed_args = steering_args
    elif "加热" in first_part and "手" in first_part and any(term in first_part for term in ("冷", "冻", "暖")):
        fixed_tool = "SteeringwheelControl"
        fixed_args = {"action": "打开", "device": "方向盘", "feature": "制热"}
        if "最猛" in first_part or "最大" in first_part:
            fixed_args.update({"action": "调到", "value": "最大"})
    elif (
        "节能模式" in first_part
        and "驾驶" not in first_part
        and not _query_mentions_close(first_part)
        and not any(term in first_part for term in ("电量", "省点电", "撑过去"))
    ):
        fixed_tool = "ClimateControl"
        fixed_args = {"action": "打开", "device": "空调", "value": "节能模式"}
    elif any(term in first_part for term in ("风吹得太大", "风太大", "风太小", "风量太小", "风量太大")):
        fixed_tool = "ClimateControl"
        fixed_args = {
            "action": "调小" if "大" in first_part else "调大",
            "device": "空调",
            "feature": "风",
            "value": "一点",
        }
    elif any(term in first_part for term in ("不想吹那么冷", "别对着我吹", "外面空气不好", "狗在车里", "副驾那边有点冷")):
        fixed_tool = "ClimateControl"
        if "不想吹那么冷" in first_part:
            fixed_args = {"action": "打开", "device": "空调", "value": "自然风"}
        elif "别对着我吹" in first_part:
            fixed_args = {"action": "打开", "device": "空调", "value": "避人"}
        elif "外面空气不好" in first_part:
            fixed_args = {"action": "打开", "device": "空调", "value": "内循环"}
        elif "狗在车里" in first_part:
            fixed_args = {"action": "打开", "device": "空调", "value": "宠物模式"}
        else:
            fixed_args = {"action": "调大", "device": "空调", "feature": "温度", "position": "副驾", "value": "一点"}
    elif (
        "空调" not in first_part
        and any(term in first_part for term in ("驾驶模式", "节能模式", "运动模式", "舒适模式", "标准模式", "脱困模式", "弹射模式"))
    ) or any(
        term in first_part for term in ("省点电", "推背感", "动力响应", "悠着点开", "第一个冲出去", "回到正常驾驶", "平稳一些", "泥地", "被困")
    ):
        mode = _driving_mode_value(first_part)
        if mode:
            fixed_tool = "DrivingControl"
            fixed_args = {
                "action": "关闭" if _query_mentions_close(first_part) and "回到正常" not in first_part else "打开",
                "feature": "驾驶模式",
                "value": mode,
            }
    elif any(term in query for term in ("哨兵", "停车守护", "车辆监控", "监控一下", "盯着点车", "看着车", "别被剐蹭")):
        fixed_tool = "CameraControl"
        fixed_args = {
            "action": "关闭" if _query_mentions_close(query) or "不用再监控" in query else "打开",
            "device": "摄像头",
            "value": "哨兵模式",
        }
    elif any(term in first_part for term in ("车周围", "车头前面", "右边车道", "左边是不是有人", "并线前", "帮我看看前面")):
        fixed_tool = "CameraControl"
        fixed_args = {"action": "打开", "device": "摄像头"}
        position = _infer_camera_position(first_part)
        if position:
            fixed_args["position"] = position
    elif "车外面" in first_part and any(term in first_part for term in ("什么情况", "看看", "看一下")):
        fixed_tool = "CameraControl"
        fixed_args = {"action": "打开", "device": "摄像头", "position": "全部"}
    elif any(term in first_part for term in ("我想听歌", "来点音乐", "打开音乐应用", "进音乐应用")):
        fixed_tool = "AppControl"
        fixed_args = {"action": "打开", "feature": "音乐应用"}
    elif "不听歌" in first_part and any(term in first_part for term in ("关掉音乐", "关闭音乐")):
        fixed_tool = "AppControl"
        fixed_args = {"action": "关闭", "feature": "音乐应用"}
    elif "关闭音乐应用" in first_part or "关掉音乐应用" in first_part:
        fixed_tool = "AppControl"
        fixed_args = {"action": "关闭", "feature": "音乐应用"}
    elif "关闭音乐" in first_part or "关掉音乐" in first_part:
        fixed_tool = "MediaControl"
        fixed_args = {"media_category": "歌", "media_control_action": "关闭"}
    elif "关闭收音机" in first_part:
        fixed_tool = "MediaControl"
        fixed_args = {"media_category": "FM", "media_control_action": "关闭"}
    elif any(term in first_part for term in ("别放了", "暂停播放")):
        fixed_tool = "MediaControl"
        fixed_args = {"media_control_action": "暂停"}
    elif any(term in first_part for term in ("安静一下不想听", "不想听了")):
        fixed_tool = "MediaControl"
        fixed_args = {"media_control_action": "关闭"}
    elif any(term in first_part for term in ("下一个FM频道", "下个FM频道")):
        fixed_tool = "MediaControl"
        fixed_args = {"media_category": "FM", "media_control_action": "下一个"}
    elif "继续播放有声书" in first_part:
        fixed_tool = "MediaControl"
        fixed_args = {"media_category": "读物", "media_control_action": "继续"}
    elif "Spotify" in first_part and "下一首" in first_part:
        fixed_tool = "MediaControl"
        fixed_args = {"app_name": "spotify", "media_control_action": "下一个"}
    elif "播放FM广播" in first_part or "想听听广播" in first_part:
        fixed_tool = "MediaPlay"
        fixed_args = {"media_category": "FM"}
    elif "蓝牙音源" in first_part:
        fixed_tool = "MediaPlay"
        fixed_args = {"media_source": "蓝牙"}
    elif ("油管" in first_part or "YouTube" in first_part) and any(term in first_part for term in ("打开", "看视频", "看看")):
        fixed_tool = "MediaPlay"
        fixed_args = {"app_name": "youtube", "media_category": "视频"}
    elif re.search(r"(播放|放).*周杰伦.*(晴天|稻香)", first_part):
        song = "周杰伦 晴天" if "晴天" in first_part else "周杰伦 稻香"
        fixed_tool = "MusicSearchPlay"
        fixed_args = {"song": song}
    elif "孤勇者" in first_part and any(term in first_part for term in ("听", "播放", "放")):
        fixed_tool = "MusicSearchPlay"
        fixed_args = {"song": "孤勇者"}
    elif "七里香" in first_part:
        fixed_tool = "MusicSearchPlay"
        fixed_args = {"song": "七里香"}
    elif "陈奕迅" in first_part and "十年" in first_part:
        fixed_tool = "MusicSearchPlay"
        fixed_args = {"song": "陈奕迅 十年"}
    elif "林俊杰" in first_part and "歌" in first_part:
        fixed_tool = "MusicSearchPlay"
        fixed_args = {"song": "林俊杰"}
    elif "TuneIn" in first_part and "Hotel California" in first_part:
        fixed_tool = "MusicSearchPlay"
        fixed_args = {"app_name": "tunein", "song": "Hotel California"}
    elif "鬼吹灯" in first_part:
        fixed_tool = "XmlySearchPlay"
        fixed_args = {"podcast_name": "鬼吹灯"}
    elif "FM88.1" in first_part or "FM88" in first_part:
        fixed_tool = "FmSearchPlay"
        fixed_args = {"fm_channel": "88.1", "media_category": "FM"}
    elif "苹果" in first_part and "音乐" in first_part and any(term in first_part for term in ("放点歌", "放歌", "听歌")):
        fixed_tool = "MediaPlay"
        fixed_args = {"app_name": "applemusic", "media_category": "歌"}
    elif "空调" in first_part and any(term in first_part for term in ("面板打开", "打开面板", "调一下空调参数")):
        fixed_tool = "AppControl"
        fixed_args = {"action": "打开", "feature": "空调"}
    elif any(term in first_part for term in ("车辆状态", "车辆控制的那个界面", "车控界面")):
        fixed_tool = "AppControl"
        fixed_args = {"action": "打开", "feature": "车控"}
    elif "通话界面" in first_part and _query_mentions_close(first_part):
        fixed_tool = "AppControl"
        fixed_args = {"action": "关闭", "feature": "蓝牙电话"}
    elif any(term in first_part for term in ("路上车多不多", "会不会很堵", "堵不堵")):
        fixed_tool = "InfoQuery"
        fixed_args = {"feature": "路况信息"}
    elif any(term in first_part for term in ("显示我在哪个位置", "我在哪个位置", "当前位置")):
        fixed_tool = "InfoQuery"
        fixed_args = {"feature": "当前位置"}
    elif "换换空气" in first_part:
        fixed_tool = "WindowControl"
        fixed_args = {"action": "打开", "device": "车窗"}
    elif any(term in first_part for term in ("隔绝一下外面的声音", "外面灰太大", "窗户关上")):
        fixed_tool = "WindowControl"
        fixed_args = {"action": "关闭", "device": "车窗"}
    elif any(term in first_part for term in ("暂停车窗", "暂停窗户")):
        fixed_tool = "WindowControl"
        fixed_args = {"action": "暂停", "device": "车窗"}
    elif any(term in first_part for term in ("开一条缝", "小口透气")):
        fixed_tool = "WindowControl"
        fixed_args = {"action": "开到", "device": "车窗", "value": "10%"}
    elif any(term in first_part for term in ("进高速", "准备睡一会儿")) and any(term in first_part for term in ("关上窗", "窗户关上")):
        fixed_tool = "WindowControl"
        fixed_args = {"action": "关闭", "device": "车窗", "position": "全部"}
    elif any(term in first_part for term in ("我这边", "老婆那边")) and "窗" in first_part:
        fixed_tool = "WindowControl"
        fixed_args = {"action": "开到", "device": "车窗", "position": "主驾", "value": "50%"}
    elif any(term in first_part for term in ("阳光太刺眼", "不想被外面看到", "后排晒", "留一点光线", "遮阳")):
        fixed_tool = "WindowControl"
        fixed_args = {"action": "关闭", "device": "遮阳帘"}
        if "不需要遮阳" in first_part or "天黑" in first_part:
            fixed_args["action"] = "打开"
        if "留一点光线" in first_part:
            fixed_args.update({"action": "再开", "value": "一点"})
        if "后排" in first_part:
            fixed_args["position"] = "第二排"
    elif re.search(r"(查|找|搜).*1\d{2,}", first_part):
        fixed_tool = "PhoneControl"
        m = re.search(r"(1\d{2,11})", first_part)
        fixed_args = {"action": "搜索", "telephone": m.group(1) if m else ""}
    elif any(term in first_part for term in ("回拨", "未接来电", "没接到", "打回去", "回过去")):
        fixed_tool = "PhoneControl"
        fixed_args = {"action": "回拨"}
    elif any(term in first_part for term in ("再试一次", "再拨一遍", "再来一次")):
        fixed_tool = "PhoneControl"
        fixed_args = {"action": "重拨"}
    elif any(term in first_part for term in ("通讯录", "电话号码", "搜索", "搜一下", "找下")) and any(
        term in first_part for term in ("电话", "号码", "小王", "弟弟", "王总", "王经理", "老板", "外卖店", "姓周", "妈妈", "陈总", "李总", "老婆", "前台", "孙师傅")
    ):
        fixed_tool = "PhoneControl"
        fixed_args = {"action": "搜索"}
        if "姓周" in first_part:
            fixed_args["person"] = "周"
        elif "小王" in first_part:
            fixed_args["person"] = "小王"
        elif "弟弟" in first_part:
            fixed_args["person"] = "弟弟"
        elif "王总" in first_part:
            fixed_args["person"] = "王总"
        elif "王经理" in first_part:
            fixed_args["person"] = "王经理"
        elif "老板" in first_part:
            fixed_args["person"] = "老板"
        elif "外卖店" in first_part:
            fixed_args["person"] = "外卖店"
        elif "妈妈" in first_part:
            fixed_args["person"] = "妈妈"
        elif "陈总" in first_part:
            fixed_args["person"] = "陈总"
        elif "李总" in first_part:
            fixed_args["person"] = "李总"
        elif "老婆" in first_part:
            fixed_args["person"] = "老婆"
        elif "前台" in first_part:
            fixed_args["person"] = "前台"
        elif "孙师傅" in first_part:
            fixed_args["person"] = "孙师傅"

    return fixed_tool, postprocess_action_args(query, fixed_tool, fixed_args)


def postprocess_action_args(query: str, tool: str | None, args: dict[str, Any] | None) -> dict[str, Any] | None:
    """Fix systematic parser/model artifacts without changing explicit user intent."""
    if not isinstance(args, dict):
        return args

    fixed = dict(args)
    query = query or ""
    if tool == "AppControl":
        first_part = _first_command(query)
        if any(term in first_part for term in ("照片", "视频", "车上拍")):
            fixed.update({"action": "打开", "feature": "智能影像"})
    elif tool == "CameraControl":
        first_part = _first_command(query)
        camera_part = first_part if "摄像头" in first_part or "车周围" in first_part or "车外" in first_part else query
        fixed["device"] = "摄像头"
        if _query_mentions_close(camera_part):
            fixed["action"] = "关闭"
        elif _query_mentions_open(camera_part) or any(term in camera_part for term in ("看看", "看一下", "监控")):
            fixed["action"] = "打开"
        if "行车记录仪" in camera_part:
            fixed["device"] = "摄像头"
        if "哨兵模式" in camera_part or any(term in camera_part for term in ("不太安全", "监控一下", "停车守护", "车辆监控", "看着车", "剐蹭")):
            fixed["value"] = "哨兵模式"
            fixed.pop("position", None)
        else:
            position = _infer_camera_position(camera_part)
            if position:
                fixed["position"] = position
            fixed.pop("value", None)
    elif tool == "DrivingControl":
        first_part = _first_command(query)
        if any(term in first_part for term in ("默认", "标准", "正常")):
            fixed.update({"action": "打开", "feature": "驾驶模式", "value": "标准模式"})
        if any(term in first_part for term in ("第一个冲出去", "弹射")):
            fixed.update({"action": "打开", "feature": "驾驶模式", "value": "弹射模式"})
    elif tool == "FridgeControl":
        first_part = _first_command(query)
        if "冰箱" in first_part and not fixed:
            fixed.update({"action": "打开", "device": "冰箱"})
        if "冰箱" in first_part:
            fixed.setdefault("device", "冰箱")
        if "不用" in first_part or "关掉" in first_part:
            fixed["action"] = "关闭"
        position = _infer_common_position(first_part.replace("二排", "第二排"))
        if position in {"第二排", "第三排"}:
            fixed["position"] = position
    elif tool == "SeatControl":
        first_part = _first_command(query)
        if fixed.get("position") == "主驾" and not _mentions_seat_position(query):
            fixed.pop("position", None)
        if fixed.get("action") == "调高":
            fixed["action"] = "调大"
        if fixed.get("action") == "收平":
            fixed["action"] = "收起"
        if fixed.get("value") in {"最前", "最后"}:
            fixed["action"] = "调到"
            fixed.setdefault("feature", "位置")
        if fixed.get("device") == "座椅" and fixed.get("action") in {"调前", "调后"}:
            fixed.setdefault("feature", "位置")
        if fixed.get("action") == "切换" and fixed.get("value"):
            fixed["action"] = "调到"
        if fixed.get("action") in {"调前", "调后"} and fixed.get("value") == "一点" and "一点" not in first_part:
            fixed.pop("value", None)
        if fixed.get("feature") == "按摩" and fixed.get("action") in {"调大", "调小"} and _query_has_small_delta(first_part):
            fixed["value"] = "一点"
        if fixed.get("device") == "靠背" and fixed.get("action") in {"调前", "调后"} and any(term in first_part for term in ("一点", "一些")):
            fixed["value"] = "一点"
        if fixed.get("action") == "打开":
            fixed.pop("value", None)
        if "太高" in first_part or "顶到车顶" in first_part:
            fixed.update({"action": "调小", "device": "座椅", "feature": "高度", "value": "一点"})
        if "太躺" in first_part or "直起来" in first_part:
            fixed.update({"action": "调前", "device": "靠背", "value": "一点"})
            fixed.pop("feature", None)
        if "长途解乏" in query:
            fixed.update({"action": "调到", "device": "座椅", "feature": "按摩", "value": "长途解乏"})
        if "靠背" in first_part:
            fixed["device"] = "靠背"
            if fixed.get("feature") == "位置":
                fixed.pop("feature", None)
        if "放平" in first_part:
            fixed["action"] = "放平"
        position = _infer_common_position(first_part)
        if position:
            fixed["position"] = position
        if "加热" in first_part:
            fixed["feature"] = "制热"
            if _query_mentions_close(first_part):
                fixed["action"] = "关闭"
            elif any(term in first_part for term in ("太猛", "调低", "低一些")):
                fixed["action"] = "调小"
                fixed["value"] = "一点"
        if "按摩" in first_part:
            fixed["feature"] = "按摩"
            if _query_mentions_close(first_part):
                fixed["action"] = "关闭"
        if "加热" in first_part and "按摩" in first_part and first_part.find("加热") < first_part.find("按摩"):
            fixed["feature"] = "制热"
        if "通风" in first_part and "加热" not in first_part and "按摩" not in first_part:
            fixed["feature"] = "通风"
    elif tool == "WindowControl":
        if not fixed:
            fixed = _infer_empty_window_args(query)
        value = fixed.get("value")
        if isinstance(value, str) and value.isdigit() and f"{value}%" in query:
            fixed["value"] = f"{value}%"
        first_part = _first_command(query)
        position = _infer_window_position(first_part)
        if position:
            fixed["position"] = position
        if "我这边" in first_part:
            fixed["position"] = "主驾"
        if "前排" in first_part:
            fixed["position"] = "前排"
        if fixed.get("device") == "遮阳帘":
            if "再开" in first_part:
                fixed["action"] = "再开"
                fixed["value"] = "一点"
            elif "再关" in first_part:
                fixed["action"] = "再关"
                fixed["value"] = "一点"
            elif "开到" in first_part:
                fixed["action"] = "开到"
            elif "关到" in first_part:
                fixed["action"] = "关到"
            elif _query_mentions_open(first_part):
                fixed["action"] = "打开"
            elif any(term in first_part for term in ("晒", "刺眼", "挡", "遮住", "不想被外面看到")):
                fixed["action"] = "关到" if "一半" in first_part or "半" in first_part else "关闭"
            elif "不需要遮阳" in first_part or "不用遮阳" in first_part:
                fixed["action"] = "打开"
        if fixed.get("device") == "车窗":
            if "再开" in first_part:
                fixed["action"] = "再开"
                fixed["value"] = "一点"
            elif "再关" in first_part:
                fixed["action"] = "再关"
                fixed["value"] = "一点"
            elif "关到" in first_part:
                fixed["action"] = "关到"
            elif "开到" in first_part:
                fixed["action"] = "开到"
            elif "一条缝" in first_part or "小口" in first_part:
                fixed["action"] = "开到"
                fixed["value"] = "10%"
            elif "开太大" in first_part:
                fixed["action"] = "再关"
                fixed["value"] = "一点"
            elif "摇下来一点" in first_part:
                fixed["action"] = "再开"
                fixed["value"] = "一点"
        percent_value = _infer_percent_value(first_part)
        if percent_value and fixed.get("value") in {None, "一半"}:
            fixed["value"] = percent_value
            if fixed.get("action") not in {"关到", "开到"}:
                if fixed.get("device") == "车窗":
                    fixed["action"] = "开到"
                elif fixed.get("device") == "遮阳帘":
                    fixed["action"] = "关到"
        if fixed.get("position") and not _mentions_seat_position(first_part) and fixed.get("position") != "全部":
            fixed.pop("position", None)
        if any(term in first_part for term in ("进高速", "关上窗")) and fixed.get("device") == "车窗":
            fixed["position"] = "全部"
        if "下雨" in first_part and fixed.get("device") == "车窗":
            fixed["position"] = "全部"
    elif tool == "LightControl":
        first_part = _first_command(query)
        if _query_mentions_close(first_part):
            fixed["action"] = "关闭"
        if "远光灯" in first_part:
            fixed["device"] = "远光灯"
        if "前面太暗" in first_part or "大灯不够亮" in first_part:
            fixed.update({"action": "打开", "device": "远光灯"})
            fixed.pop("value", None)
        if "不用开灯" in first_part or "天亮了" in first_part:
            fixed.update({"action": "关闭", "device": "大灯"})
        if "切换到自动" in first_part or "调成自动" in first_part or "自动模式" in first_part:
            fixed["action"] = "调到"
            fixed["device"] = "大灯"
            fixed["value"] = "自动"
        if "自己控制" in first_part:
            fixed.update({"action": "调到", "device": "大灯", "value": "自动"})
        if fixed.get("device") in {"雾灯", "尾灯"}:
            fixed["device"] = "前雾灯" if "前" in first_part or "雾灯" in first_part else fixed["device"]
        if fixed.get("device") == "小灯":
            fixed["device"] = "示宽灯"
        if "小灯" in first_part:
            fixed["device"] = "示宽灯"
        if "阅读灯" in first_part:
            fixed["device"] = "阅读灯"
        if "前雾灯" in first_part:
            fixed["device"] = "前雾灯"
        if "后雾灯" in first_part and "前雾灯" not in first_part and "前后雾灯" not in first_part:
            fixed["device"] = "后雾灯"
        if "雾灯" in first_part and ("后雾灯" not in first_part or "前后雾灯" in first_part):
            fixed["device"] = "前雾灯"
        if "车内灯光" in first_part or "阅读灯" in first_part or "看书" in first_part or "要睡觉" in first_part:
            fixed["device"] = "阅读灯"
            if "主驾" in first_part:
                position = "主驾"
            elif "副驾" in first_part:
                position = "副驾"
            else:
                position = _infer_common_position(first_part)
            if position:
                fixed["position"] = position
        if "头顶" in first_part and "灯" in first_part:
            fixed["device"] = "阅读灯"
        if "氛围灯" in first_part and "车内灯光" not in first_part:
            fixed["device"] = "氛围灯"
            position = _infer_common_position(first_part)
            if position:
                fixed["position"] = position
            color = _extract_light_color(first_part)
            if color and first_part.startswith(("打开", "开")):
                fixed["action"] = "打开"
                fixed.pop("feature", None)
                fixed.pop("value", None)
            elif color:
                fixed["feature"] = "颜色"
                fixed["value"] = color
                fixed["action"] = "调到"
            elif "暖一点" in first_part:
                fixed.update({"action": "调到", "feature": "颜色", "value": "橙色"})
            elif "光剑" in first_part:
                fixed["action"] = "调到"
                fixed["value"] = "光剑"
            elif "科技感" in first_part or "灯光效果" in first_part:
                fixed["action"] = "调到"
                fixed["value"] = "光剑"
            elif "换个颜色" in first_part or "颜色" in first_part:
                fixed["feature"] = "颜色"
                if fixed.get("action") == "切换":
                    fixed["value"] = "下一个"
            elif fixed.get("value") in {"下一个", "xx色"} and fixed.get("action") == "打开":
                fixed.pop("value", None)
            if fixed.get("position") and not _mentions_seat_position(first_part):
                fixed.pop("position", None)
        elif fixed.get("device") == "氛围灯" and ("科技感" in first_part or "灯光效果" in first_part):
            fixed["action"] = "调到"
            fixed["value"] = "光剑"
        elif fixed.get("device") == "氛围灯" and "暖一点" in first_part:
            fixed.update({"action": "调到", "feature": "颜色", "value": "橙色"})
            fixed.pop("position", None)
        if "示宽灯" in first_part and "大灯" not in first_part:
            fixed["device"] = "示宽灯"
            if any(term in first_part for term in ("光剑", "酷炫")):
                fixed["action"] = "调到"
                fixed["value"] = "光剑"
        if fixed.get("device") == "阅读灯" and fixed.get("position") == "主驾" and "主驾" not in first_part:
            fixed.pop("position", None)
        if fixed.get("value") == "智能" and fixed.get("action") != "关闭":
            fixed["action"] = "调到"
        if fixed.get("device") == "远光灯" and fixed.get("value") == "自动":
            fixed["value"] = "智能"
        if fixed.get("action") == "关闭":
            fixed.pop("value", None)
        if fixed.get("device") in {"大灯", "阅读灯"} and fixed.get("value") == "下一个":
            fixed.pop("value", None)
        if fixed.get("position") == "全部" and "全部" not in first_part and "所有" not in first_part:
            fixed.pop("position", None)
    elif tool == "ClimateControl":
        first_part = _first_command(query)
        if "节能模式" in first_part and "空调" in first_part:
            fixed.clear()
            fixed.update({"action": "打开", "device": "空调", "value": "节能模式"})
        if "小孩" in first_part and "空调别停" in first_part:
            fixed.clear()
            fixed.update({"action": "打开", "device": "空调", "value": "守护模式"})
        if "不想吹那么冷" in first_part:
            fixed.clear()
            fixed.update({"action": "打开", "device": "空调", "value": "自然风"})
        if "温度同步" in first_part or "左右两边温度一样" in first_part:
            fixed.clear()
            fixed.update({"action": "打开", "device": "空调", "value": "温度同步"})
        if "守护模式" in first_part:
            fixed.clear()
            fixed.update({"action": "打开", "device": "空调", "value": "守护模式"})
        if "极速降温" in first_part:
            fixed.clear()
            fixed.update({"action": "打开", "device": "空调", "value": "极速降温"})
        if "极速升温" in first_part:
            fixed.clear()
            fixed.update({"action": "打开", "device": "空调", "value": "极速升温"})
        if "避人吹" in first_part:
            fixed.clear()
            fixed.update({"action": "打开", "device": "空调", "value": "避人"})
        if "制冷" in first_part:
            fixed.clear()
            fixed.update({"action": "打开", "device": "空调", "value": "制冷"})
        if "制热" in first_part:
            fixed.clear()
            fixed.update({"action": "打开", "device": "空调", "value": "制热"})
        if "内循环" in first_part:
            fixed.clear()
            fixed.update({"action": "打开", "device": "空调", "value": "内循环"})
        if "外循环" in first_part:
            fixed.clear()
            fixed.update({"action": "打开", "device": "空调", "value": "外循环"})
        if "风量" in first_part:
            fixed["device"] = "空调"
            fixed["feature"] = "风"
            if "比较大" in first_part or "较大" in first_part:
                fixed["action"] = "调到"
                fixed["value"] = "较大"
            if "小一些" in first_part or "较小" in first_part:
                fixed["action"] = "调到"
                fixed["value"] = "较小"
        if "除雾" in first_part or "起雾" in first_part or "玻璃全是雾" in first_part:
            fixed["device"] = "后挡风" if "后" in first_part and "前" not in first_part else "前挡风"
            fixed["feature"] = "除雾"
            fixed.pop("value", None)
            fixed.pop("position", None)
        if "下大雨" in first_part and "看不清" in first_part:
            fixed.update({"action": "打开", "device": "前挡风", "feature": "除雾"})
            fixed.pop("value", None)
            fixed.pop("position", None)
        if fixed.get("value") in {"制冷", "制热", "自然风", "外循环", "内循环", "温度同步"}:
            fixed.setdefault("device", "空调")
            fixed["action"] = "关闭" if _query_mentions_close(first_part) else "打开"
            fixed.pop("feature", None)
        if fixed.get("value") in {"守护模式", "极速降温", "极速升温", "避人", "宠物模式"}:
            fixed.setdefault("device", "空调")
            if _query_mentions_close(first_part):
                fixed["action"] = "关闭"
        if "冻死" in first_part or "赶紧暖和" in first_part:
            fixed.update({"action": "打开", "device": "空调", "value": "极速升温"})
            fixed.pop("feature", None)
        elif "好冷" in first_part:
            fixed.update({"action": "调大", "device": "空调", "feature": "温度", "value": "一点"})
        if "太热" in first_part and "快速降温" in first_part:
            fixed.update({"action": "打开", "device": "空调", "value": "极速降温"})
            fixed.pop("feature", None)
        if "空气" in first_part and any(term in first_part for term in ("差", "不好", "味道")):
            fixed.clear()
            fixed.update({"action": "打开", "device": "空气净化器"})
        if "吹头" in first_part:
            fixed.clear()
            fixed.update({"action": "打开", "device": "出风口", "feature": "风", "position": "中间"})
        if "吹脚" in first_part:
            fixed.clear()
            fixed.update({"action": "打开", "device": "出风口", "feature": "风"})
        if fixed.get("feature") == "温度" and fixed.get("value") in {"最大", "最小"} and "风" in first_part:
            fixed["feature"] = "风"
        if "风关小" in first_part or "风调小" in first_part:
            fixed.update({"action": "调小", "device": "空调", "feature": "风", "value": "一点"})
        if "温度调到" in first_part:
            m = re.search(r"温度调到(\d{1,2})", first_part)
            if m:
                fixed.update({"action": "调到", "device": "空调", "feature": "温度", "value": m.group(1)})
        if fixed.get("device") == "空气净化器":
            fixed.pop("value", None)
    elif tool == "RearviewControl":
        first_part = _first_command(query)
        if fixed.get("device") == "外后视镜":
            fixed["device"] = "后视镜"
        if fixed.get("feature") == "加热":
            fixed.pop("feature", None)
        if fixed.get("action") == "打开" and "后视镜" in first_part and "展开" in first_part and not ("加热" in first_part):
            fixed["action"] = "展开"
        if fixed.get("action") in {"调大", "调小"} and _query_has_small_delta(first_part):
            fixed["value"] = "一点"
        if re.search(r"后视镜调[大小]一点", first_part):
            fixed.pop("feature", None)
        if any(term in first_part for term in ("调高", "往上", "抬一抬", "往上调")):
            fixed["action"] = "调大"
            fixed.setdefault("feature", "高度")
        if any(term in first_part for term in ("调低", "往下", "低一点", "低一点点")):
            fixed["action"] = "调小"
            fixed.setdefault("feature", "高度")
        if fixed.get("feature") == "高度" and fixed.get("value") == "一点" and not any(
            term in first_part for term in ("一点", "一点点", "一下", "一抬", "抬一抬", "往下调一下")
        ):
            fixed.pop("value", None)
        position = _infer_common_position(first_part)
        if position in {"主驾", "副驾", "左侧", "右侧"}:
            fixed["position"] = position
        if "右前方" in first_part:
            fixed["position"] = "副驾"
        if fixed.get("feature") == "高度" and fixed.get("action") in {"展开", "打开", "关闭"} and "高度" not in first_part:
            fixed.pop("feature", None)
    elif tool == "PowerControl":
        first_part = _first_command(query)
        if fixed.get("feature") == "能量回收" and fixed.get("action") in {"调大", "调小"} and _query_has_small_delta(first_part):
            fixed["value"] = "一点"
        if _query_mentions_close(first_part):
            fixed["action"] = "关闭"
        if fixed.get("feature") == "无线充电":
            position = _infer_common_position(first_part)
            if position in {"主驾", "副驾", "前排", "第二排", "第二排左侧", "第二排右侧"}:
                fixed["position"] = position
        if "单踏板" in first_part:
            fixed.update({"action": "调到", "feature": "能量回收", "value": "单踏板"})
    elif tool == "VoiceControl":
        first_part = _first_command(query)
        if _query_mentions_close(first_part) or "静音" in first_part:
            fixed["action"] = "关闭"
            fixed.pop("value", None)
            if any(term in first_part for term in ("声音", "静音", "别出声", "安静")):
                fixed["feature"] = "声音"
        if any(term in first_part for term in ("声音太小", "听不清", "太安静")):
            fixed["action"] = "调大"
            fixed["feature"] = "音量"
        if "调大" in first_part:
            fixed["action"] = "调大"
        if "调小" in first_part:
            fixed["action"] = "调小"
        if "调到" in first_part or "开到" in first_part:
            fixed["action"] = "调到"
        if fixed.get("feature") == "声音" and "声音" not in first_part and "静音" not in first_part and "别出声" not in first_part:
            fixed["feature"] = "音量"
        if "音量" in first_part and not any(term in first_part for term in ("导航音量", "语音音量", "通话音量")):
            fixed["feature"] = "音量"
        if any(term in first_part for term in ("导航播报", "导航声音")):
            fixed["feature"] = "导航音量"
        if fixed.get("feature") == "声音" and any(term in first_part for term in ("开到", "调到", "中等")):
            fixed["feature"] = "音量"
        if _query_has_small_delta(first_part) and "一点" in first_part:
            fixed["value"] = "一点"
        elif fixed.get("value") == "一点" and "一点" not in first_part:
            fixed.pop("value", None)
        percent_value = _infer_percent_value(first_part)
        if percent_value:
            fixed["value"] = percent_value
        if "一半" in first_part or "中等" in first_part:
            fixed["value"] = "中"
        if "最大" in first_part:
            fixed["value"] = "最大"
            fixed["action"] = "调到"
        if "最小" in first_part:
            fixed["value"] = "最小"
            fixed["action"] = "调到"
        if re.search(r"调到[一二三四五六七八九十]\\b|调到[0-9]+\\b", first_part):
            cn_digits = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"}
            m = re.search(r"调到([一二三四五六七八九十]|[0-9]+)", first_part)
            if m:
                fixed["value"] = cn_digits.get(m.group(1), m.group(1))
        if "全部" in first_part or "全车" in first_part:
            fixed["position"] = "全部"
        position = _infer_common_position(first_part)
        if position in {"副驾", "第二排", "第三排"} and fixed.get("feature") == "音量":
            fixed["position"] = position
        if "后面" in first_part and fixed.get("feature") == "音量":
            fixed["position"] = "第二排"
        if "拉满" in first_part:
            fixed["action"] = "调到"
            fixed["value"] = "最大"
        if "导航声音不用那么大" in first_part:
            fixed.update({"action": "调小", "feature": "导航音量"})
        if fixed.get("feature") in {"导航音量", "语音音量", "通话音量"}:
            fixed.pop("position", None)
        if isinstance(fixed.get("feature"), str) and "," in fixed["feature"]:
            fixed["feature"] = fixed["feature"].split(",", 1)[0].strip()
    elif tool == "ScreenControl":
        first_part = _first_command(query)
        screen_modes = ("清洁模式", "白天模式", "等人模式", "自动模式", "黑夜模式")
        mode = next((item for item in screen_modes if item in first_part), None)
        if mode:
            fixed.update({"action": "调到", "device": "屏幕", "value": mode})
            fixed.pop("feature", None)
        if "亮度" in first_part or "暗" in first_part or "亮" in first_part or "刺眼" in first_part:
            fixed["feature"] = "亮度"
        if "亮起来" in first_part:
            fixed["action"] = "打开"
            fixed.pop("feature", None)
        if "关掉" in first_part and "屏幕" in first_part:
            fixed["action"] = "关闭"
            fixed.pop("feature", None)
        if "调小" in first_part:
            fixed["action"] = "调小"
        if "调大" in first_part or "调高" in first_part:
            fixed["action"] = "调大"
        if "暗" in first_part or "模糊" in first_part:
            fixed["action"] = "调大"
        if "太亮" in first_part or "刺眼" in first_part:
            fixed["action"] = "调小"
        if _query_has_small_delta(first_part) and "一点" in first_part:
            fixed["value"] = "一点"
        elif fixed.get("value") == "一点" and "一点" not in first_part:
            fixed.pop("value", None)
        percent_value = _infer_percent_value(first_part)
        if percent_value:
            fixed["value"] = percent_value
        if "最大" in first_part:
            fixed.update({"action": "调到", "value": "最大"})
        if "最小" in first_part:
            fixed.update({"action": "调到", "value": "最小"})
        if "然后" in first_part and fixed.get("value") == "一点":
            fixed.pop("value", None)
        if "清洁" in query or "擦" in query:
            fixed.update({"action": "调到", "device": "屏幕", "value": "清洁模式"})
            fixed.pop("feature", None)
        position = _infer_common_position(first_part)
        if position:
            fixed["position"] = position
        if "放下来" in query or "弄出来" in query or "展开" in first_part:
            fixed["action"] = "展开"
            if "动画片" in query:
                fixed["device"] = "娱乐屏"
        if "收" in first_part:
            fixed["action"] = "收起"
        if fixed.get("value") == "打开":
            fixed.pop("value", None)
        if fixed.get("action") == "打开" and fixed.get("value") in {"最大", "最小"}:
            fixed.pop("value", None)
        if fixed.get("action") == "展开":
            fixed.pop("feature", None)
            if _infer_percent_value(first_part):
                fixed.pop("value", None)
    elif tool == "LockControl":
        first_part = _first_command(query)
        if "儿童锁" not in first_part and "车窗锁" not in first_part and "窗户锁" not in first_part:
            first_part = query
        if any(term in first_part for term in ("车窗锁", "窗户锁", "乱按车窗", "窗户不要让孩子碰", "窗户锁上", "自己控制窗户")):
            fixed["device"] = "车窗锁"
        else:
            fixed["device"] = "儿童锁"
        if _query_mentions_close(first_part) or any(term in first_part for term in ("解除", "解锁", "可以自己控制")):
            fixed["action"] = "关闭"
        else:
            fixed["action"] = "打开"
        position = _infer_common_position(first_part)
        if position:
            fixed["position"] = position
        if "第二排" in first_part and not any(term in first_part for term in ("第二排左", "第二排右", "第二排左侧", "第二排右侧")):
            fixed["position"] = "第二排"
        if "后排" in first_part and not any(term in first_part for term in ("后排左", "后排右", "左边", "右边", "左侧", "右侧")):
            fixed["position"] = "第二排"
        if not any(term in first_part for term in ("左", "右", "第二排", "第三排", "后排")) and fixed.get("position") == "第二排":
            fixed.pop("position", None)
        if fixed.get("device") == "车窗锁":
            fixed.pop("position", None)
        if "行驶途中" in first_part and "后排乘客" in first_part:
            fixed.pop("position", None)
        fixed.pop("third_position", None)
    elif tool == "GateControl":
        first_part = _first_command(query)
        if _query_mentions_close(first_part):
            fixed["action"] = "关闭"
        if _query_mentions_open(first_part) and _query_mentions_close(first_part) and first_part.find("打开") < first_part.find("关"):
            fixed["action"] = "打开"
        if "暂停" in first_part or "先停" in first_part or "先别关" in first_part or "别升" in first_part:
            fixed["action"] = "暂停"
        if fixed.get("device") == "车门" or ("门" in first_part and "鹏翼门" not in first_part and "充电口" not in first_part):
            fixed["device"] = "侧滑门"
        if "慢充" in first_part:
            fixed["device"] = "交流充电口"
        if "快充" in first_part:
            fixed["device"] = "直流充电口"
        if "鹏翼门" in first_part and ("都" in first_part or "全部" in first_part):
            fixed["position"] = "全部"
        if fixed.get("device") == "侧滑门" and "后排乘客" in first_part:
            fixed.pop("position", None)
        if fixed.get("device") == "侧滑门" and "左右两边" in first_part:
            fixed["position"] = "左侧"
    elif tool == "InfoQuery":
        first_part = _first_command(query)
        if any(term in first_part for term in ("哪条路", "迷路", "什么位置", "我在什么位置")):
            fixed["feature"] = "当前位置"
            fixed.pop("constraint", None)
        if any(term in first_part for term in ("堵不堵", "路况")) and not any(term in first_part for term in ("还有多远", "多少公里")):
            fixed["feature"] = "路况信息"
            fixed.pop("constraint", None)
        if "还有多远" in first_part or "多少公里" in first_part:
            fixed["feature"] = "剩余距离"
            fixed.pop("constraint", None)
    elif tool == "MediaControl":
        first_part = _first_command(query)
        if any(term in first_part for term in ("音乐", "歌", "下一首", "上一首", "换一首")):
            fixed["media_category"] = "歌"
        if "收音机" in first_part or "FM" in first_part:
            fixed["media_category"] = "FM"
        if "YouTube" in first_part or "youtube" in first_part:
            fixed.pop("media_category", None)
            fixed["app_name"] = "youtube"
        if "TuneIn" in first_part or "tunein" in first_part:
            fixed["app_name"] = "tunein"
            fixed["media_category"] = "TuneIn"
        if "Spotify" in first_part or "spotify" in first_part:
            fixed["app_name"] = "spotify"
            fixed["media_category"] = "歌"
        if "别放了" in first_part:
            fixed["media_control_action"] = "暂停"
        if "关闭" in first_part or "关掉" in first_part:
            fixed["media_control_action"] = "关闭"
        if "继续播放" in first_part:
            fixed["media_control_action"] = "继续"
            if "有声书" in first_part:
                fixed["media_category"] = "读物"
        if "再放一遍" in first_part:
            fixed["media_category"] = "歌"
            fixed["media_control_action"] = "上一个"
    elif tool == "MediaPlay":
        first_part = _first_command(query)
        if "TuneIn" in first_part or "tunein" in first_part:
            fixed["app_name"] = "tunein"
            fixed["media_category"] = "FM"
        if "广播" in first_part or "新闻" in first_part:
            fixed["media_category"] = "FM"
        if "手机" in first_part and "音乐" in first_part:
            fixed["media_source"] = "蓝牙"
            fixed.pop("media_category", None)
    elif tool == "FmSearchPlay":
        first_part = _first_command(query)
        if "AM1638" in first_part or "am1638" in first_part:
            fixed["fm_channel"] = "1638"
            fixed["media_category"] = "1638 am"
    elif tool == "XmlySearchPlay":
        first_part = _first_command(query)
        if "郭德纲相声" in first_part:
            fixed["podcast_name"] = "郭德纲相声"
    return fixed
