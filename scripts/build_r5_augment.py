#!/usr/bin/env python3
"""R5 data augmentation.

Targets from eval_report_0427 analysis:
  P0-A: SeatControl/WindowControl position field coverage (？→main driver default bug)
  P0-B: Over-Clarify suppression for SeatControl / ClimateControl / WindowControl /
         InfoQuery / ScreenControl (53 total, these tools account for ~40)
  P1-A: ClimateControl feature/value diversity (feature='?' gap)
  P1-B: LightControl device coverage (日行灯, 近光灯 missing; position variants)

Run:
    python scripts/build_r5_augment.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path


RNG = random.Random(2025)
ACTION_FILE = Path("data/splits/action.jsonl")


def make_action(user: str, tool: str, args: dict, answer: str) -> dict:
    args_json = json.dumps(args, ensure_ascii=False)
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": f"Action: {tool}\nAction Input: {args_json}"},
            {"role": "user", "content": 'Tool Result: {"status": "success"}'},
            {"role": "assistant", "content": f"Final Answer: {answer}"},
        ]
    }


def make_action_value(user: str, tool: str, args: dict, answer: str) -> dict:
    """Action that returns a value (for InfoQuery etc.)"""
    result_val = args.pop("_result", '{"status": "success"}')
    args_json = json.dumps(args, ensure_ascii=False)
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": f"Action: {tool}\nAction Input: {args_json}"},
            {"role": "user", "content": f"Tool Result: {result_val}"},
            {"role": "assistant", "content": f"Final Answer: {answer}"},
        ]
    }


# ── P0-A: SeatControl position coverage ──────────────────────────────────────
# Issue: model defaults to 主驾 when position is ambiguous → 22 errors of ?→主驾
# Need explicit training for 副驾/后排/全部 positions against clear queries.

def build_seat_position() -> list[dict]:
    samples = []

    seat_templates = [
        # (user_query, position, feature, value, action, answer_suffix)
        # 副驾 seat
        ("把副驾的座椅调前一点", "副驾", "前后", "前移", "调节", "已将副驾座椅向前调整。"),
        ("副驾座椅往后移一下", "副驾", "前后", "后移", "调节", "好的，已将副驾座椅向后移动。"),
        ("副驾的椅背往后倒", "副驾", "靠背", "后仰", "调节", "已将副驾靠背向后调整。"),
        ("副驾靠背立起来一点", "副驾", "靠背", "前倾", "调节", "好的，已将副驾靠背前倾。"),
        ("副驾座椅抬高点", "副驾", "高低", "升高", "调节", "已为您升高副驾座椅。"),
        ("副驾座椅加热打开", "副驾", "加热", "开启", "打开", "好的，已打开副驾座椅加热。"),
        ("给副驾开下座椅加热", "副驾", "加热", "开启", "打开", "副驾座椅加热已开启。"),
        ("副驾座椅通风打开", "副驾", "通风", "开启", "打开", "已为您打开副驾座椅通风。"),
        ("关掉副驾的座椅加热", "副驾", "加热", "关闭", "关闭", "已关闭副驾座椅加热。"),

        # 后排左侧
        ("后排左边的座椅加热打开", "第二排左侧", "加热", "开启", "打开", "已打开第二排左侧座椅加热。"),
        ("后座左边靠背往后倒", "第二排左侧", "靠背", "后仰", "调节", "好的，已将第二排左侧靠背向后调整。"),
        ("后排左侧椅背放平一点", "第二排左侧", "靠背", "后仰", "调节", "已将第二排左侧靠背放平。"),
        ("把左边后座的椅子向前移", "第二排左侧", "前后", "前移", "调节", "好的，已将第二排左侧座椅前移。"),
        ("后排左侧座椅通风开下", "第二排左侧", "通风", "开启", "打开", "已为您开启第二排左侧座椅通风。"),

        # 后排右侧
        ("后排右边座椅加热开一下", "第二排右侧", "加热", "开启", "打开", "已打开第二排右侧座椅加热。"),
        ("后右边的椅背拉直一点", "第二排右侧", "靠背", "前倾", "调节", "已将第二排右侧靠背前倾。"),
        ("右边后座往后挪一下", "第二排右侧", "前后", "后移", "调节", "已将第二排右侧座椅向后调整。"),
        ("后排右侧座椅通风关掉", "第二排右侧", "通风", "关闭", "关闭", "已关闭第二排右侧座椅通风。"),

        # 第二排（无左右）
        ("后排座椅加热开一下", "第二排", "加热", "开启", "打开", "已打开后排座椅加热。"),
        ("后座椅背往后倒一下", "第二排", "靠背", "后仰", "调节", "好的，已将后排靠背向后调整。"),
        ("后排通风开下", "第二排", "通风", "开启", "打开", "好的，后排座椅通风已开启。"),

        # 全部
        ("所有座椅加热都打开", "全部", "加热", "开启", "打开", "好的，已为全部座椅开启加热。"),
        ("全部座椅加热关掉", "全部", "加热", "关闭", "关闭", "已关闭全部座椅加热。"),
        ("全部座椅通风打开", "全部", "通风", "开启", "打开", "已为全部座椅开启通风。"),

        # 主驾（更多自然表达）
        ("我这边的座椅往前移一下", "主驾", "前后", "前移", "调节", "好的，已将主驾座椅向前调整。"),
        ("我的椅背往后倒倒", "主驾", "靠背", "后仰", "调节", "已将主驾靠背向后倾斜。"),
        ("主驾的座椅靠背直一点", "主驾", "靠背", "前倾", "调节", "好的，已将主驾靠背前倾。"),
        ("我这里冷，打开座椅加热", "主驾", "加热", "开启", "打开", "已为您开启主驾座椅加热。"),
        ("帮我打开座椅通风，有点热", "主驾", "通风", "开启", "打开", "好的，已开启主驾座椅通风。"),
        ("座椅降低一点", "主驾", "高低", "降低", "调节", "已为您降低主驾座椅高度。"),
        ("我的座椅升高一下", "主驾", "高低", "升高", "调节", "好的，已升高主驾座椅。"),
    ]

    for user, pos, feat, val, act, ans in seat_templates:
        args = {"action": act, "device": "座椅", "position": pos,
                "feature": feat, "value": val}
        samples.append(make_action(user, "SeatControl", args, ans))

    return samples


# ── P0-A: WindowControl position coverage ────────────────────────────────────
# Issue: model adds 全部 or 第二排 when query doesn't specify → ?→全部 (11), ?→第二排 (2)
# Also 主驾→? misses. Need explicit training for unspecified = 主驾 (driver default).

def build_window_position() -> list[dict]:
    samples = []

    win_templates = [
        # 副驾
        ("副驾那边的车窗开一下", "副驾", "打开", "已打开副驾车窗。"),
        ("把副驾车窗关上", "副驾", "关闭", "好的，已关闭副驾车窗。"),
        ("副驾窗户开一半", "副驾", "半开", "好的，副驾车窗已半开。"),
        ("副驾位车窗降下来", "副驾", "打开", "已为您打开副驾车窗。"),

        # 后排
        ("后排车窗打开透气", "第二排", "打开", "已打开后排车窗。"),
        ("后面的窗户关上", "第二排", "关闭", "好的，已关闭后排车窗。"),
        ("后座窗户开一点", "第二排", "半开", "好的，后排车窗已半开。"),

        # 后排左侧
        ("后排左边的窗户开下", "第二排左侧", "打开", "已打开第二排左侧车窗。"),
        ("后左边窗关上", "第二排左侧", "关闭", "好的，已关闭第二排左侧车窗。"),

        # 后排右侧
        ("后排右边窗户打开", "第二排右侧", "打开", "已打开第二排右侧车窗。"),
        ("后右边的车窗关一下", "第二排右侧", "关闭", "已关闭第二排右侧车窗。"),

        # 全部
        ("所有车窗都关上", "全部", "关闭", "好的，已关闭全部车窗。"),
        ("车窗全部打开透气", "全部", "打开", "已打开全部车窗。"),
        ("所有窗户升上去", "全部", "关闭", "好的，全部车窗已关闭。"),
        ("全部车窗降下来", "全部", "打开", "好的，全部车窗已打开。"),

        # 主驾（driver-side = 主驾, no 副驾 mentioned）
        ("窗户开一点点", "主驾", "半开", "好的，已将主驾车窗半开。"),
        ("我这边的窗关上", "主驾", "关闭", "已关闭主驾车窗。"),
        ("把我这窗户打开透下气", "主驾", "打开", "好的，已打开主驾车窗。"),
        ("我的窗户有点进风，关下", "主驾", "关闭", "已为您关闭主驾车窗。"),
        ("主驾窗户开一半", "主驾", "半开", "好的，主驾车窗已半开。"),

        # 前排
        ("前排车窗都打开", "前排", "打开", "已打开前排车窗。"),
        ("前排窗户关上", "前排", "关闭", "好的，已关闭前排车窗。"),
    ]

    for user, pos, act, ans in win_templates:
        args = {"action": act, "device": "车窗", "position": pos}
        samples.append(make_action(user, "WindowControl", args, ans))

    return samples


# ── P0-B: Over-Clarify suppression ───────────────────────────────────────────
# 53 over-clarify errors; major tools: SeatControl(18), ClimateControl(7),
# WindowControl(7), InfoQuery(5), ScreenControl(5).
# Add clear Action samples that the model currently over-clarifies.

def build_anti_over_clarify() -> list[dict]:
    samples = []

    # --- SeatControl: "调高/调低/往前/靠背" without explicit position → 主驾 default ---
    seat_oc = [
        ("座椅往前移一下", {"action":"调节","device":"座椅","position":"主驾","feature":"前后","value":"前移"}, "好的，已将主驾座椅向前调整。"),
        ("把椅子往前推一点", {"action":"调节","device":"座椅","position":"主驾","feature":"前后","value":"前移"}, "已将主驾座椅前移。"),
        ("座位往后退一点", {"action":"调节","device":"座椅","position":"主驾","feature":"前后","value":"后移"}, "好的，主驾座椅已向后调整。"),
        ("座椅调高一点", {"action":"调节","device":"座椅","position":"主驾","feature":"高低","value":"升高"}, "已为您升高主驾座椅。"),
        ("椅子降低一点", {"action":"调节","device":"座椅","position":"主驾","feature":"高低","value":"降低"}, "已降低主驾座椅高度。"),
        ("靠背往后倒", {"action":"调节","device":"座椅","position":"主驾","feature":"靠背","value":"后仰"}, "好的，已将主驾靠背向后调整。"),
        ("靠背再直一点", {"action":"调节","device":"座椅","position":"主驾","feature":"靠背","value":"前倾"}, "已将主驾靠背前倾。"),
        ("帮我调整一下腰靠", {"action":"调节","device":"座椅","position":"主驾","feature":"腰托","value":"伸出"}, "好的，已调整主驾腰托。"),
        ("我的座椅太靠后了，往前移点", {"action":"调节","device":"座椅","position":"主驾","feature":"前后","value":"前移"}, "已将主驾座椅向前调整。"),
        ("这个坐着不舒服，把椅背调直", {"action":"调节","device":"座椅","position":"主驾","feature":"靠背","value":"前倾"}, "好的，已将主驾靠背调直。"),
        ("帮我开一下座椅加热", {"action":"打开","device":"座椅","position":"主驾","feature":"加热","value":"开启"}, "已为您开启主驾座椅加热。"),
        ("座椅有点热，关掉加热", {"action":"关闭","device":"座椅","position":"主驾","feature":"加热","value":"关闭"}, "已关闭主驾座椅加热。"),
        ("座椅通风打开", {"action":"打开","device":"座椅","position":"主驾","feature":"通风","value":"开启"}, "好的，已为您开启主驾座椅通风。"),
        ("我出汗了，开下通风吧", {"action":"打开","device":"座椅","position":"主驾","feature":"通风","value":"开启"}, "已为您开启主驾座椅通风。"),
        ("头枕向上调一下", {"action":"调节","device":"座椅","position":"主驾","feature":"头枕","value":"升高"}, "已为您升高主驾头枕。"),
        ("你帮我把坐垫延伸出来一点", {"action":"调节","device":"座椅","position":"主驾","feature":"坐垫","value":"延伸"}, "好的，已为您延伸主驾坐垫。"),
        ("按摩功能打开", {"action":"打开","device":"座椅","position":"主驾","feature":"按摩","value":"开启"}, "已为您开启主驾座椅按摩功能。"),
        ("把座椅按摩关掉", {"action":"关闭","device":"座椅","position":"主驾","feature":"按摩","value":"关闭"}, "好的，已关闭主驾座椅按摩。"),
    ]
    for user, args, ans in seat_oc:
        samples.append(make_action(user, "SeatControl", args, ans))

    # --- ClimateControl: vague but actionable queries ---
    climate_oc = [
        ("温度调高一点", {"action":"升温","device":"空调","feature":"温度","value":"升高"}, "好的，已为您升高空调温度。"),
        ("温度调低一点", {"action":"降温","device":"空调","feature":"温度","value":"降低"}, "已为您降低空调温度。"),
        ("有点热了", {"action":"降温","device":"空调","feature":"温度","value":"降低"}, "好的，已为您降低温度。"),
        ("好冷啊", {"action":"升温","device":"空调","feature":"温度","value":"升高"}, "已为您升高空调温度，稍后会暖和的。"),
        ("车里太闷了", {"action":"打开","device":"空调","feature":"外循环"}, "好的，已切换为外循环通风。"),
        ("这车里空气好差", {"action":"打开","device":"空调","feature":"外循环"}, "已为您打开外循环，改善车内空气。"),
        ("吹得我头疼把风关小点温度也别那么低了", {"action":"调节","device":"空调","feature":"风速","value":"降低"}, "好的，已为您降低风速。"),
        ("帮我开下空调", {"action":"打开","device":"空调"}, "好的，空调已打开。"),
        ("把空调关了", {"action":"关闭","device":"空调"}, "已关闭空调。"),
        ("风大了点，小一点", {"action":"调节","device":"空调","feature":"风速","value":"降低"}, "已为您降低风速。"),
    ]
    for user, args, ans in climate_oc:
        samples.append(make_action(user, "ClimateControl", args, ans))

    # --- WindowControl: vague but actionable ---
    window_oc = [
        ("车窗开一点", {"action":"半开","device":"车窗","position":"主驾"}, "好的，已将主驾车窗半开。"),
        ("把窗户关上", {"action":"关闭","device":"车窗","position":"主驾"}, "已关闭主驾车窗。"),
        ("窗户开着呢，关上", {"action":"关闭","device":"车窗","position":"主驾"}, "好的，已关闭主驾车窗。"),
        ("进风了，把窗关了", {"action":"关闭","device":"车窗","position":"主驾"}, "已为您关闭主驾车窗。"),
        ("能开下窗通通气吗", {"action":"半开","device":"车窗","position":"主驾"}, "好的，已将主驾车窗半开通气。"),
    ]
    for user, args, ans in window_oc:
        samples.append(make_action(user, "WindowControl", args, ans))

    # --- InfoQuery: direct questions that don't need clarification ---
    info_oc = [
        ("现在几点了", {"query_type":"时间"}, '{"time": "下午3点15分"}', "现在是下午3点15分。"),
        ("快到了吗，还有多少路要走", {"query_type":"导航剩余距离"}, '{"remaining": "约12公里，预计20分钟"}', "还有约12公里，预计20分钟到达。"),
        ("我开到哪儿了", {"query_type":"当前位置"}, '{"location": "北京市朝阳区建国路附近"}', "您当前在北京市朝阳区建国路附近。"),
        ("感觉车开起来有点软，看看是不是轮胎没气了", {"query_type":"轮胎气压"}, '{"tire_pressure": "四轮气压正常"}', "检测结果显示四轮气压正常，请您放心驾驶。"),
        ("刚才压了个坑，帮我检查下轮胎有没有问题", {"query_type":"轮胎状态"}, '{"tire_status": "轮胎状态正常"}', "轮胎自检结果正常，暂无异常。"),
        ("今天天气怎么样", {"query_type":"天气"}, '{"weather": "晴，22℃，微风"}', "今天晴天，气温22℃，微风。"),
        ("查一下导航还有多久", {"query_type":"导航时间"}, '{"eta": "约18分钟"}', "导航显示还需约18分钟到达。"),
        ("油够用吗", {"query_type":"燃油"}, '{"fuel": "剩余燃油约50%，可行驶约350公里"}', "当前燃油约50%，可行驶约350公里。"),
    ]
    for item in info_oc:
        if len(item) == 4:
            user, args, result, ans = item
            args = dict(args)
            args["_result"] = result
        else:
            user, args, ans = item
        samples.append(make_action_value(user, "InfoQuery", dict(args), ans))

    # --- ScreenControl ---
    screen_oc = [
        ("帮我导航一下", {"action":"打开","app":"导航"}, "好的，已为您打开导航。"),
        ("打开音乐", {"action":"打开","app":"音乐"}, "已为您打开音乐播放。"),
        ("打开地图", {"action":"打开","app":"地图"}, "好的，已打开地图。"),
        ("帮我开下K歌", {"action":"打开","app":"K歌"}, "已为您打开K歌应用。"),
        ("屏幕亮度调高点", {"action":"调节","feature":"亮度","value":"升高"}, "已为您调高屏幕亮度。"),
    ]
    for user, args, ans in screen_oc:
        samples.append(make_action(user, "ScreenControl", args, ans))

    return samples


# ── P1-A: ClimateControl feature/value diversity ─────────────────────────────
# Gap: feature='?' means feature field missing; value='?' means value missing too.
# Many samples are lacking explicit feature+value pairs.

def build_climate_feature_value() -> list[dict]:
    samples = []

    templates = [
        # 温度具体值
        ("把温度设到25度", {"action":"设置","device":"空调","feature":"温度","value":"25"}, "好的，已将温度设置为25℃。"),
        ("温度调到24度", {"action":"设置","device":"空调","feature":"温度","value":"24"}, "已将温度调整为24℃。"),
        ("空调温度设为26度", {"action":"设置","device":"空调","feature":"温度","value":"26"}, "好的，温度已设置为26℃。"),
        ("温度设到23度", {"action":"设置","device":"空调","feature":"温度","value":"23"}, "已将温度调整为23℃。"),
        ("调到22度吧", {"action":"设置","device":"空调","feature":"温度","value":"22"}, "好的，温度已设置为22℃。"),
        ("帮我把温度设到27度", {"action":"设置","device":"空调","feature":"温度","value":"27"}, "已将温度设置为27℃。"),
        ("温度设成20度", {"action":"设置","device":"空调","feature":"温度","value":"20"}, "好的，已将温度调整为20℃。"),

        # 风速
        ("风速调到最大", {"action":"调节","device":"空调","feature":"风速","value":"最大"}, "已将风速调至最大。"),
        ("风速调小一点", {"action":"调节","device":"空调","feature":"风速","value":"降低"}, "好的，已为您降低风速。"),
        ("风速调高点", {"action":"调节","device":"空调","feature":"风速","value":"升高"}, "已为您提高风速。"),
        ("风速关掉", {"action":"关闭","device":"空调","feature":"风速"}, "好的，已关闭风速（送风关闭）。"),
        ("自动风速", {"action":"设置","device":"空调","feature":"风速","value":"自动"}, "已切换为自动风速。"),

        # 出风模式
        ("吹脚", {"action":"设置","device":"空调","feature":"出风模式","value":"吹脚"}, "好的，已切换为吹脚模式。"),
        ("换成吹脸", {"action":"设置","device":"空调","feature":"出风模式","value":"吹脸"}, "已切换为吹脸出风。"),
        ("换成吹脸和吹脚", {"action":"设置","device":"空调","feature":"出风模式","value":"脸脚同吹"}, "已切换为脸脚同吹模式。"),

        # 循环模式
        ("切换到内循环", {"action":"切换","device":"空调","feature":"循环","value":"内循环"}, "已切换为内循环模式。"),
        ("切外循环", {"action":"切换","device":"空调","feature":"循环","value":"外循环"}, "好的，已切换为外循环。"),
        ("换成内循环", {"action":"切换","device":"空调","feature":"循环","value":"内循环"}, "好的，已切换为内循环。"),

        # AC / 制冷
        ("AC开一下", {"action":"打开","device":"空调","feature":"AC"}, "好的，AC已开启。"),
        ("关掉AC", {"action":"关闭","device":"空调","feature":"AC"}, "已关闭AC。"),
        ("开一下制冷", {"action":"打开","device":"空调","feature":"制冷"}, "好的，已开启制冷功能。"),

        # 除雾
        ("前挡风玻璃除雾开一下", {"action":"打开","device":"前挡风","feature":"除雾"}, "好的，已开启前挡风除雾。"),
        ("后窗除雾开下", {"action":"打开","device":"后挡风","feature":"除雾"}, "已开启后窗除雾。"),
        ("前除雾关掉", {"action":"关闭","device":"前挡风","feature":"除雾"}, "好的，已关闭前挡风除雾。"),
        ("帮我关下后窗除雾", {"action":"关闭","device":"后挡风","feature":"除雾"}, "已关闭后窗除雾。"),

        # 座椅加热/通风（ClimateControl也管）
        ("车里有点冷把座椅加热打开", {"action":"打开","device":"座椅","position":"主驾","feature":"加热"}, "好的，已为您开启主驾座椅加热。"),
        ("座椅太烫了，通风开下", {"action":"打开","device":"座椅","position":"主驾","feature":"通风"}, "已为您开启主驾座椅通风。"),

        # 智能模式
        ("开下极速降温", {"action":"打开","device":"空调","feature":"极速降温"}, "好的，已开启极速降温模式。"),
        ("极速降温关掉", {"action":"关闭","device":"空调","feature":"极速降温"}, "已关闭极速降温。"),
        ("开一下极速升温", {"action":"打开","device":"空调","feature":"极速升温"}, "好的，已开启极速升温模式。"),
        ("宠物模式开", {"action":"打开","device":"空调","feature":"宠物模式"}, "好的，已开启宠物模式。"),
        ("宠物模式关掉", {"action":"关闭","device":"空调","feature":"宠物模式"}, "已关闭宠物模式。"),
        ("帮我开下智能除味", {"action":"打开","device":"空调","feature":"智能除味"}, "好的，已开启智能除味功能。"),
        ("节能模式打开", {"action":"打开","device":"空调","feature":"节能模式"}, "已开启节能模式。"),
    ]

    for user, args, ans in templates:
        samples.append(make_action(user, "ClimateControl", args, ans))

    return samples


# ── P1-B: LightControl device/action coverage ────────────────────────────────
# Gap: 日行灯/近光灯 missing; position variants; value (亮度) missing

def build_light_device() -> list[dict]:
    samples = []

    templates = [
        # 日行灯
        ("日行灯打开", {"action":"打开","device":"日行灯"}, "好的，已打开日行灯。"),
        ("日行灯关闭", {"action":"关闭","device":"日行灯"}, "已关闭日行灯。"),
        ("开下日行灯", {"action":"打开","device":"日行灯"}, "日行灯已开启。"),

        # 近光灯
        ("开下近光灯", {"action":"打开","device":"近光灯"}, "好的，已打开近光灯。"),
        ("近光灯关掉", {"action":"关闭","device":"近光灯"}, "已关闭近光灯。"),
        ("切换一下近光和远光", {"action":"切换","device":"远光灯"}, "好的，已切换远近光灯。"),

        # 远光灯
        ("远光灯关掉", {"action":"关闭","device":"远光灯"}, "已关闭远光灯。"),
        ("开下远光灯", {"action":"打开","device":"远光灯"}, "好的，已打开远光灯。"),
        ("前面有车别用远光了", {"action":"关闭","device":"远光灯"}, "已为您关闭远光灯。"),

        # 氛围灯颜色/亮度
        ("氛围灯换个颜色", {"action":"切换","device":"氛围灯","value":"随机"}, "好的，已切换氛围灯颜色。"),
        ("氛围灯调蓝色", {"action":"调到","device":"氛围灯","value":"蓝色"}, "已将氛围灯调为蓝色。"),
        ("氛围灯调成红色", {"action":"调到","device":"氛围灯","value":"红色"}, "好的，氛围灯已调为红色。"),
        ("氛围灯亮度调高一点", {"action":"调到","device":"氛围灯","feature":"亮度","value":"升高"}, "已为您提高氛围灯亮度。"),
        ("氛围灯暗一点", {"action":"调到","device":"氛围灯","feature":"亮度","value":"降低"}, "好的，已降低氛围灯亮度。"),
        ("氛围灯关掉", {"action":"关闭","device":"氛围灯"}, "已关闭氛围灯。"),
        ("开下氛围灯", {"action":"打开","device":"氛围灯"}, "好的，氛围灯已打开。"),

        # 阅读灯 + position
        ("打开我这边的阅读灯", {"action":"打开","device":"阅读灯","position":"主驾"}, "已打开主驾阅读灯。"),
        ("副驾阅读灯打开", {"action":"打开","device":"阅读灯","position":"副驾"}, "好的，已打开副驾阅读灯。"),
        ("后排阅读灯开一下", {"action":"打开","device":"阅读灯","position":"第二排"}, "已打开后排阅读灯。"),
        ("关掉我这边的阅读灯", {"action":"关闭","device":"阅读灯","position":"主驾"}, "已关闭主驾阅读灯。"),
        ("副驾那边的阅读灯关掉", {"action":"关闭","device":"阅读灯","position":"副驾"}, "好的，已关闭副驾阅读灯。"),
        ("后排阅读灯全关", {"action":"关闭","device":"阅读灯","position":"第二排"}, "已关闭后排阅读灯。"),

        # 示宽灯
        ("开下示宽灯", {"action":"打开","device":"示宽灯"}, "好的，已打开示宽灯。"),
        ("示宽灯关掉", {"action":"关闭","device":"示宽灯"}, "已关闭示宽灯。"),

        # 雾灯
        ("前雾灯关掉", {"action":"关闭","device":"前雾灯"}, "已关闭前雾灯。"),
        ("后雾灯打开", {"action":"打开","device":"后雾灯"}, "好的，已打开后雾灯。"),
        ("后雾灯关掉", {"action":"关闭","device":"后雾灯"}, "已关闭后雾灯。"),
        ("前后雾灯都打开", {"action":"打开","device":"全部雾灯"}, "好的，前后雾灯已全部打开。"),

        # 大灯
        ("大灯自动", {"action":"设置","device":"大灯","value":"自动"}, "好的，大灯已切换为自动模式。"),
        ("大灯关掉", {"action":"关闭","device":"大灯"}, "已关闭大灯。"),
        ("开下大灯", {"action":"打开","device":"大灯"}, "好的，已打开大灯。"),
    ]

    for user, args, ans in templates:
        samples.append(make_action(user, "LightControl", args, ans))

    return samples


def main():
    seat_samples   = build_seat_position()
    window_samples = build_window_position()
    oc_samples     = build_anti_over_clarify()
    climate_samples = build_climate_feature_value()
    light_samples  = build_light_device()

    all_new = seat_samples + window_samples + oc_samples + climate_samples + light_samples

    # Shuffle new samples
    RNG.shuffle(all_new)

    # Append to action.jsonl
    existing = ACTION_FILE.read_text(encoding="utf-8")
    new_lines = "\n".join(json.dumps(s, ensure_ascii=False) for s in all_new)
    ACTION_FILE.write_text(existing.rstrip("\n") + "\n" + new_lines + "\n", encoding="utf-8")

    print(f"R5 augmentation complete:")
    print(f"  P0-A SeatControl position:    {len(seat_samples):3d} samples")
    print(f"  P0-A WindowControl position:  {len(window_samples):3d} samples")
    print(f"  P0-B Anti-Over-Clarify:       {len(oc_samples):3d} samples")
    print(f"  P1-A ClimateControl feat/val: {len(climate_samples):3d} samples")
    print(f"  P1-B LightControl device:     {len(light_samples):3d} samples")
    print(f"  Total new:                    {len(all_new):3d} samples")
    print(f"  action.jsonl: {len(existing.strip().splitlines())} → "
          f"{len(existing.strip().splitlines()) + len(all_new)} lines")


if __name__ == "__main__":
    main()
