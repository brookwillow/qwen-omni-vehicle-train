from tool_postprocess import postprocess_action_args


def test_removes_implicit_driver_position_from_generic_seat_command():
    args = {
        "action": "调到",
        "device": "座椅",
        "position": "主驾",
        "value": "最前",
    }

    fixed = postprocess_action_args("座椅调到最前", "SeatControl", args)

    assert fixed == {
        "action": "调到",
        "device": "座椅",
        "value": "最前",
    }


def test_keeps_driver_position_when_query_mentions_driver_seat():
    args = {
        "action": "调到",
        "device": "座椅",
        "position": "主驾",
        "value": "最后",
    }

    fixed = postprocess_action_args("主驾座椅调到最后", "SeatControl", args)

    assert fixed == args


def test_keeps_non_seat_control_args_unchanged():
    args = {
        "action": "打开",
        "device": "车窗",
        "position": "主驾",
    }

    fixed = postprocess_action_args("打开车窗", "WindowControl", args)

    assert fixed == args


def test_adds_position_feature_for_seat_forward_backward_adjustment():
    args = {
        "action": "调前",
        "device": "座椅",
        "position": "副驾",
        "value": "一点",
    }

    fixed = postprocess_action_args("副驾座椅往前调一点", "SeatControl", args)

    assert fixed == {
        "action": "调前",
        "device": "座椅",
        "feature": "位置",
        "position": "副驾",
        "value": "一点",
    }


def test_normalizes_seat_switch_mode_to_set_action():
    args = {
        "action": "切换",
        "device": "座椅",
        "feature": "按摩",
        "position": "主驾",
        "value": "休闲放松",
    }

    fixed = postprocess_action_args("按摩换成休闲放松", "SeatControl", args)

    assert fixed == {
        "action": "调到",
        "device": "座椅",
        "feature": "按摩",
        "value": "休闲放松",
    }


def test_adds_percent_unit_for_window_numeric_percent_value():
    args = {
        "action": "开到",
        "device": "车窗",
        "value": "50",
    }

    fixed = postprocess_action_args("车窗开到50%", "WindowControl", args)

    assert fixed == {
        "action": "开到",
        "device": "车窗",
        "value": "50%",
    }


def test_infers_empty_window_args_from_first_window_command():
    fixed = postprocess_action_args("主驾车窗开一半，副驾的全部打开", "WindowControl", {})

    assert fixed == {
        "action": "开到",
        "device": "车窗",
        "position": "主驾",
        "value": "50%",
    }


def test_infers_all_windows_close_from_empty_window_args():
    fixed = postprocess_action_args("所有窗户都关上，车窗锁也打开", "WindowControl", {})

    assert fixed == {
        "action": "关闭",
        "device": "车窗",
        "position": "全部",
    }
