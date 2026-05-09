from tool_postprocess import postprocess_action_args, postprocess_action_call


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
        "feature": "位置",
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

    assert fixed == {
        "action": "调到",
        "device": "座椅",
        "feature": "位置",
        "position": "主驾",
        "value": "最后",
    }


def test_keeps_non_seat_control_args_unchanged():
    args = {
        "action": "打开",
        "device": "车窗",
        "position": "主驾",
    }

    fixed = postprocess_action_args("打开车窗", "WindowControl", args)

    assert fixed == {
        "action": "打开",
        "device": "车窗",
        "position": "主驾",
    }


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


def test_normalizes_light_color_from_query():
    args = {
        "action": "调到",
        "device": "氛围灯",
        "feature": "颜色",
        "value": "红",
    }

    fixed = postprocess_action_args("氛围灯调到红色", "LightControl", args)

    assert fixed == {
        "action": "调到",
        "device": "氛围灯",
        "feature": "颜色",
        "value": "红色",
    }


def test_normalizes_climate_defog_feature():
    args = {
        "action": "打开",
        "device": "前挡风",
    }

    fixed = postprocess_action_args("打开前挡风除雾", "ClimateControl", args)

    assert fixed == {
        "action": "打开",
        "device": "前挡风",
        "feature": "除雾",
    }


def test_adds_rearview_small_delta_value():
    args = {
        "action": "调大",
        "device": "后视镜",
        "feature": "高度",
    }

    fixed = postprocess_action_args("后视镜高度调大一点", "RearviewControl", args)

    assert fixed == {
        "action": "调大",
        "device": "后视镜",
        "feature": "高度",
        "value": "一点",
    }


def test_infers_power_wireless_charging_position():
    args = {
        "action": "打开",
        "feature": "无线充电",
    }

    fixed = postprocess_action_args("副驾那边的手机需要充一下电", "PowerControl", args)

    assert fixed == {
        "action": "打开",
        "feature": "无线充电",
        "position": "副驾",
    }


def test_closes_lock_when_query_says_release_child_lock():
    args = {
        "action": "打开",
        "device": "儿童锁",
    }

    fixed = postprocess_action_args("小孩不在车上了，解除儿童锁吧", "LockControl", args)

    assert fixed == {
        "action": "关闭",
        "device": "儿童锁",
    }


def test_removes_voice_small_delta_when_query_is_generic():
    args = {
        "action": "调大",
        "feature": "音量",
        "value": "一点",
    }

    fixed = postprocess_action_args("声音太小了，听不清", "VoiceControl", args)

    assert fixed == {
        "action": "调大",
        "feature": "音量",
    }


def test_normalizes_screen_cleaning_mode():
    args = {
        "action": "打开",
        "device": "屏幕",
    }

    fixed = postprocess_action_args("屏幕上有灰，我想擦一下", "ScreenControl", args)

    assert fixed == {
        "action": "调到",
        "device": "屏幕",
        "value": "清洁模式",
    }


def test_maps_air_quality_to_air_purifier():
    args = {
        "action": "打开",
        "device": "空调",
        "value": "外循环",
    }

    fixed = postprocess_action_args("这车里空气好差", "ClimateControl", args)

    assert fixed == {
        "action": "打开",
        "device": "空气净化器",
    }


def test_normalizes_medium_voice_volume():
    args = {
        "action": "调到",
        "feature": "声音",
        "value": "一点",
    }

    fixed = postprocess_action_args("音量帮我搞到一半就好了", "VoiceControl", args)

    assert fixed == {
        "action": "调到",
        "feature": "音量",
        "value": "中",
    }


def test_maps_child_animation_screen_to_rear_entertainment_screen():
    args = {
        "action": "打开",
        "device": "屏幕",
    }

    fixed = postprocess_action_args("孩子要在后排看动画片，把屏幕弄出来", "ScreenControl", args)

    assert fixed == {
        "action": "展开",
        "device": "娱乐屏",
        "position": "第二排",
    }


def test_maps_long_trip_relief_to_seat_massage_mode():
    args = {
        "action": "调前",
        "device": "座椅",
        "feature": "位置",
        "value": "最前",
    }

    fixed = postprocess_action_args("跑了一天好累，来个长途解乏", "SeatControl", args)

    assert fixed == {
        "action": "调到",
        "device": "座椅",
        "feature": "按摩",
        "value": "长途解乏",
    }


def test_normalizes_window_half_open_value():
    args = {
        "action": "打开",
        "device": "车窗",
        "position": "主驾",
        "value": "一半",
    }

    fixed = postprocess_action_args("主驾车窗开一半，副驾的全部打开", "WindowControl", args)

    assert fixed == {
        "action": "开到",
        "device": "车窗",
        "position": "主驾",
        "value": "50%",
    }


def test_maps_car_media_gallery_to_smart_image_app():
    fixed = postprocess_action_args(
        "我想看看车上拍的照片和视频",
        "AppControl",
        {"action": "打开", "feature": "应用列表"},
    )

    assert fixed == {
        "action": "打开",
        "feature": "智能影像",
    }


def test_infers_surrounding_camera_position():
    fixed = postprocess_action_args(
        "帮我看一下车周围的环境",
        "CameraControl",
        {"action": "打开", "device": "摄像头"},
    )

    assert fixed == {
        "action": "打开",
        "device": "摄像头",
        "position": "全部",
    }


def test_maps_camera_monitoring_to_sentry_mode():
    fixed = postprocess_action_args(
        "车停在外面不太安全，帮我监控一下",
        "CameraControl",
        {"action": "打开", "device": "摄像头"},
    )

    assert fixed == {
        "action": "打开",
        "device": "摄像头",
        "value": "哨兵模式",
    }


def test_maps_child_guard_climate_mode():
    fixed = postprocess_action_args(
        "小孩在车上我先去办个事空调别停",
        "ClimateControl",
        {"action": "打开", "device": "空调"},
    )

    assert fixed == {
        "action": "打开",
        "device": "空调",
        "value": "守护模式",
    }


def test_maps_wind_level_large_to_exact_value():
    fixed = postprocess_action_args(
        "把风量调到比较大",
        "ClimateControl",
        {"action": "调大", "device": "空调", "feature": "风", "value": "一点"},
    )

    assert fixed == {
        "action": "调到",
        "device": "空调",
        "feature": "风",
        "value": "较大",
    }


def test_maps_default_driving_state_to_standard_mode():
    fixed = postprocess_action_args(
        "恢复到默认的驾驶状态",
        "DrivingControl",
        {"action": "关闭", "feature": "驾驶模式"},
    )

    assert fixed == {
        "action": "打开",
        "feature": "驾驶模式",
        "value": "标准模式",
    }


def test_fills_empty_fridge_command():
    fixed = postprocess_action_args("把冰箱开了，顺便把车窗关上", "FridgeControl", {})

    assert fixed == {
        "action": "打开",
        "device": "冰箱",
    }


def test_maps_slow_charging_gate_device():
    fixed = postprocess_action_args(
        "用慢充的把那个充电盖子打开",
        "GateControl",
        {"action": "打开", "device": "充电口"},
    )

    assert fixed == {
        "action": "打开",
        "device": "交流充电口",
    }


def test_maps_lost_road_query_to_current_location():
    fixed = postprocess_action_args(
        "这是哪条路啊，我迷路了",
        "InfoQuery",
        {"feature": "路况信息"},
    )

    assert fixed == {
        "feature": "当前位置",
    }


def test_infers_all_camera_close_after_preface():
    fixed = postprocess_action_args(
        "不看了，把摄像头都关了吧",
        "CameraControl",
        {"action": "关闭", "device": "摄像头"},
    )

    assert fixed == {
        "action": "关闭",
        "device": "摄像头",
        "position": "全部",
    }


def test_prefers_first_camera_position_in_multi_position_command():
    fixed = postprocess_action_args(
        "打开前方摄像头和后方摄像头",
        "CameraControl",
        {"action": "打开", "device": "摄像头", "position": "全部"},
    )

    assert fixed == {
        "action": "打开",
        "device": "摄像头",
        "position": "前侧",
    }


def test_maps_dark_road_light_to_high_beam_without_delta():
    fixed = postprocess_action_args(
        "前面太暗了大灯不够亮",
        "LightControl",
        {"action": "调大", "device": "大灯", "value": "一点"},
    )

    assert fixed == {
        "action": "打开",
        "device": "远光灯",
    }


def test_maps_daylight_no_lights_to_headlight_close():
    fixed = postprocess_action_args(
        "天亮了不用开灯了",
        "LightControl",
        {"action": "关闭", "device": "阅读灯"},
    )

    assert fixed == {
        "action": "关闭",
        "device": "大灯",
    }


def test_maps_rear_sleep_light_to_reading_light():
    fixed = postprocess_action_args(
        "后排小朋友要睡觉了把那边灯灭了",
        "LightControl",
        {"action": "关闭", "device": "氛围灯", "position": "第二排"},
    )

    assert fixed == {
        "action": "关闭",
        "device": "阅读灯",
        "position": "第二排",
    }


def test_keeps_first_ambient_light_open_action_when_color_follows():
    fixed = postprocess_action_args(
        "打开氛围灯调成蓝色",
        "LightControl",
        {"action": "打开", "device": "氛围灯", "value": "下一个"},
    )

    assert fixed == {
        "action": "打开",
        "device": "氛围灯",
    }


def test_maps_car_controlled_lights_to_auto_headlight():
    fixed = postprocess_action_args(
        "出了隧道了把灯光交给车自己控制吧",
        "LightControl",
        {"action": "关闭", "device": "大灯"},
    )

    assert fixed == {
        "action": "调到",
        "device": "大灯",
        "value": "自动",
    }


def test_maps_rear_child_lock_position_to_second_row():
    fixed = postprocess_action_args(
        "两个娃都在后排，把后排都锁住",
        "LockControl",
        {"action": "打开", "device": "后门", "position": "全部"},
    )

    assert fixed == {
        "action": "打开",
        "device": "儿童锁",
        "position": "第二排",
    }


def test_keeps_headlight_as_first_light_device_when_closing_multiple_lights():
    fixed = postprocess_action_args(
        "关掉大灯和示宽灯",
        "LightControl",
        {"action": "关闭", "device": "大灯", "position": "全部"},
    )

    assert fixed == {
        "action": "关闭",
        "device": "大灯",
    }


def test_closes_high_beam_smart_mode_without_setting_mode():
    fixed = postprocess_action_args(
        "关闭远光灯智能模式",
        "LightControl",
        {"action": "关闭", "device": "远光灯", "value": "智能"},
    )

    assert fixed == {
        "action": "关闭",
        "device": "远光灯",
    }


def test_maps_all_interior_lights_to_all_reading_lights():
    fixed = postprocess_action_args(
        "关闭所有车内灯光包括阅读灯和氛围灯",
        "LightControl",
        {},
    )

    assert fixed == {
        "action": "关闭",
        "device": "阅读灯",
        "position": "全部",
    }


def test_prefers_front_fog_light_for_front_and_rear_fog_phrase():
    fixed = postprocess_action_args(
        "把前后雾灯都关了然后大灯调到自动模式",
        "LightControl",
        {},
    )

    assert fixed == {
        "action": "关闭",
        "device": "前雾灯",
    }


def test_keeps_first_reading_light_position_when_two_front_positions_mentioned():
    fixed = postprocess_action_args(
        "主驾和副驾的阅读灯都打开",
        "LightControl",
        {"action": "打开", "device": "阅读灯", "position": "主驾", "value": "下一个"},
    )

    assert fixed == {
        "action": "打开",
        "device": "阅读灯",
        "position": "主驾",
    }


def test_keeps_second_row_when_multiple_child_lock_rows_mentioned():
    fixed = postprocess_action_args(
        "关闭第二排和第三排的儿童锁",
        "LockControl",
        {"action": "关闭", "device": "儿童锁", "position": "第二排", "third_position": "第三排"},
    )

    assert fixed == {
        "action": "关闭",
        "device": "儿童锁",
        "position": "第二排",
    }


def test_preserves_second_row_left_child_lock_position():
    fixed = postprocess_action_args(
        "关闭第二排左侧儿童锁",
        "LockControl",
        {"action": "打开", "device": "儿童锁", "position": "第二排左侧"},
    )

    assert fixed == {
        "action": "关闭",
        "device": "儿童锁",
        "position": "第二排左侧",
    }


def test_removes_secondary_climate_mode_from_air_purifier():
    fixed = postprocess_action_args(
        "开空气净化器然后开内循环",
        "ClimateControl",
        {"action": "打开", "device": "空气净化器", "value": "内循环"},
    )

    assert fixed == {
        "action": "打开",
        "device": "空气净化器",
    }


def test_maps_warm_ambient_color_to_orange():
    fixed = postprocess_action_args(
        "我想要暖一点的氛围灯颜色",
        "LightControl",
        {"action": "调到", "device": "氛围灯", "feature": "颜色", "value": "下一个"},
    )

    assert fixed == {
        "action": "调到",
        "device": "氛围灯",
        "feature": "颜色",
        "value": "橙色",
    }


def test_removes_position_from_first_ambient_color_command():
    fixed = postprocess_action_args(
        "氛围灯换成红色然后第二排阅读灯打开",
        "LightControl",
        {"action": "调到", "device": "氛围灯", "value": "xx色", "position": "第二排"},
    )

    assert fixed == {
        "action": "调到",
        "device": "氛围灯",
        "feature": "颜色",
        "value": "红色",
    }


def test_adds_song_category_for_simple_media_control():
    fixed = postprocess_action_args(
        "暂停音乐",
        "MediaControl",
        {"media_control_action": "暂停"},
    )

    assert fixed == {
        "media_category": "歌",
        "media_control_action": "暂停",
    }


def test_maps_youtube_media_control_app_name():
    fixed = postprocess_action_args(
        "关闭YouTube视频",
        "MediaControl",
        {"media_category": "视频", "media_control_action": "关闭"},
    )

    assert fixed == {
        "app_name": "youtube",
        "media_control_action": "关闭",
    }


def test_maps_tunein_media_play_to_fm():
    fixed = postprocess_action_args(
        "用TuneIn播放电台",
        "MediaPlay",
        {"media_category": "歌"},
    )

    assert fixed == {
        "app_name": "tunein",
        "media_category": "FM",
    }


def test_maps_am_channel_media_category():
    fixed = postprocess_action_args(
        "播放AM1638",
        "FmSearchPlay",
        {"fm_channel": "1638", "media_category": "am"},
    )

    assert fixed == {
        "fm_channel": "1638",
        "media_category": "1638 am",
    }


def test_keeps_full_xmly_podcast_name():
    fixed = postprocess_action_args(
        "在喜马拉雅搜索郭德纲相声",
        "XmlySearchPlay",
        {"podcast_name": "郭德纲"},
    )

    assert fixed == {
        "podcast_name": "郭德纲相声",
    }


def test_remaps_camera_environment_from_app_control():
    tool, args = postprocess_action_call(
        "帮我看一下车周围的环境",
        "AppControl",
        {"action": "打开", "feature": "智能影像"},
    )

    assert tool == "CameraControl"
    assert args == {
        "action": "打开",
        "device": "摄像头",
        "position": "全部",
    }


def test_remaps_driving_energy_mode_from_climate_control():
    tool, args = postprocess_action_call(
        "关闭节能模式",
        "ClimateControl",
        {"action": "关闭", "device": "空调", "value": "节能模式"},
    )

    assert tool == "DrivingControl"
    assert args == {
        "action": "关闭",
        "feature": "驾驶模式",
        "value": "节能模式",
    }


def test_does_not_remap_explicit_climate_mode_to_driving():
    tool, args = postprocess_action_call(
        "空调切换到舒适模式",
        "ClimateControl",
        {"action": "调到", "device": "空调", "value": "舒适模式"},
    )

    assert tool == "ClimateControl"
    assert args == {
        "action": "调到",
        "device": "空调",
        "value": "舒适模式",
    }


def test_closes_climate_guard_mode():
    tool, args = postprocess_action_call(
        "关闭守护模式",
        "ClimateControl",
        {"action": "打开", "device": "空调", "value": "守护模式"},
    )

    assert tool == "ClimateControl"
    assert args == {
        "action": "关闭",
        "device": "空调",
        "value": "守护模式",
    }


def test_remaps_sentry_monitoring_from_full_query():
    tool, args = postprocess_action_call(
        "车停在外面不太安全，帮我监控一下",
        "AppControl",
        {"action": "打开", "feature": "智能影像"},
    )

    assert tool == "CameraControl"
    assert args == {
        "action": "打开",
        "device": "摄像头",
        "value": "哨兵模式",
    }


def test_remaps_steering_assist_from_voice_control():
    tool, args = postprocess_action_call(
        "方向盘助力切换到舒适模式",
        "VoiceControl",
        {"action": "打开", "feature": "导航音量"},
    )

    assert tool == "SteeringwheelControl"
    assert args == {
        "action": "调到",
        "device": "方向盘",
        "feature": "助力",
        "value": "舒适模式",
    }


def test_remaps_window_lock_from_window_control():
    tool, args = postprocess_action_call(
        "打开车窗锁",
        "WindowControl",
        {"action": "打开", "device": "车窗"},
    )

    assert tool == "LockControl"
    assert args == {
        "action": "打开",
        "device": "车窗锁",
    }


def test_remaps_phone_search_from_noise_action():
    tool, args = postprocess_action_call("帮我查一下18688886666这个号码", "NoiseAction", {})

    assert tool == "PhoneControl"
    assert args == {
        "action": "搜索",
        "telephone": "18688886666",
    }


def test_remaps_first_window_command_before_phone_search():
    tool, args = postprocess_action_call(
        "关上车窗然后帮我搜索李四的号码",
        "PhoneControl",
        {"action": "搜索", "person": "李四"},
    )

    assert tool == "WindowControl"
    assert args == {
        "action": "关闭",
        "device": "车窗",
    }


def test_fixes_steering_exact_mode_and_level_values():
    assert postprocess_action_call(
        "方向盘制热调到三档",
        "SteeringwheelControl",
        {"action": "调到", "device": "方向盘", "feature": "制热"},
    ) == (
        "SteeringwheelControl",
        {"action": "调到", "device": "方向盘", "feature": "制热", "value": "3"},
    )

    assert postprocess_action_call(
        "方向盘助力调到稳重模式",
        "SteeringwheelControl",
        {"action": "调到", "device": "方向盘", "feature": "助力"},
    ) == (
        "SteeringwheelControl",
        {"action": "调到", "device": "方向盘", "feature": "助力", "value": "稳重模式"},
    )


def test_fixes_screen_mode_and_brightness_direction():
    assert postprocess_action_call(
        "屏幕调到黑夜模式",
        "ScreenControl",
        {"action": "调小", "device": "屏幕", "feature": "亮度"},
    ) == (
        "ScreenControl",
        {"action": "调到", "device": "屏幕", "value": "黑夜模式"},
    )

    assert postprocess_action_call(
        "调小屏幕亮度",
        "ScreenControl",
        {"action": "调大", "device": "屏幕", "feature": "亮度"},
    ) == (
        "ScreenControl",
        {"action": "调小", "device": "屏幕", "feature": "亮度"},
    )


def test_fixes_window_delta_percent_and_precise_position():
    assert postprocess_action_call(
        "车窗再关一点",
        "WindowControl",
        {"action": "打开", "device": "车窗"},
    ) == (
        "WindowControl",
        {"action": "再关", "device": "车窗", "value": "一点"},
    )

    assert postprocess_action_call(
        "打开第二排右侧的车窗",
        "WindowControl",
        {"action": "打开", "device": "车窗", "position": "右侧"},
    ) == (
        "WindowControl",
        {"action": "打开", "device": "车窗", "position": "第二排右侧"},
    )


def test_preserves_model_window_position_when_query_is_unavailable():
    assert postprocess_action_call(
        "",
        "WindowControl",
        {"action": "打开", "device": "车窗", "position": "主驾"},
    ) == (
        "WindowControl",
        {"action": "打开", "device": "车窗", "position": "主驾"},
    )


def test_preserves_complete_model_window_call_when_query_is_less_specific():
    assert postprocess_action_call(
        "关闭车窗",
        "WindowControl",
        {"action": "关闭", "device": "车窗", "position": "主驾"},
    ) == (
        "WindowControl",
        {"action": "关闭", "device": "车窗", "position": "主驾"},
    )


def test_preserves_model_window_action_when_query_conflicts_with_complete_call():
    assert postprocess_action_call(
        "关闭主驾车窗",
        "WindowControl",
        {"action": "打开", "device": "车窗", "position": "主驾"},
    ) == (
        "WindowControl",
        {"action": "打开", "device": "车窗", "position": "主驾"},
    )


def test_fixes_lock_first_command_without_later_window_pollution():
    assert postprocess_action_call(
        "打开儿童锁，再把车窗也关上",
        "LockControl",
        {"action": "关闭", "device": "儿童锁"},
    ) == (
        "LockControl",
        {"action": "打开", "device": "儿童锁"},
    )

    assert postprocess_action_call(
        "帮我把后排左边的儿童锁开一下",
        "LockControl",
        {"action": "打开", "device": "儿童锁", "position": "第二排"},
    ) == (
        "LockControl",
        {"action": "打开", "device": "儿童锁", "position": "第二排左侧"},
    )


def test_fixes_voice_exact_value_and_sound_feature():
    assert postprocess_action_call(
        "音量调到百分之五十",
        "VoiceControl",
        {"action": "调到", "feature": "音量"},
    ) == (
        "VoiceControl",
        {"action": "调到", "feature": "音量", "value": "50%"},
    )

    assert postprocess_action_call(
        "静音",
        "VoiceControl",
        {"action": "关闭", "feature": "音量"},
    ) == (
        "VoiceControl",
        {"action": "关闭", "feature": "声音"},
    )


def test_fixes_app_media_and_info_boundaries():
    assert postprocess_action_call(
        "我想听歌",
        "MediaPlay",
        {"media_category": "歌"},
    ) == (
        "AppControl",
        {"action": "打开", "feature": "音乐应用"},
    )

    assert postprocess_action_call(
        "关闭音乐",
        "AppControl",
        {"action": "关闭", "feature": "音乐应用"},
    ) == (
        "MediaControl",
        {"media_category": "歌", "media_control_action": "关闭"},
    )

    assert postprocess_action_call(
        "现在路上车多不多，会不会很堵",
        "AppControl",
        {"action": "打开", "feature": "导航地图"},
    ) == (
        "InfoQuery",
        {"feature": "路况信息"},
    )


def test_fixes_0504_steering_heat_actions():
    assert postprocess_action_call(
        "方向盘太烫了，把加热关了",
        "SteeringwheelControl",
        {"action": "打开", "device": "方向盘", "feature": "制热"},
    ) == (
        "SteeringwheelControl",
        {"action": "关闭", "device": "方向盘", "feature": "制热"},
    )

    assert postprocess_action_call(
        "方向盘还是不够暖，再加大一点",
        "SteeringwheelControl",
        {"action": "打开", "device": "方向盘", "feature": "制热"},
    ) == (
        "SteeringwheelControl",
        {"action": "调大", "device": "方向盘", "feature": "制热", "value": "一点"},
    )


def test_fixes_0504_climate_wind_and_mode_boundaries():
    assert postprocess_action_call(
        "风太小了几乎感觉不到",
        "WindowControl",
        {"action": "打开", "device": "遮阳帘"},
    ) == (
        "ClimateControl",
        {"action": "调大", "device": "空调", "feature": "风", "value": "一点"},
    )

    assert postprocess_action_call(
        "开启节能模式",
        "DrivingControl",
        {"action": "打开", "feature": "驾驶模式", "value": "节能模式"},
    ) == (
        "ClimateControl",
        {"action": "打开", "device": "空调", "value": "节能模式"},
    )


def test_fixes_0504_media_boundaries():
    assert postprocess_action_call(
        "关闭音乐应用",
        "MediaControl",
        {"media_category": "歌", "media_control_action": "关闭"},
    ) == (
        "AppControl",
        {"action": "关闭", "feature": "音乐应用"},
    )

    assert postprocess_action_call(
        "播放FM广播",
        "FmSearchPlay",
        {"fm_channel": "100.7", "media_category": "FM"},
    ) == (
        "MediaPlay",
        {"media_category": "FM"},
    )

    assert postprocess_action_call(
        "别放了太吵了",
        "NoiseAction",
        {},
    ) == (
        "MediaControl",
        {"media_control_action": "暂停"},
    )


def test_fixes_0504_window_and_lock_args():
    assert postprocess_action_call(
        "暂停车窗",
        "WindowControl",
        {"action": "打开", "device": "车窗"},
    ) == (
        "WindowControl",
        {"action": "暂停", "device": "车窗"},
    )

    assert postprocess_action_call(
        "孩子们可以自己控制窗户了",
        "LockControl",
        {"action": "打开", "device": "车窗锁"},
    ) == (
        "LockControl",
        {"action": "关闭", "device": "车窗锁"},
    )


def test_fixes_0504_voice_and_fridge_args():
    assert postprocess_action_call(
        "导航播报的声音太小了，大一点",
        "VoiceControl",
        {"action": "调大", "feature": "音量"},
    ) == (
        "VoiceControl",
        {"action": "调大", "feature": "导航音量"},
    )

    assert postprocess_action_call(
        "打开第二排冰箱",
        "FridgeControl",
        {"action": "打开", "device": "冰箱"},
    ) == (
        "FridgeControl",
        {"action": "打开", "device": "冰箱", "position": "第二排"},
    )
