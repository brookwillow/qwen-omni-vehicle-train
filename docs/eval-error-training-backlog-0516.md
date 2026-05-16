# Eval Error Training Backlog

## P0 over_noise (59)
- Focus: 类型边界与拒识保守性
- Primary GT tool in examples: ClimateControl
- Recommendation: 补充短指令和 ASR 口语化动作样本，避免把低信息但明确的车控指令判成 Noise。
- Examples:
  - 太冷了太冷了 | gt=ClimateControl {'value': '4', 'action': '调高', 'feature': '温度', 'device': '空调'} | pred=NoiseDoNotAct {}
  - 窗户关了小P | gt=WindowControl {'action': '关闭', 'device': '车窗'} | pred=NoiseDoNotAct {}
  - 小P太冷了 | gt=ClimateControl {'value': '4', 'action': '调高', 'feature': '温度', 'device': '空调'} | pred=NoiseDoNotAct {}

## P0 wrong_arg_value:action (42)
- Focus: 参数值强对比
- Primary GT tool in examples: PullOverControl
- Recommendation: 为同一 query 槽位补充打开/关闭/调到/调高/调低/再开/再关/开到/关到的强对比样本。
- Examples:
  - 感觉有点晕车，请现在找个方便的地方靠边停一下，停车的时候尽量稳一点。 | gt=PullOverControl {'action': '立即执行停车'} | pred=PullOverControl {'action': '过路口停车'}
  - 麻烦先别急着并线出去，靠边停一下，我确认一下后座的文件袋还在不在，这地方视野不好，别急着走。 | gt=PullOverControl {'action': '立即执行停车'} | pred=PullOverControl {'action': '过路口停车'}
  - 升起车窗小P | gt=WindowControl {'action': '再关', 'device': '车窗'} | pred=WindowControl {'action': '打开', 'device': '车窗'}

## P0 wrong_arg_value:value (31)
- Focus: 参数值强对比
- Primary GT tool in examples: RearviewControl
- Recommendation: 补充 schema 枚举值归一化样本，特别是最高/最低/较高/较低/其他/标准模式等。
- Examples:
  - 关闭倒车自动下翻后视镜 | gt=RearviewControl {'action': '关闭', 'device': '后视镜', 'value': '自动下翻'} | pred=RearviewControl {'action': '关闭', 'device': '后视镜', 'value': '倒车自动下翻'}
  - 语音音量调到十五 | gt=VoiceControl {'action': '调到', 'feature': '语音音量', 'value': '中'} | pred=VoiceControl {'action': '调到', 'feature': '语音音量', 'value': '15'}
  - 空调风量调到一小P | gt=ClimateControl {'value': '1', 'action': '调到', 'feature': '风', 'device': '空调'} | pred=ClimateControl {'action': '调到', 'device': '空调', 'feature': '风', 'value': '小'}

## P1 missing_optional_args:device (26)
- Focus: 参数槽位边界
- Primary GT tool in examples: ProfileControl
- Recommendation: 补充槽位有无边界样本，让模型学习何时输出 position/value/device/feature，何时省略。
- Examples:
  - 主驾座椅放松模式小P | gt=ProfileControl {'action': '调到', 'device': '座椅', 'feature': '习惯', 'position': '主驾', 'value': '放松模式'} | pred=ProfileControl {'action': '调到', 'feature': '习惯', 'position': '主驾', 'value': '放松模式'}
  - 保存当前主驾座椅习惯 | gt=ProfileControl {'action': '保存', 'device': '座椅', 'feature': '习惯', 'position': '主驾'} | pred=ProfileControl {'action': '保存', 'feature': '习惯', 'position': '主驾'}
  - 小P主驾座椅调整到标准模式 | gt=ProfileControl {'action': '调到', 'device': '座椅', 'feature': '习惯', 'position': '主驾', 'value': '标准模式'} | pred=ProfileControl {'action': '调到', 'feature': '习惯', 'position': '主驾', 'value': '标准模式'}

## P1 extra_args:value (24)
- Focus: 参数槽位边界
- Primary GT tool in examples: ProfileControl
- Recommendation: 补充槽位有无边界样本，让模型学习何时输出 position/value/device/feature，何时省略。
- Examples:
  - 恢复成习惯一 | gt=ProfileControl {'action': '复位', 'feature': '习惯'} | pred=ProfileControl {'action': '复位', 'feature': '习惯', 'value': '上一个'}
  - 切换到主驾习惯开车 | gt=ProfileControl {'action': '调到', 'feature': '习惯', 'position': '主驾'} | pred=ProfileControl {'action': '调到', 'feature': '习惯', 'position': '主驾', 'value': '标准模式'}
  - 开启用车习惯一 | gt=ProfileControl {'action': '打开', 'feature': '习惯'} | pred=ProfileControl {'action': '打开', 'feature': '习惯', 'value': '下一个'}

## P1 mixed_arg_error:extra=value;changed=action (21)
- Focus: 参数值强对比
- Primary GT tool in examples: WindowControl
- Recommendation: 按该错误族补充非评估原句的泛化 hard case。
- Examples:
  - 前排车窗打开三分之一 | gt=WindowControl {'position': '前排', 'action': '再开', 'device': '车窗'} | pred=WindowControl {'action': '开到', 'device': '车窗', 'position': '前排', 'value': '33%'}
  - 全车车窗开一点 | gt=WindowControl {'position': '全部', 'action': '再开', 'device': '车窗'} | pred=WindowControl {'action': '开到', 'device': '车窗', 'position': '全部', 'value': '10%'}
  - 用车习惯二小P | gt=ProfileControl {'action': '复位', 'feature': '习惯'} | pred=ProfileControl {'action': '调到', 'feature': '习惯', 'value': '小憩模式'}

## P1 wrong_arg_value:device (19)
- Focus: 参数值强对比
- Primary GT tool in examples: WindowControl
- Recommendation: 按该错误族补充非评估原句的泛化 hard case。
- Examples:
  - 关掉天窗 | gt=WindowControl {'action': '关闭', 'device': '遮阳帘'} | pred=WindowControl {'action': '关闭', 'device': '车窗'}
  - 开天窗 | gt=WindowControl {'action': '打开', 'device': '遮阳帘'} | pred=WindowControl {'action': '打开', 'device': '车窗'}
  - 打开天窗幕布 | gt=WindowControl {'action': '打开', 'device': '遮阳帘'} | pred=WindowControl {'action': '打开', 'device': '天幕'}

## P1 over_reject (18)
- Focus: 类型边界与拒识保守性
- Primary GT tool in examples: AppControl
- Recommendation: 补充工具范围内的自然表达样本，避免工具内请求被拒识。
- Examples:
  - 帮我导航一下 | gt=AppControl {'action': '打开', 'feature': '导航地图'} | pred=None None
  - 查一下我的日程表 | gt=AppControl {'action': '打开', 'feature': '日历'} | pred=None None
  - 右边车道有车吗，打开看看 | gt=CameraControl {'action': '打开', 'device': '摄像头', 'position': '右侧'} | pred=None None

## P1 mixed_arg_error:missing=device;extra=value (17)
- Focus: 参数值强对比
- Primary GT tool in examples: ProfileControl
- Recommendation: 按该错误族补充非评估原句的泛化 hard case。
- Examples:
  - 主驾座椅恢复到用车习惯一 | gt=ProfileControl {'action': '复位', 'device': '座椅', 'feature': '习惯', 'position': '主驾'} | pred=ProfileControl {'action': '复位', 'feature': '习惯', 'position': '主驾', 'value': '上一个'}
  - 座椅模式调到习惯二 | gt=ProfileControl {'action': '调到', 'device': '座椅', 'feature': '习惯'} | pred=ProfileControl {'action': '调到', 'feature': '习惯', 'value': '下一个'}
  - 副驾座椅切换到习惯二 | gt=ProfileControl {'action': '调到', 'device': '座椅', 'feature': '习惯', 'position': '副驾'} | pred=ProfileControl {'action': '调到', 'feature': '习惯', 'position': '副驾', 'value': '下一个'}

## P1 missing_optional_args:position (17)
- Focus: 参数槽位边界
- Primary GT tool in examples: ClimateControl
- Recommendation: 补充槽位有无边界样本，让模型学习何时输出 position/value/device/feature，何时省略。
- Examples:
  - 小P全车空调开到二十四度 | gt=ClimateControl {'position': '全部', 'value': '24', 'action': '调到', 'feature': '温度', 'device': '空调'} | pred=ClimateControl {'action': '调到', 'device': '空调', 'feature': '温度', 'value': '24'}
  - 车内温度调到十八度 | gt=ClimateControl {'position': '全部', 'value': '18', 'action': '调到', 'feature': '温度', 'device': '空调'} | pred=ClimateControl {'action': '调到', 'device': '空调', 'feature': '温度', 'value': '18'}
  - 车内温度调到二十二度 | gt=ClimateControl {'position': '全部', 'value': '22', 'action': '调到', 'feature': '温度', 'device': '空调'} | pred=ClimateControl {'action': '调到', 'device': '空调', 'feature': '温度', 'value': '22'}

## P1 missing_optional_args:value (13)
- Focus: 参数槽位边界
- Primary GT tool in examples: LightControl
- Recommendation: 补充槽位有无边界样本，让模型学习何时输出 position/value/device/feature，何时省略。
- Examples:
  - 关闭智能远光灯 | gt=LightControl {'action': '关闭', 'device': '远光灯', 'value': '智能'} | pred=LightControl {'action': '关闭', 'device': '远光灯'}
  - 打开车窗通风 | gt=WindowControl {'action': '打开', 'device': '车窗', 'value': '通风'} | pred=WindowControl {'action': '打开', 'device': '车窗'}
  - 空调温度调高三度 | gt=ClimateControl {'value': '3', 'action': '调高', 'feature': '温度', 'device': '空调'} | pred=ClimateControl {'action': '调高', 'device': '空调', 'feature': '温度'}

## P1 wrong_arg_value:position (12)
- Focus: 参数值强对比
- Primary GT tool in examples: WindowControl
- Recommendation: 按该错误族补充非评估原句的泛化 hard case。
- Examples:
  - 打开主副驾车窗玻璃 | gt=WindowControl {'position': '前排', 'action': '打开', 'device': '车窗'} | pred=WindowControl {'action': '打开', 'device': '车窗', 'position': '主驾'}
  - 打开主副驾窗户 | gt=WindowControl {'position': '前排', 'action': '打开', 'device': '车窗'} | pred=WindowControl {'action': '打开', 'device': '车窗', 'position': '全部'}
  - 打开左后方车窗 | gt=WindowControl {'position': '第二排左侧', 'action': '打开', 'device': '车窗'} | pred=WindowControl {'action': '打开', 'device': '车窗', 'position': '左侧'}

## P1 mixed_arg_error:missing=position;extra=value (10)
- Focus: 参数值强对比
- Primary GT tool in examples: WindowControl
- Recommendation: 按该错误族补充非评估原句的泛化 hard case。
- Examples:
  - 所有车窗开到一半 | gt=WindowControl {'position': '全部', 'action': '开到', 'device': '车窗'} | pred=WindowControl {'action': '开到', 'device': '车窗', 'value': '50%'}
  - 关闭室内空调 | gt=ClimateControl {'position': '全部', 'action': '关闭', 'device': '空调'} | pred=ClimateControl {'action': '关闭', 'device': '空调', 'value': '制冷'}
  - 关闭全部屏幕 | gt=ScreenControl {'action': '关闭', 'device': '屏幕', 'position': '全部'} | pred=ScreenControl {'action': '关闭', 'device': '屏幕', 'value': '全屏'}

## P1 missing_optional_args:feature (10)
- Focus: 参数槽位边界
- Primary GT tool in examples: ClimateControl
- Recommendation: 补充槽位有无边界样本，让模型学习何时输出 position/value/device/feature，何时省略。
- Examples:
  - 空调调至二十 | gt=ClimateControl {'value': '20', 'action': '调到', 'feature': '温度', 'device': '空调'} | pred=ClimateControl {'action': '调到', 'device': '空调', 'value': '20'}
  - 空调风对人吹 | gt=ClimateControl {'action': '打开', 'feature': '风', 'device': '空调', 'value': '对人'} | pred=ClimateControl {'action': '打开', 'device': '空调', 'value': '对人'}
  - 打开全车扫风小P | gt=ClimateControl {'action': '打开', 'feature': '风', 'device': '空调', 'value': '自动扫风'} | pred=ClimateControl {'action': '打开', 'device': '空调', 'value': '自动扫风'}

## P2 extra_args:position (9)
- Focus: 参数槽位边界
- Primary GT tool in examples: WindowControl
- Recommendation: 补充槽位有无边界样本，让模型学习何时输出 position/value/device/feature，何时省略。
- Examples:
  - 你好小P关车窗 | gt=WindowControl {'action': '关闭', 'device': '车窗'} | pred=WindowControl {'action': '关闭', 'device': '车窗', 'position': '主驾'}
  - 打开儿童锁，把后排阅读灯也开一下 | gt=LockControl {'action': '打开', 'device': '儿童锁'} | pred=LockControl {'action': '打开', 'device': '儿童锁', 'position': '第二排'}
  - 把车窗关上音量调小一点然后搜索老板的电话 | gt=WindowControl {'action': '关闭', 'device': '车窗'} | pred=WindowControl {'action': '关闭', 'device': '车窗', 'position': '主驾'}

## P2 over_action (9)
- Focus: 类型边界与拒识保守性
- Primary GT tool in examples: -
- Recommendation: 按该错误族补充非评估原句的泛化 hard case。
- Examples:
  - 帮我调一下空调温度 | gt=None {} | pred=ClimateControl {'action': '调到', 'device': '空调', 'feature': '温度', 'value': '中'}
  - 空调温度不太合适 | gt=None {} | pred=ClimateControl {'action': '调低', 'device': '空调', 'feature': '温度'}
  - 帮我调一下音量 | gt=None {} | pred=VoiceControl {'action': '调高', 'feature': '声音'}

## P2 tool_confusion:SeatControl->ClimateControl (6)
- Focus: 工具边界强对比
- Primary GT tool in examples: ClimateControl
- Recommendation: 补充相邻工具强对比样本，使用相似话术但不同工具标签。
- Examples:
  - 小P空调加热 | gt=ClimateControl {'action': '打开', 'device': '空调', 'value': '制热'} | pred=SeatControl {'action': '打开', 'device': '座椅', 'feature': '制热'}
  - 开点暖风 | gt=ClimateControl {'action': '打开', 'device': '空调', 'value': '制热'} | pred=SeatControl {'action': '打开', 'device': '座椅', 'feature': '制热'}
  - 暖气 | gt=ClimateControl {'action': '打开', 'device': '空调', 'value': '制热'} | pred=SeatControl {'action': '打开', 'device': '座椅', 'feature': '制热'}

## P2 mixed_arg_error:missing=value;changed=action (6)
- Focus: 参数值强对比
- Primary GT tool in examples: ClimateControl
- Recommendation: 按该错误族补充非评估原句的泛化 hard case。
- Examples:
  - 小P风不要对我吹 | gt=ClimateControl {'action': '关闭', 'feature': '风', 'device': '空调', 'value': '对人'} | pred=ClimateControl {'action': '调低', 'device': '空调', 'feature': '风'}
  - 怎么这么热啊 | gt=ClimateControl {'value': '4', 'action': '调低', 'feature': '温度', 'device': '空调'} | pred=ClimateControl {'action': '调高', 'device': '空调', 'feature': '温度'}
  - 把风量调到比较大 | gt=ClimateControl {'action': '调到', 'device': '空调', 'feature': '风', 'value': '较高'} | pred=ClimateControl {'action': '调高', 'device': '空调', 'feature': '风'}

## P2 wrong_arg_value:feature (6)
- Focus: 参数值强对比
- Primary GT tool in examples: SeatControl
- Recommendation: 按该错误族补充非评估原句的泛化 hard case。
- Examples:
  - 吹得我头疼把风关小点温度也别那么低了 | gt=ClimateControl {'action': '调低', 'device': '空调', 'feature': '风'} | pred=ClimateControl {'action': '调低', 'device': '空调', 'feature': '温度'}
  - 打开按摩和座椅通风 | gt=SeatControl {'action': '打开', 'device': '座椅', 'feature': '按摩'} | pred=SeatControl {'action': '打开', 'device': '座椅', 'feature': '通风'}
  - 好冷好累，座椅加热开起来，再来个按摩 | gt=SeatControl {'action': '打开', 'device': '座椅', 'feature': '制热'} | pred=SeatControl {'action': '打开', 'device': '座椅', 'feature': '按摩'}

## P2 tool_confusion:ClimateControl->LightControl (5)
- Focus: 工具边界强对比
- Primary GT tool in examples: LightControl
- Recommendation: 补充相邻工具强对比样本，使用相似话术但不同工具标签。
- Examples:
  - 关闭前雾灯 | gt=LightControl {'action': '关闭', 'device': '前雾灯'} | pred=ClimateControl {'action': '关闭', 'device': '前挡风', 'feature': '除雾'}
  - 天黑了看不清路 | gt=LightControl {'action': '打开', 'device': '大灯'} | pred=ClimateControl {'action': '打开', 'device': '前挡风', 'feature': '除雾'}
  - 雾好大看不清前面 | gt=LightControl {'action': '打开', 'device': '前雾灯'} | pred=ClimateControl {'action': '打开', 'device': '空调', 'feature': '除雾'}
