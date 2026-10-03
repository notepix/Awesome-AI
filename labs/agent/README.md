# Agent：可观察、受限的工具控制器

本项目将模型输出、工具执行与停止规则拆开，默认验证控制器，扩展后验证真实本地模型。只读工具只有 `calculator` 和 `search`；不存在 shell、任意路径读写、网页或外部通信工具。先读 [公共环境](../README.md)。

## 离线运行

```powershell
python -m labs.run agent prepare
python -m labs.run agent index
python -m labs.run agent eval
python -m labs.run agent demo
```

`index` 建立自建校园记录的内存检索入口；`train` 是此阶段的别名，没有模型训练。离线 demo 只回放“计算 6×7”的 fixture；离线模式拒绝把任意提示伪装成已被 LLM 理解。

## 控制器接口

每轮提供器返回一个 JSON 对象，严格接受以下两类形状：

```json
{"tool":"calculator","arguments":{"expression":"(6+1)*6"}}
```

```json
{"final":"42"}
```

`search` 参数为必需的 `query` 与可选的 `k`，`k` 必须是 1–5 的整数。工具调用存在未知字段、未知工具或类型错误时返回错误观察，模型/fixture 可在剩余预算内纠正。默认控制器要求至少一次成功工具调用才接受 final，不能直接跳过工具作答。

计算器只通过 Python AST 解释数值、括号和 `+ - * /`，限制输入长度、节点数和数值范围；不调用 eval，不允许幂、函数、属性或导入。搜索仅访问启动时载入的固定文档集合，用户不能提供文件路径。

## 预算、超时和轨迹

配置包含最大步骤、工具次数、错误次数、总等待时间、单次调用等待时间和输出字符数。每一步保存动作、工具结果或错误；终止状态区分 completed、timeout 和各类预算耗尽。

调用超时表示**控制器停止等待**，不声称能强制终止第三方 GPU 内核。工具本身是短小且有界的只读操作；本地生成额外传入最大生成时长与新 token 上限。不要把本实现直接改为可执行任意代码的生产沙箱。

提供器收到的是单次调用实际预算。本地评估一旦超时，不再进入同一个模型；后续案例记为`not_run_after_timeout`、成功为false，并记录`provider_reusable=false`。这一规则防止前一生成线程未退出时又发起下一次生成，但不宣称后台任务已被强制终止。

fixture 测试涵盖正常计算、搜索、未知工具恢复、schema 恢复、除零恢复、工具预算和超时。fixture 最终答案由脚本提供，只证明控制器按预期转移；结果明确记录 `controller_fixture`。

## 真实本地 Qwen

自行准备本地模型并安装公共说明中的可选依赖：

```powershell
python -m labs.run agent eval --config labs/agent/local.json --model-path D:/models/Qwen2.5-0.5B-Instruct
python -m labs.run agent demo --config labs/agent/local.json --model-path D:/models/Qwen2.5-0.5B-Instruct --prompt "When does the library open?"
```

真实评估由 Qwen 自行生成计算/搜索动作，成功要求最终答案匹配目标且轨迹中存在实际成功的工具调用。非法 JSON、错误工具、超时和预算耗尽均计入结果。小模型不保证稳定遵循工具格式；运行失败仍是有价值的教学证据，不能用 fixture 成绩替代。

本评估约定计算题给出数值结果，时间题给出一个明确的HH:MM时间。检查器解析数值/时间并要求与目标一致，不用子串命中判成功；因此`142`、`08:00 and 18:00`不会通过相应目标。这是固定格式教学任务的评测，不是自由文本语义理解能力的完整评估。

提供器将工具观察标为数据；允许列表与参数校验由控制器强制执行。提示词本身不是安全边界。

实现位于 [`agent.py`](../ai_learning/agent.py)，结果 JSON 保存完整轨迹。公共测试额外验证导入/属性/幂等表达式被拒绝，未知路径字段与布尔型 k 不能绕过 schema。

## 回到讲义与精读

[检索与 Agent 讲义](../../docs/08-walkthroughs/06-retrieval-agents.md) · [ReAct 论文精读](../../readings/papers/react.md) · [LangGraph 工程精读](../../readings/projects/langgraph.md)。
