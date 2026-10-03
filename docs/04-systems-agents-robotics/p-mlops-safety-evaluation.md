# P. MLOps、安全与评测


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](o-llm-posttraining-rag-agents.md) · [下一章 →](q-embodied-ai-robotics.md)

---

## 配套系统讲解

[串联本章的推导、例子与练习解析](../08-walkthroughs/04-pretraining-inference.md) · [论文与源码精读](../../readings/README.md) · [完整实践代码](../../labs/README.md)

原有知识单元保留稳定编号；概念卡用于定位，系统讲解用于连接完整过程。

## 知识单元

<a id="p01"></a>

### P01 环境、随机性与可复现运行【稳定工程原则】
- **先修**：Python 环境、随机数、版本控制。
- **定义与解析**：可复现不是“我这里又跑通一次”，而是记录代码、数据、配置、依赖、硬件和随机源，使同一环境内结果落在预声明容差；跨版本/设备不保证逐位一致。
- **公式/机制**：一次 run 的身份可写为 $\operatorname{hash}(\text{code commit},\text{data version},\text{config},\text{environment})$；随机实验应报告多种子分布。
- **资料**：PyTorch [Reproducibility：randomness/deterministic algorithms/DataLoader](https://docs.pytorch.org/docs/stable/notes/randomness)；Python [`venv` 创建与激活](https://docs.python.org/3/library/venv.html)。
- **最小代码（可执行）**：
```python
import random, hashlib, json, numpy as np
def run(seed):
    random.seed(seed); g=np.random.default_rng(seed)
    return float(g.normal(size=1000).mean())
cfg={'lr':.01,'batch':32,'seeds':list(range(5))}
run_id=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:10]
vals=[run(s) for s in cfg['seeds']]
print(run_id,np.mean(vals),np.std(vals)); assert vals==[run(s) for s in cfg['seeds']]
```
- **检测/实验**：在同环境复跑五种子；升级一个依赖后执行回归测试并记录差异，不承诺跨平台 bitwise 相同。
- **常见坑**：只设 PyTorch seed、漏 NumPy/DataLoader；把最佳 seed 当均值；未保存数据与 tokenizer 版本。

<a id="p02"></a>

### P02 数据版本、血缘与数据契约【稳定工程原则】
- **先修**：P01、文件哈希、schema、数据切分。
- **定义与解析**：数据版本标识内容，血缘记录原始数据到特征/标签的转换，契约约束字段、类型、范围和隐私；模型文件可复现但数据不可追踪仍不算复现。
- **公式/机制**：content-addressed id 为 $h=\operatorname{SHA256}(\mathrm{bytes})$；contract 在训练前 fail closed，记录 train/val/test 的生成查询与时间边界。
- **资料**：DVC [Data pipelines：stage/dependency/output](https://dvc.org/doc/user-guide/pipelines/defining-pipelines)；Gebru et al. [Datasheets for Datasets，§3–4 questions](https://arxiv.org/abs/1803.09010)。
- **最小代码（可执行契约检查）**：
```python
import hashlib, json
rows=[{'age':20,'label':1},{'age':35,'label':0}]
def validate(r):
    assert set(r)=={'age','label'}
    assert isinstance(r['age'],int) and 0<=r['age']<=120
    assert r['label'] in (0,1)
for row in rows: validate(row)
version=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
print(version[:12])
```
- **检测/实验**：故意加入缺列、未来时间戳、重复主键与非法类别，确认流水线在训练前失败并给可定位错误。
- **常见坑**：文件名写 v2 代替内容版本；标签修订未升版本；日志包含个人信息或训练/测试主体重叠。

<a id="p03"></a>

### P03 实验追踪、基线与持续测试【稳定工程原则】
- **先修**：P01–P02、单元测试、统计指标。
- **定义与解析**：实验追踪绑定参数、代码、数据、指标和 artifact；CI 应验证数据/张量契约、过拟合小批次、保存加载等价和性能回归，而不是完整重训大模型。
- **公式/机制**：候选必须相对固定 baseline 按预设方向改进，并同时满足质量、延迟、安全护栏。
- **资料**：MLflow [Tracking runs/models/datasets](https://mlflow.org/docs/latest/ml/tracking/)；PyTorch [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)。
- **最小代码（可执行回归门）**：
```python
baseline={'acc':.82,'p95_ms':12.}; candidate={'acc':.84,'p95_ms':12.5}
gates={
    'quality':candidate['acc']>=baseline['acc']+.01,
    'latency':candidate['p95_ms']<=baseline['p95_ms']*1.10,
}
for name,ok in gates.items():
    print(name,ok)
    assert ok, f'gate failed: {name}'
```
- **检测/实验**：让质量提升但 p95 超限，确认 CI 阻断；区分 smoke test、确定性单测和昂贵统计回归。
- **常见坑**：run 名称手填导致冲突；只记录最终指标；对波动指标使用零容差或一次运行门禁。

<a id="p04"></a>

### P04 打包、推理接口与服务契约【稳定工程原则】

<!-- readings:start -->
**进一步精读：** [PagedAttention：KV缓存分页与共享](../../readings/papers/pagedattention.md) · [vLLM：缓存调度与模型服务](../../readings/projects/vllm.md)
<!-- readings:end -->
- **先修**：P02–P03、HTTP/JSON、序列化、异常处理。
- **定义与解析**：服务契约定义输入 schema、版本、批量语义、错误码、超时与输出；推理函数应无隐藏训练状态，并对超限输入 fail fast。
- **公式/机制**：请求经历 validate→preprocess→predict→postprocess；幂等读取请求可重试，副作用请求需 idempotency key。
- **资料**：FastAPI [Request Body/Pydantic models](https://fastapi.tiangolo.com/tutorial/body/)；KServe [Inference protocol V2](https://kserve.github.io/website/latest/modelserving/data_plane/v2_protocol/)。
- **最小代码（可执行本地模拟；不是 HTTP 服务）**：
```python
MODEL_VERSION='1.2.0'
def predict(req):
    if set(req)!={'values'} or not isinstance(req['values'],list):
        return {'status':400,'error':'invalid schema'}
    if not req['values'] or len(req['values'])>8:
        return {'status':422,'error':'length out of range'}
    return {'status':200,'version':MODEL_VERSION,'score':sum(req['values'])/len(req['values'])}
r=predict({'values':[1.,3.]})
print(r); assert r['status']==200 and r['score']==2
```
- **检测/实验**：测试空输入、NaN、超长 batch、版本不兼容、超时和并发；真实服务还需认证、限流与结构化日志。
- **常见坑**：预处理散落客户端；返回模型内部异常堆栈；线上模型与 schema 版本没有绑定。

<a id="p05"></a>

### P05 导出、量化与性能剖析【成熟技术，硬件配方演进中】

<!-- readings:start -->
**进一步精读：** [FlashAttention：分块精确注意力](../../readings/papers/flashattention.md) · [PagedAttention：KV缓存分页与共享](../../readings/papers/pagedattention.md) · [vLLM：缓存调度与模型服务](../../readings/projects/vllm.md)
<!-- readings:end -->
- **先修**：P03–P04、数值精度、延迟分位数。
- **定义与解析**：导出把训练图转为部署 IR；量化用低位整数降低存储/算力，但必须验证输出误差和任务指标。吞吐、单请求 p95、冷启动和内存不可互相替代。
- **公式/机制**：对称 int8：$q=\operatorname{clip}(\operatorname{round}(x/s),-127,127)$，$\hat x=sq$，$s=\max|x|/127$。
- **资料**：PyTorch [Export a model to ONNX](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)；ONNX Runtime [Performance tuning](https://onnxruntime.ai/docs/performance/)；PyTorch [Quantization overview](https://docs.pytorch.org/docs/stable/quantization.html)。
- **最小代码（可执行量化模拟）**：
```python
import numpy as np, time
g=np.random.default_rng(0); x=g.normal(size=10000).astype('float32')
s=max(abs(x).max()/127,1e-12); q=np.clip(np.rint(x/s),-127,127).astype('int8')
xhat=q.astype('float32')*s
mae=np.mean(abs(x-xhat)); rel_bytes=q.nbytes/x.nbytes
t0=time.perf_counter(); _=xhat@xhat; ms=(time.perf_counter()-t0)*1000
print(mae,rel_bytes,ms); assert rel_bytes==.25 and mae<s
```
- **检测/实验**：原模型与导出模型在正常/边界输入 `allclose`，再测任务指标、warm/cold p50/p95 和峰值内存。
- **常见坑**：只比文件大小；计时含首次编译却未说明；动态 shape 或 tokenizer 不在导出契约中。

<a id="p06"></a>

### P06 监控、漂移与反馈闭环【稳定原则，检测方法演进中】
- **先修**：P02–P05、统计检验、业务指标。
- **定义与解析**：监控分系统健康、输入数据、预测、延迟标签后的性能与业务结果；漂移是诊断信号，不等于模型必然失效，也不应自动触发未经审批的重训上线。
- **公式/机制**：$\operatorname{PSI}=\sum_i(p_i-q_i)\ln(p_i/q_i)$；还可用 KS/MMD/分类器两样本检验，阈值需用历史误报校准。
- **资料**：Sculley et al. [Hidden Technical Debt §3–5](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)；Rabanser et al. [Failing Loudly §2–3](https://papers.neurips.cc/paper/2019/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html)。
- **最小代码（可执行 PSI）**：
```python
import numpy as np
ref=np.array([.2,.3,.5]); cur=np.array([.1,.3,.6]); eps=1e-8
ref=np.clip(ref,eps,None); cur=np.clip(cur,eps,None)
ref/=ref.sum(); cur/=cur.sum()
psi=np.sum((cur-ref)*np.log(cur/ref))
print(psi); assert psi>=0
alerts={'data_drift':psi>.1,'quality_unknown':True}
print(alerts)
```
- **检测/实验**：模拟 covariate shift、label shift、concept drift，说明仅看输入 PSI 能/不能发现哪类；设计影子评估和人工审批闭环。
- **常见坑**：阈值照搬经验数字；训练/线上分桶不同；模型自反馈改变数据后仍当自然分布。

<a id="p07"></a>

### P07 评测集、统计不确定性与回归决策【稳定原则】
- **先修**：抽样、置信区间、P01–P03。
- **定义与解析**：评测必须声明目标人群、采样单位、指标、版本和污染检查；比较模型宜做 paired bootstrap，并报告效应量和区间而非仅点数。
- **公式/机制**：对样本索引有放回重采样，得到 $\Delta=m_A-m_B$ 的经验分布；序列/用户相关数据应按群组重采样。
- **资料**：SciPy [`stats.bootstrap` 参数/BCa interval](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)；NIST AI RMF [MEASURE function](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)。
- **最小代码（可执行 paired bootstrap）**：
```python
import numpy as np
g=np.random.default_rng(0); a=np.array([1,1,0,1,0,1,1,0]); b=np.array([1,0,0,1,0,0,1,0])
delta=a-b; boots=[]
for _ in range(5000):
    idx=g.integers(0,len(delta),len(delta)); boots.append(delta[idx].mean())
ci=np.quantile(boots,[.025,.975])
print(delta.mean(),ci)
assert ci[0]<=delta.mean()<=ci[1]
```
- **检测/实验**：同数据分别按行与按用户 bootstrap，解释区间差异；在看结果前写出通过、持平、拒绝门槛。
- **常见坑**：测试集反复调参；非独立样本按行重采样；只报 p-value、不报效应量与失败切片。

<a id="p08"></a>

### P08 威胁建模、对抗输入与 LLM 工具安全【安全实践演进中】

<!-- readings:start -->
**进一步精读：** [LangGraph：有状态工具流程与恢复](../../readings/projects/langgraph.md)
<!-- readings:end -->
- **先修**：P04、最小权限、O07–O08。
- **定义与解析**：威胁建模明确资产、攻击者、信任边界与滥用路径；prompt injection 是不可信内容影响模型指令，不能只靠“更强 system prompt”解决。
- **公式/机制**：分层防御：数据标记/隔离→工具 allowlist 与 schema→最小权限→敏感动作确认→超时/预算→审计与补偿。
- **资料**：OWASP GenAI [LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)；MCP [Security best practices](https://modelcontextprotocol.io/specification/2025-06-18/basic/security_best_practices) 与 [Tools safety](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)。
- **最小代码（可执行安全模拟；不执行字符串代码）**：
```python
ALLOWED={'lookup':lambda key:{'a':1}.get(key)}
def dispatch(name,args,approved=False):
    if name not in ALLOWED: return {'ok':False,'error':'tool denied'}
    if set(args)!={'key'} or not isinstance(args['key'],str): return {'ok':False,'error':'bad schema'}
    if not approved: return {'ok':False,'error':'confirmation required'}
    return {'ok':True,'value':ALLOWED[name](**args)}
assert not dispatch('shell',{'cmd':'anything'})['ok']
print(dispatch('lookup',{'key':'a'},approved=True))
```
- **检测/实验**：红队覆盖间接注入、越权工具、数据外传、资源耗尽和供应链；度量 attack success、正常任务成功、误拦率与成本。
- **常见坑**：正则黑名单当边界；让模型自行批准高风险动作；日志泄露密钥；第三方 tool metadata 被默认信任。

<a id="p09"></a>

### P09 治理、隐私、公平与模型卡【稳定框架，法规会变化】
- **先修**：P02、P07–P08、基本隐私与分组指标。
- **定义与解析**：治理把用途、责任、证据、风险接受和退出机制贯穿生命周期；公平指标必须绑定伤害模型，隐私需数据最小化、访问控制与必要时形式化机制。
- **公式/机制**：分组 $\mathrm{TPR}=\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$；equalized odds 比较各组 TPR/FPR。风险按 NIST 的 GOVERN→MAP→MEASURE→MANAGE 循环处理。
- **资料**：NIST [AI RMF 1.0，Part 2 Core](https://doi.org/10.6028/NIST.AI.100-1)；Mitchell et al. [Model Cards §4](https://doi.org/10.1145/3287560.3287596)；Gebru et al. [Datasheets](https://arxiv.org/abs/1803.09010)。
- **最小代码（可执行分组 TPR 诊断）**：
```python
import numpy as np
y=np.array([1,1,0,0,1,1,0,0]); p=np.array([1,0,0,0,1,1,1,0]); group=np.array('AAAABBBB')
stats={}
for g in np.unique(group):
    m=group==g; pos=m&(y==1); neg=m&(y==0)
    stats[g]={'TPR':p[pos].mean(),'FPR':p[neg].mean()}
gap=max(v['TPR'] for v in stats.values())-min(v['TPR'] for v in stats.values())
print(stats,gap); assert 0<=gap<=1
```
- **检测/实验**：写一页模型卡：用途/禁用、数据版本、总体与交叉分组指标、局限、监控、申诉与退役；高风险结论需法务/伦理审查。
- **常见坑**：选一个公平指标宣布“公平”；匿名化等同无隐私风险；模型卡只写优点；把旧版法规描述成当前法律建议。
