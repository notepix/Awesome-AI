# RAG：检索、重排、引用与拒答

这是可离线运行的完整检索问答实验。默认答案为证据原文的抽取式输出，不把字符串拼接称作 LLM 生成。先读 [公共环境](../README.md)。

## 运行

```powershell
python -m labs.run rag prepare
python -m labs.run rag index
python -m labs.run rag eval
python -m labs.run rag demo --prompt "Where is the robotics laboratory?"
python -m labs.run rag demo --prompt "Who discovered planet Neptune?"
```

`train` 是 `index` 的别名；这里训练的是索引中的统计表示，没有训练生成模型。数据由十条虚构 Lumen 校园记录、独立校准问题和测试问题构成，完全由仓库编写。每条短文档就是一个带稳定 ID 的证据片段；长文档分块可在这个接口上继续扩展。

## 检索与回答链

1. 小写英文词项进入手写 BM25，保留长度归一化和 IDF。
2. TF-IDF 经 TruncatedSVD 得到潜在语义向量并归一化；它是统计语义表示，**不是神经编码器**。
3. BM25 分数经有界变换后与语义相似度加权；前六条进入词项覆盖率规则重排，再返回前三条。
4. 仅用 validation 问题选择回答阈值，优化可回答/不可回答判断的校准准确率。测试问题不参与阈值选择。
5. 最高分低于阈值时输出 `NO_ANSWER` 和空引用，否则返回最高分证据原文与真实文档 ID。

参数是教学选择，不保证适合其他语料。更换语料、编码器、重排器或分词方案后必须重新建索引并重新校准。

## 指标分别看

- `retrieval_recall_at_3`：可回答问题的正确证据是否进入前三。
- `answer_keyword_accuracy`：可回答问题的预设答案关键词是否出现在输出中；这是窄小数据上的代理指标，不等同语义正确性。
- `citation_accuracy_supported`：可回答问题是否引用了标注的正确文档。
- `no_answer_accuracy`：对全部测试问题的回答/拒答分类是否正确。
- `generation_failures`：可选生成器出现非法 JSON 或非法引用的数量。

阶段 JSON 保存逐问题证据、分数、回答和引用，便于定位“没检索到”“生成错了”“拒答错了”的不同问题。

## 本地 MiniLM + Qwen

自行准备两个完整本地模型目录，先按公共说明安装可选依赖。MiniLM 按 attention mask 平均池化并归一化；重排仍明确采用规则方法。Qwen 只根据检索证据输出规定 JSON，引用必须属于实际检索到的 ID。

```powershell
python -m labs.run rag index --config labs/rag/local.json --embedding-path D:/models/paraphrase-multilingual-MiniLM-L12-v2
python -m labs.run rag eval --config labs/rag/local.json --embedding-path D:/models/paraphrase-multilingual-MiniLM-L12-v2 --model-path D:/models/Qwen2.5-0.5B-Instruct
python -m labs.run rag demo --config labs/rag/local.json --embedding-path D:/models/paraphrase-multilingual-MiniLM-L12-v2 --model-path D:/models/Qwen2.5-0.5B-Instruct --prompt "When does the library open?"
```

评估先计算问题向量，再释放编码器，最后加载生成器；两者不需要同时驻留显存。只校验引用 ID 并不能证明整句话由证据支持，因此保留原始生成文本和独立指标。非法生成会标记为失败，不能因为回退拒答就隐藏生成错误。

## 文件与测试

实现：[`rag.py`](../ai_learning/rag.py)。产物：`data.json`、`index_offline.pkl` / `index_local.pkl` 和阶段结果 JSON。pickle 仅用于本项目自己生成的索引，不应加载外部不可信文件。索引内部保存原始文档与编码器路径；更新文档后重新执行 index。

公共测试覆盖 BM25 未知词、完整建库/评估链、未知主题拒答和指标范围；真实质量以自己的结果记录为准，不设置未经测量的性能保证。

## 回到讲义与精读

[检索与 Agent 讲义](../../docs/08-walkthroughs/06-retrieval-agents.md) · [DPR 论文精读](../../readings/papers/dpr.md) · [RAG 论文精读](../../readings/papers/rag.md) · [Transformers 本地模型接口精读](../../readings/projects/transformers.md)。
