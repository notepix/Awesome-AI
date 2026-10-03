# B+. 经典人工智能：搜索、约束、逻辑与规划

现代深度学习不是 AI 的全部。搜索、约束满足、逻辑和规划提供“显式状态、规则与可验证决策”的另一套工具，也直接连接强化学习、Agent 和机器人。


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](b-programming-data-experiments.md) · [下一章 →](c-machine-learning.md)

---

## 配套系统讲解

[串联本章的推导、例子与练习解析](../08-walkthroughs/08-domain-bridges.md) · [论文与源码精读](../../readings/README.md) · [完整实践代码](../../labs/README.md)

原有知识单元保留稳定编号；概念卡用于定位，系统讲解用于连接完整过程。

## 知识单元

<a id="b07"></a>

### B07 `[核心]` 状态空间、图搜索与问题建模

**先修**：A01 集合/函数、B02 复杂度。

**定义与解析**：搜索问题由状态集合、初始状态、动作、转移函数、动作代价和目标测试组成。关键不是先选 BFS/DFS，而是构造只保留决策所需信息的 search state。图搜索保存已到达状态，避免搜索树中的重复路径。

**机制**：BFS 以深度排序，UCS 以累计代价 $g(n)$ 排序；时间和内存取决于 branching factor、解深和重复状态。

**资料定位**：[Berkeley CS188 1.2 State Spaces](https://inst.eecs.berkeley.edu/~cs188/textbook/search/state.html) 与 [1.3 Uninformed Search](https://inst.eecs.berkeley.edu/~cs188/textbook/search/uninformed.html)。

```python
# 代码：可运行。BFS 求无权图最短动作序列。
from collections import deque
graph = {"S": [("A", "left"), ("B", "right")],
         "A": [("G", "down")], "B": [("A", "back")], "G": []}
q, reached = deque([("S", [])]), {"S"}
while q:
    state, path = q.popleft()
    if state == "G": break
    for nxt, action in graph[state]:
        if nxt not in reached:
            reached.add(nxt); q.append((nxt, path + [action]))
assert path == ["left", "down"]
```

**检测题/小实验**：把 `reached` 删除，加入环 `A→S`，观察扩展次数；为“吃完所有食物”设计状态，说明为何只存当前位置不够。

**常见坑**：把世界的所有信息都塞进搜索状态；用 list 做 membership；混淆搜索树节点与世界状态。

<a id="b08"></a>

### B08 `[核心]` 启发式搜索与 A*

**先修**：B07、A04 距离。

**定义与解析**：启发式 $h(n)$ 估计从当前状态到目标的剩余代价。A* 按 $f(n)=g(n)+h(n)$ 扩展。树搜索中 admissible 表示不高估；图搜索通常还要求 consistency 或允许重新打开更便宜的路径。

**机制**：open list 每次弹出 $f$ 最小节点；若 $0\le h(n)\le h^*(n)$，则启发式 admissible。更强的 consistency 要求每条边满足 $h(n)\le c(n,n')+h(n')$，使图搜索不必反复重新打开已扩展节点。

**资料定位**：[CS188 1.4 Informed Search，尤其 1.4.3–1.4.5](https://inst.eecs.berkeley.edu/~cs188/textbook/search/informed.html)。

```python
# 代码：可运行。四邻域网格上的 A*。
import heapq
start, goal, walls = (0, 0), (3, 3), {(1, 1), (1, 2)}
h = lambda p: abs(p[0]-goal[0]) + abs(p[1]-goal[1])
frontier, best = [(h(start), 0, start, [])], {start: 0}
while frontier:
    _, g, p, path = heapq.heappop(frontier)
    if p == goal: break
    for d in ((1,0),(-1,0),(0,1),(0,-1)):
        q = (p[0]+d[0], p[1]+d[1]); ng = g + 1
        if 0 <= q[0] <= 3 and 0 <= q[1] <= 3 and q not in walls and ng < best.get(q, 1e9):
            best[q] = ng; heapq.heappush(frontier, (ng+h(q), ng, q, path+[q]))
assert len(path) == 6
```

**检测题/小实验**：把 $h$ 乘以 2，找一个不再保证最优的例子；比较 `h=0`、曼哈顿距离和真实剩余距离的扩展节点数。

**常见坑**：说“有启发式就最优”；关闭状态后不处理更低代价路径；用测试答案反向设计泄漏启发式。

<a id="b09"></a>

### B09 `[核心]` 约束满足问题（CSP）

**先修**：A01 关系、B07 搜索。

**定义与解析**：CSP 由变量、每个变量的 domain 和 constraints 组成；目标是找到满足全部约束的赋值，而非优化神经网络损失。回溯搜索配合变量/取值排序、forward checking 和 arc consistency 可显著剪枝。

**机制**：回溯每次只扩展与已赋值变量一致的取值；MRV 优先选择剩余 domain 最小的变量，forward checking 删除邻居的不相容取值，AC-3 反复保证每个弧上的取值至少有一个支持值。

**资料定位**：[CS188 2.1 CSP 定义](https://inst.eecs.berkeley.edu/~cs188/textbook/csp/csps.html)、[2.2 求解](https://inst.eecs.berkeley.edu/~cs188/textbook/csp/solving.html)与[2.3 Filtering](https://inst.eecs.berkeley.edu/~cs188/textbook/csp/filtering.html)。

```python
# 代码：可运行。澳洲地图三色问题的最小回溯。
nodes = ["WA", "NT", "SA", "Q"]
edges = {tuple(sorted(e)) for e in [("WA","NT"),("WA","SA"),("NT","SA"),
                                    ("NT","Q"),("SA","Q")]}
colors = "RGB"
def solve(i=0, assign=None):
    assign = {} if assign is None else assign
    if i == len(nodes): return assign.copy()
    x = nodes[i]
    for c in colors:
        if all(assign.get(y) != c for y in nodes if tuple(sorted((x,y))) in edges):
            assign[x] = c
            out = solve(i+1, assign)
            if out: return out
            del assign[x]
solution = solve(); assert solution and solution["WA"] != solution["NT"]
```

**检测题/小实验**：加入 MRV 和 forward checking，统计递归次数；解释为何“所有变量选一个不同颜色”错误地增加了约束。

**常见坑**：把约束检查放到完整赋值后；混淆局部一致性和全局可解；没有明确 domain。

<a id="b10"></a>

### B10 `[核心]` 对抗搜索、Minimax 与 Alpha–Beta

**先修**：B07、A17 最优化。

**定义与解析**：在双人零和、完全信息、轮流行动的有限博弈中，MAX 假设 MIN 也最优，选择最大化最坏结果的动作。Alpha–beta pruning 不改变 minimax 值，只剪掉不可能影响当前决定的分支；其效率强烈依赖动作排序。

**机制**：递推为 $V(s)=\max_a V(T(s,a))$（MAX 层）或 $V(s)=\min_a V(T(s,a))$（MIN 层）。若当前下界 $\alpha$ 已不小于上界 $\beta$，该分支不可能改变祖先决策，可以剪枝。

**资料定位**：[CS188 3.2 Minimax](https://inst.eecs.berkeley.edu/~cs188/textbook/games/minimax.html)；MCTS 见 [3.5](https://inst.eecs.berkeley.edu/~cs188/textbook/games/mcts.html)。

```python
# 代码：可运行。嵌套列表是博弈树，叶子是 MAX 视角效用。
tree = [[[3, 5], [2, 9]], [[0, 1], [7, 4]]]
def minimax(node, maximizing):
    if isinstance(node, int): return node
    values = [minimax(child, not maximizing) for child in node]
    return max(values) if maximizing else min(values)
value = minimax(tree, True)
assert value == 5
```

**检测题/小实验**：手算根节点值；实现 alpha–beta 并统计叶子访问数；改变子节点顺序，观察剪枝量但验证根值不变。

**常见坑**：真实对手随机却仍声称 minimax 概率最优；评估函数与终局效用混淆；把搜索深度增大当作无成本提升。

<a id="b11"></a>

### B11 `[核心]` 命题逻辑、蕴含与规则推理

**先修**：A01 逻辑记号。

**定义与解析**：语法规定合法公式，语义给每个模型/世界赋真假值。知识库 $KB\models q$ 表示所有满足 $KB$ 的模型都满足 $q$；这不同于某个证明算法是否成功。Horn rule 的 forward chaining 可以从已知事实反复推出新事实。

**机制**：对 Horn 规则 $p_1\land\cdots\land p_k\Rightarrow q$，当全部前件已在事实集时加入 $q$，直到没有新事实。可靠性（soundness）和完备性（completeness）是证明算法相对语义的性质，不是同一个概念。

**资料定位**：[CS188 10.3 Propositional Logic](https://inst.eecs.berkeley.edu/~cs188/textbook/logic/propositional.html)、[10.4 Inference](https://inst.eecs.berkeley.edu/~cs188/textbook/logic/inference.html)与 [10.6 Forward Chaining](https://inst.eecs.berkeley.edu/~cs188/textbook/logic/forward.html)。

```python
# 代码：可运行。规则 antecedents -> consequent 的前向链。
facts = {"rain", "have_umbrella"}
rules = [({"rain", "have_umbrella"}, "stay_dry"),
         ({"stay_dry"}, "can_walk")]
changed = True
while changed:
    changed = False
    for need, out in rules:
        if need <= facts and out not in facts:
            facts.add(out); changed = True
assert "can_walk" in facts
```

**检测题/小实验**：删除 `have_umbrella` 后还能否推出 `stay_dry`？增加互相依赖但无初始事实的规则，解释为何不会凭空推出结论。

**常见坑**：把蕴含当作相关性；把“未证明”当作“为假”；规则冲突时没有一致性或优先级策略。

<a id="b12"></a>

### B12 `[核心]` 自动规划、STRIPS 与执行监控

<!-- readings:start -->
**进一步精读：** [ReAct：推理行动与外部观察](../../readings/papers/react.md) · [LangGraph：有状态工具流程与恢复](../../readings/projects/langgraph.md)
<!-- readings:end -->

**先修**：B07–B11。

**定义与解析**：经典规划把动作写成 preconditions、add effects 和 delete effects，在符号状态空间中寻找使 goal 成立的动作序列。规划器给出的是离散计划；真实执行还需要感知、控制、前置/后置条件检查和失败恢复。

**机制**：若动作 $a$ 的前置条件满足，即 $\operatorname{pre}(a)\subseteq s$，则 STRIPS 状态更新为 $s'=(s\setminus\operatorname{del}(a))\cup\operatorname{add}(a)$；搜索目标是找到动作序列使 $G\subseteq s_T$。

**资料定位**：[AIMA 4e Chapter 11 Automated Planning 总目录](https://aima.cs.berkeley.edu/)；状态搜索背景回看 [CS188 1.2](https://inst.eecs.berkeley.edu/~cs188/textbook/search/state.html)。

```python
# 代码：可运行。极小 STRIPS 规划器。
from collections import deque
actions = {
 "pickup": ({"hand_empty", "object_on_table"}, {"holding"}, {"hand_empty"}),
 "place":  ({"holding"}, {"object_on_shelf", "hand_empty"}, {"holding"})}
start, goal = frozenset({"hand_empty", "object_on_table"}), {"object_on_shelf"}
q, seen = deque([(start, [])]), {start}
while q:
    state, plan = q.popleft()
    if goal <= state: break
    for name, (pre, add, delete) in actions.items():
        if pre <= state:
            nxt = frozenset((state - delete) | add)
            if nxt not in seen: seen.add(nxt); q.append((nxt, plan+[name]))
assert plan == ["pickup", "place"]
```

**检测题/小实验**：加入“货架被占用”前置条件与清理动作；模拟 `pickup` 后物体掉落，设计后置条件检查和局部恢复。

**常见坑**：计划出来就等同于执行成功；状态遗漏资源占用；动作无副作用模型；LLM 生成步骤后不做符号或物理验证。

---
