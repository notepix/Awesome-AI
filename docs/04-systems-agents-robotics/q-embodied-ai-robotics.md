# Q. 具身智能与机器人


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](p-mlops-safety-evaluation.md) · [下一章 →](../05-frontier/r-frontier-2024-2026.md)

---

## 知识单元

### Q01 机器人系统分层、坐标系与 SE(3)【稳定】
- **先修**：线代、三维几何、控制循环。
- **定义与解析**：机器人系统至少分任务规划、感知/状态估计、运动规划、轨迹控制与硬件安全。学习模型可提出高层技能或参考动作，不能替代坐标变换、限位和实时闭环。
- **公式/机制**：齐次变换 $T_{ab}=\begin{bmatrix}R_{ab}&p_{ab}\\0&1\end{bmatrix}$，链式 $T_{ac}=T_{ab}T_{bc}$，逆变换不是逐元素倒数。
- **资料**：Lynch & Park [Modern Robotics Ch.3，§3.3.1](https://modernrobotics.northwestern.edu/chapters/chapter3/)；ROS [REP-105 Coordinate Frames](https://www.ros.org/reps/rep-0105.html)。
- **最小代码（可执行）**：
```python
import numpy as np
R=np.array([[0,-1,0],[1,0,0],[0,0,1.]],float); p=np.array([1.,2.,0.])
T=np.eye(4); T[:3,:3]=R; T[:3,3]=p
point=np.array([1.,0.,0.,1.]); world=T@point
Ti=np.eye(4); Ti[:3,:3]=R.T; Ti[:3,3]=-R.T@p
assert np.allclose(Ti@world,point)
assert np.allclose(Ti@T,np.eye(4))
```
- **检测/实验**：画清 world/base/tool/camera frame，手算一点的两级变换；为高层命令标明目标 frame 与时间戳。
- **常见坑**：左右乘、主动/被动变换混淆；米/毫米混用；VLM 输出像素位置直接当机械臂基座坐标。

### Q02 感知、标定与状态估计【稳定基础，学习感知演进中】
- **先修**：Q01、概率、高斯分布、传感器噪声。
- **定义与解析**：感知网络给检测/深度等观测，状态估计融合带时间戳的不确定观测与运动模型；标定误差和时延必须显式进入误差预算。
- **公式/机制**：Kalman update：$K=P^-H^T(HP^-H^T+R)^{-1}$，$\hat x=\hat x^-+K(z-H\hat x^-)$。
- **资料**：Welch & Bishop [Kalman Filter §1.4，式 (1.7)–(1.13)](https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf)；ROS 2 [tf2 concepts](https://docs.ros.org/en/rolling/Concepts/Intermediate/About-Tf2.html)。
- **最小代码（可执行一维滤波）**：
```python
import numpy as np
z=np.array([1.2,.9,1.1,1.0]); x,P=0.,1.; Q,R=.01,.04
for obs in z:
    P=P+Q                              # predict
    K=P/(P+R); x=x+K*(obs-x); P=(1-K)*P
print(x,P)
assert abs(x-1)<.15 and P<R
```
- **检测/实验**：人为增加测量噪声、延迟和外参偏差，分别看创新残差与抓取误差；感知置信度不能直接当几何安全界。
- **常见坑**：不同传感器时间不同步；训练集标定固定导致换相机崩溃；无效深度被当 0 米障碍或自由空间。

### Q03 正运动学、逆运动学与可达性【稳定】
- **先修**：Q01、三角函数、Jacobian。
- **定义与解析**：FK 从关节求末端位姿；IK 求满足目标的关节，可能多解、无解或奇异。学习策略给目标位姿后仍需确定性 IK、关节限位与误差检查。
- **公式/机制**：平面二连杆 $x=l_1\cos q_1+l_2\cos(q_1+q_2)$，$y=l_1\sin q_1+l_2\sin(q_1+q_2)$。
- **资料**：Modern Robotics [Ch.4 Forward Kinematics](https://modernrobotics.northwestern.edu/chapters/chapter4/)；[Ch.6 Inverse Kinematics，§6.2 numerical IK](https://modernrobotics.northwestern.edu/chapters/chapter6/)。
- **最小代码（可执行解析 IK/FK 验证）**：
```python
import numpy as np
l1=l2=1.; target=np.array([1.,1.]); x,y=target
c2=(x*x+y*y-l1*l1-l2*l2)/(2*l1*l2); assert abs(c2)<=1
q2=np.arccos(np.clip(c2,-1,1))
q1=np.arctan2(y,x)-np.arctan2(l2*np.sin(q2),l1+l2*np.cos(q2))
fk=np.array([l1*np.cos(q1)+l2*np.cos(q1+q2),l1*np.sin(q1)+l2*np.sin(q1+q2)])
print(q1,q2,fk); assert np.linalg.norm(fk-target)<1e-8
```
- **检测/实验**：测试不可达目标、肘上/肘下多解和伸直奇异位姿；验收含限位、姿态与碰撞，而非仅位置误差。
- **常见坑**：IK 收敛即安全；忽略末端姿态；学习模型生成关节值后绕过限位和速度/加速度约束。

### Q04 碰撞检测、运动规划与轨迹控制【稳定核心】
- **先修**：Q01–Q03、搜索、插值、反馈控制。
- **定义与解析**：motion planning 在配置空间找无碰路径，trajectory generation 加时间、速度和加速度，controller 才在实时闭环执行；规划 waypoint 不等于已执行轨迹。
- **公式/机制**：路径 $q(s)$ 满足 $q(s)\in\mathcal C_{\mathrm{free}}$；时间参数化还需 $q_{\min}\le q\le q_{\max},\ |\dot q|\le v_{\max},\ |\ddot q|\le a_{\max}$。
- **资料**：Modern Robotics [Ch.10 Motion Planning](https://modernrobotics.northwestern.edu/chapters/chapter10/)；MoveIt [Motion Planning concepts](https://moveit.picknik.ai/main/doc/concepts/motion_planning.html) 与 [Planning Scene collision checks](https://moveit.picknik.ai/main/doc/examples/planning_scene/planning_scene_tutorial.html)。
- **最小代码（可执行一维确定性轨迹护栏）**：
```python
import numpy as np
q0,q1=-.5,.8; q=np.linspace(q0,q1,21); dt=.1
obstacle=(-.05,.1); limit=(-1.,1.); vmax=1.
within=np.all((q>=limit[0])&(q<=limit[1]))
collision=np.any((q>=obstacle[0])&(q<=obstacle[1]))
speed=np.max(abs(np.diff(q))/dt)
safe=within and not collision and speed<=vmax
print(within,collision,speed,safe); assert not safe
```
- **检测/实验**：对候选轨迹做连续碰撞、限位、速度、急停和跟踪误差测试；上例离散检查仅教学，不能用于真实安全认证。
- **常见坑**：只查 waypoint、漏段间碰撞；计划成功等同执行成功；用 LLM/VLA 直接发送电机命令。

### Q05 行为克隆、动作分块与 Diffusion Policy【演进中】
- **先修**：I13、Q02–Q04、条件生成/扩散。
- **定义与解析**：机器人 BC 从观测预测动作；action chunk 减少逐步误差和抖动，Diffusion Policy 以条件去噪表示多峰动作序列。输出仍是候选参考轨迹，必须经确定性护栏。
- **公式/机制**：扩散训练 $\mathbb E\|\epsilon-\epsilon_\theta(a_k,o,k)\|_2^2$；receding horizon 每次只执行动作块前一段并重新观测规划。
- **资料**：Chi et al. [Diffusion Policy，§3，Fig.2](https://arxiv.org/abs/2303.04137)；官方 [project/code/data](https://diffusion-policy.cs.columbia.edu/)。
- **最小代码（可执行的一步教学去噪；不是机器人策略）**：
```python
import numpy as np
g=np.random.default_rng(0); target=np.array([.2,.4,.6,.8]); noisy=target+g.normal(0,.3,4)
for _ in range(6):
    predicted_noise=.5*(noisy-target)         # 教学 oracle，不可用于真实策略
    noisy=noisy-predicted_noise
action=np.clip(noisy,-1,1)
valid=np.all(np.abs(np.diff(action))<.3)
print(action,valid); assert valid
```
- **检测/实验**：在未见物体位置、遮挡、扰动恢复上做 ≥5 种子/固定任务集 rollout；报告成功率、干预次数、动作延迟和安全拒绝率。
- **常见坑**：随机帧切分导致同轨迹泄漏；动作坐标/频率不同仍合并数据；只报 action MSE、不报闭环成功。

### Q06 机器人 RL、仿真到现实与安全探索【演进中】
- **先修**：I01–I14、Q01–Q05、仿真器与系统辨识。
- **定义与解析**：机器人 RL 优化闭环回报；domain randomization 在仿真训练时随机物理/视觉参数以覆盖现实。它降低但不消除 reality gap，真实探索需硬件限位、shield、急停与监督。
- **公式/机制**：$\max_\pi\mathbb E_{\xi\sim p(\xi),\,\tau\sim(\pi,P_\xi)}\left[\sum_t\gamma^tr_t\right]$；部署前评估分布应与训练随机化分离。
- **资料**：Tobin et al. [Domain Randomization §III](https://arxiv.org/abs/1703.06907)；Peng et al. [Dynamics Randomization §3](https://arxiv.org/abs/1710.06537)。
- **最小代码（可执行域随机化控制模拟）**：
```python
import numpy as np
def score(k,seed,n=200):
    g=np.random.default_rng(seed); masses=g.uniform(.7,1.3,n); errs=[]
    for m in masses:
        x=1.; dt=.05
        for _ in range(60): x += dt*(-k*x/m)
        errs.append(abs(x))
    return np.mean(errs)
grid=[.5,1.,2.]; train=[np.mean([score(k,s) for s in range(5)]) for k in grid]
best=grid[int(np.argmin(train))]; test=np.mean([score(best,100+s) for s in range(5)])
print(best,test); assert np.isfinite(test)
```
- **检测/实验**：训练种子与评估种子/参数范围分离；加入未随机化的摩擦或延迟，测最差分位而不只均值。
- **常见坑**：在真实机在线试错无 safety layer；调仿真参数看过测试域；把 GPU/仿真能启动当 sim-to-real 成功。

### Q07 VLA、世界模型与分层技能【研究前沿】
- **先修**：VLM、O01–O04、I12–I13、Q01–Q06。
- **定义与解析**：VLA 将视觉、语言映射为动作 token/chunk；世界模型学习状态转移并可在潜空间想象。两者提供泛化先验，不自动保证几何、接触或实时控制正确。
- **公式/机制**：VLA 自回归 $p(a_{1:H}\mid o,\mathrm{instruction})$；世界模型 $p(z_{t+1},r_t\mid z_t,a_t)$ 配合 imagined rollout。
- **资料**：Kim et al. [OpenVLA §3，Fig.2](https://arxiv.org/abs/2406.09246) 与 [官方代码/评估](https://github.com/openvla/openvla)；Hafner et al. [DreamerV3 Methods](https://www.nature.com/articles/s41586-025-08744-2)。
- **最小代码（可执行离散动作解码护栏模拟；不是 VLA）**：
```python
import numpy as np
tokens=np.array([0,128,255]); lo=np.array([-.5,-1.,0.]); hi=np.array([.5,1.,.08])
candidate=lo+(tokens/255)*(hi-lo)
rate_limit=np.array([.2,.2,.02]); previous=np.zeros(3)
safe=np.clip(candidate,previous-rate_limit,previous+rate_limit)
accepted=np.all((safe>=lo)&(safe<=hi))
print(candidate,safe,accepted); assert accepted
# 真实系统还需 IK、碰撞、接触、同步和看门狗
```
- **检测/实验**：分离语义成功、动作可执行率、规划拒绝率、闭环成功与最坏延迟；OpenVLA 论文能力不能外推到未评测机器人。
- **常见坑**：离散动作 token 当精确控制；世界模型视频逼真等同动力学准确；高层成功描述掩盖低层执行失败。

### Q08 具身 Agent、多机器人协同与安全执行【研究前沿】
- **先修**：O07–O08、I14、Q01–Q07、并发与资源锁。
- **定义与解析**：具身 Agent 的高层策略负责拆任务、分配技能和触发恢复；确定性执行层负责 arm assignment、资源锁、IK、碰撞、轨迹同步、接触约束和急停。多机器人消息协议不等于协调策略。
- **公式/机制**：任务图含 precedence $i\prec j$ 与互斥资源 $\operatorname{resource}(i)\cap\operatorname{resource}(j)=\varnothing$；执行前必须原子获得锁并验证 pre/postcondition。
- **资料**：Ahn et al. [SayCan §3：language score × affordance value](https://say-can.github.io/)；ROS 2 [Actions：goal/feedback/result/cancel protocol](https://design.ros2.org/articles/actions.html)；MoveIt [Planning Scene](https://moveit.picknik.ai/main/doc/examples/planning_scene/planning_scene_tutorial.html)。
- **最小代码（可执行确定性资源锁模拟；策略输出仅为候选）**：
```python
held=set()
def execute(step):
    need=set(step['resources'])
    if held&need: return {'ok':False,'reason':'resource conflict'}
    if not step.get('ik_ok') or not step.get('collision_free'): return {'ok':False,'reason':'unsafe plan'}
    held.update(need)
    result={'ok':True,'action':step['skill']}
    held.difference_update(need)
    return result
candidate={'skill':'place','resources':['arm1','zoneA'],'ik_ok':True,'collision_free':False}
print(execute(candidate)); assert not execute(candidate)['ok']
```
- **检测/实验**：注入通信延迟、机器人掉线、锁冲突、抓取失败和人进入工作区；测任务成功、死锁、恢复时间、人工接管与安全违规为零。
- **常见坑**：双臂平台就声称协同控制；LLM 语言承诺当作资源锁；局部重试反复碰撞；把 planned waypoint、控制指令和物理执行轨迹混为一谈。
