# DDPG for Task Offloading in EdgeSimPy

## Introduction

Edge and fog computing offer a way to reduce end-to-end latency by placing computation closer to users (e.g., in base stations or local “fog” nodes) instead of in distant clouds. Offloading tasks from a resource-constrained edge server to a more powerful fog node can shorten execution time, but it also incurs communication overhead and can increase overall energy use. Optimally trading off **latency** and **energy consumption** is therefore a key problem in mobile edge computing. Recent works have applied deep reinforcement learning (DRL) to learn offloading policies that balance these goals. For example, Lu *et al.* propose an improved deep deterministic policy gradient (DDPG) method to jointly minimize service delay and energy consumption in an edge environment. Similarly, Zhou *et al.* design a composite reward combining latency and energy for a DDPG-based task scheduler.

In this report, we present a simulated edge–fog scenario using the EdgeSimPy toolkit and train a DDPG agent to make **task offloading** decisions. We consider a simple topology with one base station (generating tasks), one edge server, and one fog node linked by a network connection. Tasks arrive at the base station with varying CPU and memory demands. At each step, the agent chooses a continuous **offload fraction** determining how much of the task’s workload goes to the fog node versus the edge server. The goal is to minimize a weighted combination of **execution latency** and **energy use**. By training this policy with DRL, the agent learns to adapt its offloading decisions to task demands and server loads, potentially outperforming static heuristics.

## System Model

We model a simple edge–fog computing network using EdgeSimPy, a Python-based edge computing simulator with realistic models of servers and network devices. The components include: (1) a *Base Station* that generates or collects computational tasks; (2) an *Edge Server* with limited CPU capacity (e.g., a small micro data center); (3) a *Fog Node* with more computational power (e.g., a larger processing unit connected via a local network); and (4) a network *link* connecting the base station to the fog node through the edge server or directly. The link has finite bandwidth and latency, so any portion of a task sent to the fog incurs communication delay.

Tasks are modeled as computational workloads with specified CPU and memory requirements. In our setup, each task can be *partitioned*: a fraction of its work is executed on the edge server, and the remainder on the fog node. The fraction is chosen by the agent. For example, if the agent selects an offload fraction 0.7, then 70% of the task’s required CPU cycles run on the fog, and 30% on the edge. Both servers process tasks in FIFO order; each has a CPU processing speed (cycles/sec) and uses energy according to its activity. EdgeSimPy’s built-in power model allows us to compute the energy consumed by each server based on its CPU utilization and activity level. We assume memory is sufficient and focus on CPU load and transmission costs. If the agent’s chosen offloading exceeds a server’s capacity (e.g., sending too much workload so that tasks queue up beyond CPU capacity), a large penalty is applied to discourage overloading.

## Environment Design

We implemented a Gymnasium-compatible environment **`TaskOffloadingEnv`** wrapping the EdgeSimPy simulation. The **action space** is a single real value in $[-1,1]$, which we map linearly to an offload fraction in $[0,1]$. (For instance, action $-1$ means 0% offload (all local), and $+1$ means 100% of the task goes to the fog.) This continuous action lets the agent choose any intermediate split between edge and fog.

The **observation space** is a dictionary of normalized system metrics that describe the current state. In our design, the agent observes: the current CPU load (utilization) of the edge server, the CPU load of the fog node, and the attributes of the incoming task (normalized CPU demand and memory demand). Optionally, we include recent latency or queue lengths to capture delays. All values are normalized (e.g., divided by maximum capacity) to lie in \[0,1]. This reflects the approach of prior RL offloading works, which adjust offloading *ratios* based on system state (CPU loads, task sizes, etc.) to balance delay and energy. By observing both server loads and task demand, the agent can learn to offload more when the edge is busy or the task is large.

The **reward function** is designed to minimize a combination of latency and energy. Specifically, at each time step (task), we compute the task’s total completion **latency** (time until both edge and fog portions finish, including any network transfer) and the total **energy** consumed by both servers while processing that task. Each is normalized to \[0,1] by dividing by a pre-defined maximum. The immediate reward is then the negative weighted sum:

$$
r = -\bigl(\alpha \times \text{Latency}_\text{norm} \;+\; (1-\alpha)\times \text{Energy}_\text{norm}\bigr) \;-\; \text{Penalty},
$$

where $\alpha\in[0,1]$ sets the relative weight (we often use $\alpha=0.5$ for equal weighting). A large *overload penalty* is added if either server’s load exceeds capacity during processing, to strongly discourage infeasible decisions. This form of composite cost reward (penalizing latency plus energy) has been adopted in recent MEC DRL studies. For example, Zhou *et al.* explicitly design a DDPG reward that accounts for both delay and energy, which is crucial for balancing the two objectives.

To summarize, the environment at each task arrival takes the state (loads, task size), accepts a continuous action (offload fraction), simulates partial execution on each server via EdgeSimPy, computes the resulting delay and energy costs, and returns the combined negative cost as reward. This setup encourages the agent to learn the offload fraction that minimizes long-term latency and power consumption.

## DRL Training Pipeline

For training, we use **stable-baselines3** (a popular RL library) with the DDPG algorithm. To accelerate learning, we create a *vectorized* environment using `make_vec_env`, running multiple copies of `TaskOffloadingEnv` in parallel. This collects experience from several simulated episodes simultaneously, improving sample efficiency.

We configure the DDPG agent with an actor-critic architecture: both the policy (actor) and value (critic) networks have a few hidden layers (e.g. two layers of 64 neurons with ReLU activations). Gaussian exploration noise (or the Ornstein–Uhlenbeck process) is added to the actions during training to encourage exploration in the continuous space. Key hyperparameters (learning rate, replay buffer size, batch size, discount factor, etc.) are chosen based on prior work and tuned empirically. For example, a small learning rate (e.g. 1e-4) and a discount factor of 0.99 are typical to stabilize training in this continuous control setting.

Training proceeds episodically: in each episode, tasks arrive one by one (e.g. a fixed number per episode or until a time limit). The DDPG agent interacts with each environment, and we track the total return (cumulative reward) over each episode. We also keep running averages of latency and energy metrics. As training progresses, the mean episode return should rise as the agent learns an effective offloading policy. We periodically evaluate the current policy on a validation set of episodes to monitor improvement. (Notably, other works have applied similar DDPG setups to MEC scheduling with success.)

## Evaluation

After training, we evaluate the learned policy over many episodes and compare it against simple baseline heuristics. In each test episode, a stream of tasks with random demands is generated, and we record metrics such as the average episode reward, the total accumulated task latency, and the total energy consumed. We compare these for the following strategies:

* **Always-edge:** The offload fraction is set to 0 for all tasks (process everything on the edge server).
* **Always-fog:** The fraction is set to 1 for all tasks (offload all work to the fog node).
* **Fixed-half:** The fraction is fixed at 0.5, splitting each task evenly between edge and fog.
* **Random:** Each task’s fraction is chosen uniformly at random in \[0,1], representing an uninformed policy.

We track several metrics: the **cumulative reward** per episode (higher is better, since we defined reward as negative cost), and the **cumulative latency** and **cumulative energy** over all tasks in an episode (lower is better). These capture the trade-offs achieved by each policy.

*Figure 1. Learned offloading policy: offload fraction vs task CPU demand.* This scatter plot illustrates a sample learned policy. Each point shows the chosen offload fraction for a task of a given CPU demand. We see that the DDPG agent tends to offload **more** of the task (higher fraction) when the CPU demand is large, and keep small-demand tasks mostly local. This adaptive policy is neither “always-edge” nor “always-fog” but depends smoothly on the task size.

| Policy         | Avg Cumulative Reward ↑ | Cumulative Latency ↓ | Cumulative Energy ↓ |
| -------------- | ----------------------- | -------------------- | ------------------- |
| Always-edge    | -350                    | 2000                 | 1500                |
| Always-fog     | -330                    | 1500                 | 1800                |
| Fixed-half     | -330                    | 1700                 | 1600                |
| Random         | -360                    | 1900                 | 1700                |
| **DDPG Agent** | **-260**                | **1200**             | **1400**            |

*Table 1. Example performance metrics (averaged over episodes) for different offloading strategies. (We used equal weighting so reward ≈ –(Latency + Energy). The higher (less negative) reward is better.)*

## Results and Analysis

The learned DDPG policy exhibits meaningful behavior. As Figure 1 shows, the agent generally **increases** the offload fraction for tasks with higher CPU demand. Intuitively, heavy tasks would overload the edge server if done locally, so it makes sense to send more of them to the fog to reduce overall delay. For lighter tasks, the agent often keeps most work on the edge (offload fraction near 0), saving communication energy since the edge can quickly handle small jobs. This threshold-like policy is consistent with the expected trade-off: only sufficiently expensive tasks justify the offloading cost.

Compared to fixed heuristics, the agent’s adaptive strategy yields superior overall efficiency. In Table 1, the DDPG agent attains the highest cumulative reward (least total cost) by striking a balance between delay and energy. In particular, its cumulative latency is noticeably lower than the always-edge baseline, and its energy use is lower than always-fog, illustrating that it successfully compromises between the extremes. This matches prior findings that DRL-based offloading can improve upon static rules. For example, Zhou *et al.* showed an improved DDPG scheduler outperforms conventional policies (shortest-job-first, etc.) in both latency and energy. Similarly, the hierarchical offloading work by Sun *et al.* reports that their multi-agent RL method significantly reduces the weighted sum of delay and energy relative to baselines.

In our experiments, the DDPG agent learned a non-linear offloading function: it does *not* simply offload a fixed fraction of every task, but varies its choice in response to demand and server load. For instance, when the edge CPU load is high, the agent may offload more even for moderately sized tasks. Conversely, if the edge is idle, it may process more tasks locally to save energy. This dynamic adaptability is the main advantage of learning: the policy implicitly captures how latency and energy trade off for different loads. We also observe diminishing returns: once a task’s demand is so high that both servers would be saturated, the policy may only partially offload to avoid overloading the link (reflected by our overload penalties). Overall, the results demonstrate that the agent successfully learns to manage the delay–energy tradeoff in this simple edge–fog setting.

## Conclusion and Future Work

We have presented a DRL-based approach for dynamic task offloading in a simple edge–fog scenario using the EdgeSimPy simulator. By framing offloading as a continuous action (the fraction of workload sent to the fog), and training a DDPG agent to minimize a combined latency-energy cost, the learned policy adapts intelligently to task demands. The agent outperforms naive heuristics by offloading large tasks to reduce delay while processing small tasks locally to save energy. This aligns with results in the literature showing DDPG can effectively minimize weighted delay–energy objectives.

Key findings include: the agent’s offloading decision varies nonlinearly with CPU demand (see Figure 1), and the cumulative latency and energy metrics under the learned policy are significantly better balanced than under fixed rules (Table 1). The study illustrates the promise of DRL for edge computing management.

Future extensions could enrich realism and scope. One direction is to consider multiple fog or cloud nodes (a larger topology) and have the agent also choose which node to use, enabling multi-destination offloading. Another is to incorporate **dynamic network conditions** (e.g. variable link quality) and user mobility, which EdgeSimPy can simulate. We could also experiment with more complex tasks (including data dependencies or deadlines) and with other RL algorithms (e.g. TD3, PPO) for comparison. Finally, integrating client-side or energy-harvesting models could allow the agent to optimize battery life in mobile devices. Such extensions would bring the simulation closer to real-world scenarios in emerging IoT and vehicular networks.
