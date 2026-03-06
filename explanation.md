# MindDrive RL Training — Detailed Technical Explanation

This document explains in full detail how the MindDrive reinforcement learning framework trains
the StreamVLN agent: what is being optimized, how the agent perceives and acts in the world,
how every component of the reward is computed, how PPO updates the weights, and why each
design choice was made.

---

## Table of Contents

1. [Background: What is StreamVLN?](#1-background-what-is-streamvln)
2. [The Problem with Imitation Learning](#2-the-problem-with-imitation-learning)
3. [What RL Training Changes](#3-what-rl-training-changes)
4. [The Actor-Critic Architecture (Block 1)](#4-the-actor-critic-architecture-block-1)
5. [LoRA Adapters and the Reference Model (Block 2)](#5-lora-adapters-and-the-reference-model-block-2)
6. [Online Rollout Collection in Habitat (Block 3)](#6-online-rollout-collection-in-habitat-block-3)
7. [The Reward System in Full Detail (Block 4)](#7-the-reward-system-in-full-detail-block-4)
8. [Advantage Estimation with GAE-lambda](#8-advantage-estimation-with-gae-lambda)
9. [PPO Optimization — Step by Step](#9-ppo-optimization--step-by-step)
10. [The Full Training Loop](#10-the-full-training-loop)
11. [Key Hyperparameters and Their Roles](#11-key-hyperparameters-and-their-roles)
12. [GPU Memory Strategy](#12-gpu-memory-strategy)
13. [Failure Modes and Mitigations](#13-failure-modes-and-mitigations)

---

## 1. Background: What is StreamVLN?

StreamVLN is a large multimodal language model designed for Vision-and-Language Navigation (VLN).
Given a natural-language instruction like "Walk down the hallway, turn left at the painting, and
stop in front of the blue door", the model must navigate through a photorealistic 3D environment
(Matterport3D scenes rendered inside the Habitat simulator) by issuing a sequence of discrete
navigation commands.

The model architecture is built on a Qwen3-VL 4B language backbone. At each step it receives:

- An RGB image from the agent's forward-facing camera (processed by a SigLip vision tower into
  patch embeddings).
- A depth map from the depth sensor (processed into metric distances in meters).
- A camera pose matrix (4x4 homogeneous transform encoding x, y, z position and yaw).
- Camera intrinsics (focal length, principal point).
- A history of past observations stored in a streaming KV cache (slow memory).

StreamVLN implements a "slow-fast" context model. The fast stream processes the current
observation every step. The slow stream is a compressed history: every `num_frames=32` steps the
KV cache is reset and a set of sampled historical frames are injected to maintain long-range
context. This keeps inference fast while allowing the model to remember where it has been.

The pre-trained model was trained via imitation learning (IL) on expert demonstration trajectories
from the R2R (Room-to-Room) and RxR (Room-across-Room) datasets. It learned to mimic expert
behavior — but only what the expert demonstrations show.

---

## 2. The Problem with Imitation Learning

Imitation learning has two fundamental limitations for navigation:

**Distribution shift:** The IL model learned from expert paths. During inference it generates its
own actions, which inevitably deviate from the expert distribution. Once it makes an imperfect
step, it finds itself in a state it has never seen in training, and its predictions degrade
rapidly. The model was never taught how to recover from mistakes.

**STOP-bias:** Expert demonstrations always end with a correct STOP action. The model sees STOP
as a frequent and "safe" action. After IL training, the model exhibits a strong prior to output
STOP very early in an episode — sometimes on the very first step — because it has learned that
STOP is often the right answer in training. This is catastrophic for navigation, where the agent
must walk tens to hundreds of steps before stopping.

**No exploration signal:** IL can only be as good as its demonstrations. Novel shortcuts, recovery
strategies, or environment-specific navigation patterns that do not appear in training data cannot
be learned.

Reinforcement learning solves all three: the agent learns from its own experience, recovers from
its own mistakes via reward shaping, and can discover strategies beyond what demonstrations show.

---

## 3. What RL Training Changes

We do NOT retrain the entire StreamVLN model from scratch. That would destroy the rich multimodal
representations learned during IL pre-training and require enormous compute.

Instead, the RL phase makes surgical changes:

**What stays frozen (99% of parameters):**
- The entire SigLip vision tower (processes RGB images into patch embeddings)
- The multimodal projector (bridges vision embeddings into language space)
- The depth and pose encoders
- All 28 transformer layers of the Qwen3-VL backbone — their attention weights, MLP weights,
  layer norms, embeddings

**What gets trained (~1% of parameters):**
- LoRA adapter matrices injected into 7 projection layers of each transformer layer
- A new scalar Value Head (LayerNorm → Dropout → Linear → scalar) attached to the model output

The LoRA adapters have rank r=64, alpha=64 (scaling factor = alpha/r = 1.0). At rank 64 across
7 modules × 28 layers, the trainable parameter count is approximately 161M out of 4,734M total
(3.41%). With alpha=r, each LoRA update has unit scaling — previous versions used alpha=16 with
r=64, giving a scaling of 0.25 which was too conservative and slowed learning significantly.

The Value Head is a new randomly-initialized module. It reads the last hidden state of the
transformer (shape [batch, seq_len, hidden_size]) and projects it to a scalar via:
  hidden_states → LayerNorm → Dropout(0.0) → Linear(hidden_size, 1) → squeeze → [batch, seq_len]

The Value Head is initialized with Gaussian weights (mean=0, std=0.2) to keep initial value
predictions near zero, preventing the critic from dominating early training with large spurious
gradients.

---

## 4. The Actor-Critic Architecture (Block 1)

The complete model is called `StreamVLNWithValueHead`. It wraps the pre-trained
`StreamVLNForCausalLM` with the Value Head. A single forward pass produces three outputs:

```
(lm_logits, loss, value) = model(input_ids, images, depths, poses, intrinsics, ...)
```

- `lm_logits`: shape [B, seq_len, vocab_size] — the language model's next-token probability
  distribution over the full vocabulary (~32,000 tokens). This is the Actor output.
- `loss`: cross-entropy loss if labels are provided (used during IL training, set to None during RL).
- `value`: shape [B, seq_len] — per-token scalar value estimate from the Value Head. This is the
  Critic output. During RL we use `value[:, -1]`, the estimate at the last token position, as the
  value for the current state.

The Actor and Critic share all weights except the Value Head linear layer. This means:
- They see the same visual features, same language context, same history.
- The hidden states that inform the policy logits are exactly the same hidden states used to
  estimate the value.
- A single GPU forward pass gives us both the action probabilities AND the state value, making
  rollout collection compute-efficient.

**Why output_hidden_states=True?**

The Value Head needs access to the last hidden state. By setting `output_hidden_states=True`
inside `StreamVLNWithValueHead.forward()`, the transformer returns all layer hidden states, and
we extract `hidden_states[-1]` (the final layer's output) to feed into the Value Head. This flag
is set automatically inside the wrapper so callers do not need to remember it.

---

## 5. LoRA Adapters and the Reference Model (Block 2)

### LoRA Adapters

Low-Rank Adaptation (LoRA) works by adding a low-rank decomposition to each target weight matrix.
For a weight matrix W (shape [d_out, d_in]), LoRA adds:

```
W_effective = W_frozen + (alpha/r) × B × A
```

where:
- A has shape [r, d_in] — initialized with Gaussian noise
- B has shape [d_out, r] — initialized with zeros (so the adapter contributes nothing at the start)
- r=64 is the rank
- alpha=64 gives scaling factor 1.0

Only A and B are trainable. W_frozen never changes. The result is that at initialization, the
LoRA contribution is exactly zero — the model starts from its IL-pretrained behavior and the RL
phase nudges it via the adapter gradients.

LoRA is applied to 7 projection modules inside each transformer layer:
- `q_proj`, `k_proj`, `v_proj` — query, key, value projections in attention
- `o_proj` — output projection in attention
- `gate_proj`, `up_proj`, `down_proj` — the MLP (SwiGLU) gating projections

Vision tower and multimodal projector LoRA parameters are frozen by default. They consume
significant VRAM and the vision representations from IL pre-training are already high-quality —
fine-tuning them with RL noise risks degrading perceptual quality without benefit.

### The Reference Model

The reference model is a frozen copy of the StreamVLN model loaded independently from disk in
bfloat16 precision onto `cuda:1`. It serves as the baseline policy — what the model would do if
it had not been RL-trained at all.

Note: `deepcopy` cannot be used on 4-bit quantized models because the accelerate library attaches
GPU dispatch hooks to each parameter, and deep-copying them corrupts the memory layout. Instead,
the reference model is loaded fresh from the checkpoint files in plain bf16.

The reference model is used in two places:

1. **During rollout collection (Block 3):** For each action the policy takes, we immediately
   compute `ref_log_prob = log π_ref(action | state)` and cache it in the transition buffer.
   This avoids re-running the reference model during PPO optimization (which would require
   re-processing all visual inputs for every transition, every epoch).

2. **During PPO optimization (Block 4):** The cached `ref_log_prob` is used to compute the KL
   penalty term in the shaped reward (see Section 7).

With PEFT/LoRA and a single GPU, the reference policy can alternatively be computed by
temporarily disabling the LoRA adapters via `model.pretrained_model.disable_adapter()`. This
saves VRAM at the cost of extra computation per transition. In 2-GPU mode we use the explicit
reference model on cuda:1 for faster evaluation.

---

## 6. Online Rollout Collection in Habitat (Block 3)

### The Simulation Environment

Habitat is a 3D simulator that renders photorealistic Matterport3D scenes and exposes a
navigation API. At each step, Habitat:
- Returns an RGB image (resized/preprocessed for the vision tower)
- Returns a depth map (filtered, scaled to meters × 1000 to millimeters)
- Returns GPS coordinates (x, y relative to episode start)
- Returns compass heading (yaw angle)
- Returns metrics: `distance_to_goal`, `success`, `spl`, `collisions.is_collision`

The agent executes one of 4 discrete actions:
- `STOP (0)`: End the episode
- `FORWARD (1)`: Move forward 25 cm
- `TURN_LEFT (2)`: Rotate left 15 degrees
- `TURN_RIGHT (3)`: Rotate right 15 degrees

### Parallel Collection with VectorEnv

We run 4 Habitat environments in parallel worker processes using `VectorEnv`. This gives
approximately 4× speedup over sequential collection because the bottleneck is Habitat's physics
simulation and rendering (which runs in separate processes), not the model forward pass (which
runs on GPU).

In each iteration we collect 32 episodes total. With 4 parallel envs, this takes roughly 8
"rounds" of parallel stepping. The main process runs sequential GPU forward passes for each
active environment per step — the GPU work is serialized, but the environment physics all happen
in parallel.

### Observation Processing (Per Step)

For each environment at each step:

**RGB Processing:**
```
raw_rgb (H×W×3, uint8)
→ PIL.Image.fromarray(...).convert("RGB")
→ image_processor.preprocess(images=image, return_tensors="pt")["pixel_values"][0]
→ image_tensor: shape [3, crop_H, crop_W], float32
```

**Depth Processing:**
```
raw_depth (H×W×1, float32, range [0,1])
→ filter_depth(depth.reshape(H,W), blur_type=None)  # remove outliers
→ × (max_depth - min_depth) + min_depth              # rescale to metric range
→ × 1000                                              # convert to millimeters
→ PIL.Image.fromarray(depth.astype(uint16), mode="I;16")
→ .resize((target_W, target_H), NEAREST)
→ to_numpy_array(resized) / 1000.0                   # back to meters
→ depth_tensor: shape [crop_H, crop_W], float32
```

**Pose Processing:**
```
agent_state = env.sim.get_agent_state()
x, y = observations["gps"]                           # position relative to start
camera_yaw = observations["compass"][0]
height = agent_state.position[1] - initial_height

camera_position = [x, -y, camera_height + height]
tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)
                        @ axis_align_matrix            # 4×4 double precision
```

The axis_align_matrix is a fixed 4×4 tensor:
```
[[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
```
It reorders axes from Habitat's coordinate system to the convention expected by StreamVLN's pose
encoder. It is pre-computed once in `__init__` as `self._axis_align` to avoid recreating it on
every step.

### Streaming KV Cache Management

StreamVLN maintains a streaming KV cache per environment slot. The model's internal `reset(N)`
call allocates N cache slots (one per parallel environment). During rollout:

- Each environment uses its own slot index (0 to num_envs-1) for KV cache isolation.
- Every `num_frames=32` steps within an episode, `reset_for_env(env_idx)` is called to flush
  that environment's cache and inject compressed historical frames (slow memory update).
- After periodic evaluation runs (which call `reset(1)` internally), the cache is restored to
  `reset(num_envs)` so training can continue with all parallel env slots available.

### Logit Masking

The language model outputs logits over ~32,000 vocabulary tokens. For navigation, only 4 tokens
are valid. Before sampling, we apply a logit mask:

```python
mask = torch.full_like(last_logits, float("-inf"))    # block all tokens
mask[:, valid_token_ids] = 0.0                         # unblock the 4 action tokens
masked_logits = last_logits + mask
```

After applying temperature (T=0.6), a stochastic action is sampled via `torch.multinomial`.
The log-probability is computed using `F.log_softmax` (not `torch.log(F.softmax(...))`) for
numerical stability — `log(softmax(x))` can underflow to `-inf` for large negative logits, while
`log_softmax` uses the log-sum-exp trick to avoid this.

Temperature T=0.6 makes the distribution sharper than uniform but still stochastic — the model
favors the highest-probability action but still explores alternatives. T=1.0 would use the raw
distribution; T→0 would be greedy argmax.

### What Gets Stored Per Step

Each `StepData` object stores:
- `query_tensor`: input_ids (the tokenized conversation prompt)
- `response_tensor`: the chosen action token ID
- `state_images`, `state_depths`, `state_poses`, `state_intrinsics`: the multimodal inputs
  clipped to `buffer_max_views=1` (only the most recent view, to save CPU memory)
- `state_time_ids`: time indices for the streaming context
- `action_log_prob`: log π_policy(action | state) — used as old_logprob in PPO ratio
- `value`: V(state) from the critic — used in GAE
- `ref_log_prob`: log π_ref(action | state) — cached for KL penalty
- `reward`: the reward received at this step
- `action_idx`: the environment action index (0-3)
- `done`: whether the episode ended

---

## 7. The Reward System in Full Detail (Block 4)

The reward function is implemented in `VLNRewardFunction` and called once per step during
rollout collection. It maintains two pieces of state across steps:
- `_initial_distance_to_goal`: captured on the very first step of each episode
- `_prev_distance_to_goal`: the distance at the previous step

### 7.1 Per-Step Base Reward

```python
reward = step_reward   # = 0.0 by default
```

A constant per-step reward of 0.0 means each step has zero cost. This is intentional — we do not
penalize the agent for taking steps, because navigation inherently requires many steps. A
negative step reward would bias the agent toward shorter paths regardless of whether they reach
the goal.

### 7.2 Progress Shaping Reward

```python
if prev_distance_to_goal is not None:
    progress = prev_distance_to_goal - current_distance_to_goal
    reward += 0.1 * progress
prev_distance_to_goal = current_distance_to_goal
```

This is potential-based reward shaping. `progress` is positive when the agent moves closer to
the goal and negative when it moves away. The scale factor 0.1 is chosen so that a direct path
to a goal 10 meters away (100 steps × 0.1m/step × 0.1 scale) contributes about +1.0 total
shaping reward — roughly equal in magnitude to the terminal reward.

Why not use a larger scale? Too large a progress scale causes the agent to optimize purely for
DTG reduction, ignoring the need to eventually STOP at the goal. Too small and the signal is too
weak to overcome the noise in PPO updates.

The progress reward is only non-zero when `prev_distance_to_goal` is available (i.e., from step
2 onward), because on step 1 there is no previous distance to compare against.

### 7.3 Collision Penalty

```python
collisions = metrics.get("collisions", {})
is_collision = collisions.get("is_collision", False)
if is_collision:
    reward += -0.05
```

Habitat detects collisions when the agent tries to move into a wall or obstacle. The penalty
`-0.05` is small relative to the progress reward (a single step forward in a 10m path gives
`+0.01`) and relative to the terminal reward (+1 or -1). This means collisions are mildly
discouraged but do not dominate the reward signal. The agent learns to avoid obstacles as a
consequence of losing progress reward (it stops making progress when stuck against a wall) rather
than primarily from the collision penalty itself.

### 7.4 Early-STOP Penalty

```python
if action_idx == 0 and step < 5 and not metrics.get("success", False):
    reward += -0.5
```

This addresses the STOP-bias problem directly. If the agent outputs STOP within the first 5
steps of an episode (before it has had any chance to navigate meaningfully) and has not actually
reached the goal, it receives a penalty of -0.5.

The magnitude -0.5 is chosen to be strong enough to override the agent's IL-trained prior to
STOP immediately, but not so large that it causes catastrophically negative rewards that destabilize
early training. After step 5, STOP is allowed without penalty — the agent can legitimately decide
to stop anywhere after minimal exploration.

### 7.5 Terminal Reward (Continuous)

This is the most important reward component. It fires once at the end of each episode.

**First, compute normalised progress:**
```python
initial_dtg = _initial_distance_to_goal   # captured at episode start (step 0)
final_dtg = distance_to_goal              # distance at episode end

if initial_dtg > 0:
    norm_progress = max(0.0, min(1.0, 1.0 - final_dtg / initial_dtg))
else:
    norm_progress = 1.0 if success else 0.0
```

`norm_progress` is a number in [0, 1]:
- 0.0 means the agent ended at exactly the same distance from the goal as where it started
- 1.0 means the agent ended exactly at the goal (final_dtg = 0)
- 0.5 means the agent covered half the distance to the goal

**On success** (agent is within 3m of the goal):
```python
spl = float(metrics.get("spl", norm_progress))
reward += success_reward * max(spl, norm_progress)
        = 1.0 * max(spl, norm_progress)
```

SPL (Success weighted by Path Length) is computed by Habitat as:
```
SPL = (shortest_path_length / max(path_taken_length, shortest_path_length))
```
SPL = 1.0 for a perfectly efficient path, SPL < 1.0 for any detour. By multiplying the success
reward by SPL, we incentivize efficient navigation — getting to the goal via a short path earns
more than getting there via a long winding path. The `max(spl, norm_progress)` ensures the
terminal reward is never less than what norm_progress alone would give.

**On failure** (timeout or max steps exceeded):
```python
reward += failure_reward + (success_reward - failure_reward) * norm_progress
        = -1.0 + 2.0 * norm_progress
```

This is a linear interpolation between failure_reward (-1.0) and success_reward (+1.0):
- `norm_progress = 0.0` → reward = -1.0 (agent made zero progress, maximum penalty)
- `norm_progress = 0.5` → reward = 0.0 (agent got halfway, neutral)
- `norm_progress = 0.9` → reward = +0.8 (agent almost reached the goal, substantial partial credit)
- `norm_progress = 1.0` → reward = +1.0 (agent was at the goal but did not STOP in time)

The continuous terminal reward is critical for learning. With a binary ±1 terminal:
- An episode where the agent got to within 1m of the goal (but timed out) receives -1.0.
- An episode where the agent wandered randomly and ended 50m away also receives -1.0.
- The gradient sees no difference between these two outcomes — it cannot learn "closer is better".

With the continuous terminal, the first episode gets +0.98 and the second gets -0.90. The
difference in return (-1.88) gives the advantage function a clear signal that getting close to
the goal matters, even without successfully STOPping there.

### 7.6 Complete Reward Example

Consider an episode where the agent starts 10m from the goal and walks for 50 steps before
timing out, ending up 3m from the goal with 2 collisions:

```
Step 1:  reward = 0 + 0.1*(10.0 - 9.8) + 0     = +0.020  (moved 20cm closer)
Step 2:  reward = 0 + 0.1*(9.8 - 9.5) + 0      = +0.030  (moved 30cm closer)
...
Step 22: reward = 0 + 0.1*(5.5 - 5.6) + (-0.05) = -0.060  (hit a wall)
...
Step 50: reward = 0 + 0.1*(3.2 - 3.0) + 0       = +0.020  (terminal step)
         + terminal: -1.0 + 2.0*(1 - 3.0/10.0) = -1.0 + 1.4 = +0.4

Total reward ≈ +0.4 (progress) − 0.10 (2 collisions) + 0.4 (terminal) = ~+0.7
```

Compare to an agent that walked away from the goal and ended 15m away:
```
Total reward ≈ -0.5 (negative progress) + (-1.0 + 0.0) = -1.5
```

The difference in total return (+0.7 vs -1.5 = 2.2) gives GAE a strong signal to prefer
goal-directed movement.

---

## 8. Advantage Estimation with GAE-lambda

After collecting 32 episodes, all steps are flattened into a single buffer of N transitions
(typically N ≈ 32 × average_episode_length). GAE-lambda computes per-step advantages by
walking backward through this buffer.

### Why GAE?

Raw Monte Carlo advantages (`A_t = G_t - V(s_t)` where `G_t` is the full discounted return from
step t to episode end) have low bias but high variance — individual episodes are noisy, and the
advantage estimates are very unstable.

Using only one-step TD error (`A_t = r_t + γ V(s_{t+1}) - V(s_t)`) has low variance but high
bias — the value function is imperfect, so using it as a bootstrap target introduces systematic
errors.

GAE-lambda interpolates between these extremes via the lambda parameter (λ=0.95):

```
δ_t = r_t + γ × V(s_{t+1}) × (1 - done_t) - V(s_t)   # one-step TD error

A_t = δ_t + (γλ) × δ_{t+1} × (1 - done_t)
           + (γλ)² × δ_{t+2} × (1 - done_t) × (1 - done_{t+1})
           + ...
```

This is computed efficiently by iterating backward:
```python
lastgaelam = 0.0
for t in reversed(range(n_steps)):
    next_nonterminal = 1.0 - dones[t].float()
    if t == n_steps - 1:
        next_values = 0.0
    else:
        next_values = values[t+1] * next_nonterminal

    delta = rewards[t] + gamma * next_values - values[t]
    lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
    advantages[t] = lastgaelam

returns = advantages + values
```

The `next_nonterminal = 1 - dones[t]` is critical: it zeros out the bootstrap value at episode
boundaries. When `dones[t] = True`, step t is the last step of an episode. The next step belongs
to a completely different episode — bootstrapping across that boundary would contaminate the
advantage estimates. Zeroing the nonterminal flag prevents this.

With γ=0.99 and λ=0.95, the effective "lookback" is roughly `1/(1-γλ) ≈ 1/(0.0595) ≈ 17 steps`.
This means each advantage estimate looks about 17 steps into the future — enough to capture the
delayed effect of good navigation decisions but short enough to reduce variance.

### Advantage Normalization

After computing all advantages, they are normalized:
```python
adv_mean = advantages.mean()
adv_std = advantages.std(unbiased=False)
if adv_std > 1e-8:
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)
```

This ensures advantages have zero mean and unit variance across the buffer. Without normalization,
the scale of advantages grows with episode length and reward magnitude, causing the PPO loss to
have wildly varying scale across iterations. Normalization makes the policy gradient step size
consistent regardless of episode length or reward scale.

---

## 9. PPO Optimization — Step by Step

PPO runs 4 epochs over the collected transitions, processing them in random mini-batches of 8.
With gradient accumulation over 2 mini-batches, each optimizer step sees an effective batch of
16 transitions.

### 9.1 Re-evaluating the Policy

For each mini-batch of transitions, the current policy is run through the model to get fresh
log-probabilities and value estimates:

```python
logits, _, values = model(input_ids, images, depths, poses, intrinsics, time_ids, ...)
last_logits = logits[:, -1, :]                          # last token position
action_logits = last_logits[:, action_token_ids]        # restrict to 4 action tokens
action_logits /= rollout_temperature                    # apply temperature
log_probs = F.log_softmax(action_logits, dim=-1)
action_logprob = log_probs.gather(-1, action_index)     # log-prob of the chosen action
value = values[:, -1]                                   # critic at last position
entropy = -(softmax(action_logits) * log_probs).sum(-1) # over 4-action distribution
```

The log-probability is computed over the **constrained 4-action vocabulary only**, not the full
32,000-token vocabulary. This is essential for PPO stability: the ratio `π_new / π_old` must be
computed in the same action space that was used during rollout. Using the full vocabulary would
introduce a constant denominator shift that distorts the ratio.

### 9.2 Policy Loss (Clipped Surrogate Objective)

```python
ratio = torch.exp(logprobs - old_logprobs)   # = π_new(a|s) / π_old(a|s)

pg_loss1 = -advantages * ratio
pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
pg_loss = torch.max(pg_loss1, pg_loss2).mean()
```

The ratio measures how much the policy has changed for each action. If `ratio > 1.2` for a
positive-advantage action, the policy has been pushed too far toward that action — clipping
prevents further optimization in that direction. If `ratio < 0.8` for a negative-advantage
action (we're trying to push away from a bad action), clipping prevents too-aggressive
suppression.

The clipping range ε=0.2 means the policy is allowed to change by at most 20% (in probability
ratio terms) per PPO update. Without clipping, a single large gradient step could collapse the
policy onto a single action (policy collapse) or make it completely random (policy explosion).

### 9.3 Value Loss (Clipped MSE)

```python
vpredclipped = clip_by_value(
    vpreds,
    old_values - 0.2,
    old_values + 0.2
)
vf_loss1 = (vpreds - returns) ** 2
vf_loss2 = (vpredclipped - returns) ** 2
vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
```

The value function is trained to predict the GAE returns. The clipping prevents the value
function from updating too far from its rollout-time estimate in a single PPO epoch, keeping the
GAE targets (which were computed with the old value function) from becoming stale.

### 9.4 KL Penalty in the Reward

The KL penalty is applied during reward computation (Section 7), not as a loss term:

```python
kls = old_logprobs - ref_logprobs          # log π_old - log π_ref per step
non_score_rewards = -kl_coef * kls         # negative KL divergence
rewards = env_rewards + non_score_rewards  # shaped reward = env + KL penalty
```

This penalizes the policy for drifting from the reference model even before PPO sees the
transition. A positive KL (policy more confident than reference on the chosen action) reduces the
reward; a negative KL (policy less confident than reference) slightly increases it. The KL
coefficient adapts via the `AdaptiveKLController`:

```python
proportional_error = clip(current_kl / target_kl - 1, -0.2, 0.2)
kl_coef *= 1 + proportional_error * n_steps / horizon
```

If the current KL is above target (6.0), `kl_coef` increases (stronger penalty). If below target,
`kl_coef` decreases. This self-regulates to maintain the policy near — but not identical to —
the reference model.

### 9.5 Entropy Bonus

```python
entropy_bonus = entropies.mean()   # mean over mini-batch
loss = loss - entropy_coef * entropy_bonus  # = loss - 0.01 * H(π)
```

Entropy H(π) = -Σ p(a) log p(a) over the 4-action distribution. Maximizing entropy pushes the
distribution toward uniform, discouraging the policy from collapsing onto a single action.
The coefficient 0.01 is small — entropy is a regularizer, not the primary objective. Without it,
the policy quickly becomes deterministic (always picking the same action) and loses the ability
to explore different strategies.

### 9.6 Total Loss and Gradient Flow

```python
loss = pg_loss + vf_coef * vf_loss - entropy_coef * entropy_bonus
     = pg_loss + 0.5 * vf_loss - 0.01 * entropy_bonus

scaled_loss = loss / gradient_accumulation_steps   # / 2
scaled_loss.backward()
```

Gradients flow through:
- `pg_loss` → LoRA adapter matrices A and B (language model layers)
- `vf_loss` → Value Head linear layer + LayerNorm
- Both → shared frozen transformer weights (but frozen weights have `requires_grad=False`,
  so gradients are computed through them but not applied)

After accumulating gradients for 2 mini-batches:
```python
torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
optimizer.step()
optimizer.zero_grad()
```

Gradient clipping to 0.5 prevents individual large gradients (common in early training when the
value head is random) from destabilizing the LoRA adapters.

### 9.7 OOM-Safe Minibatch Skipping

During PPO optimization, if a CUDA out-of-memory error occurs on a mini-batch (which can happen
if a particular transition has unusually many visual tokens), the mini-batch is skipped:

```python
except (torch.OutOfMemoryError, RuntimeError) as e:
    if "out of memory" in str(e).lower() or "CUBLAS" in str(e) or "cudaMalloc" in str(e):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        gc.collect()
        continue   # skip this mini-batch, continue with the next
    raise          # re-raise non-OOM RuntimeErrors
```

The number of skipped mini-batches is logged per iteration. If many mini-batches are skipped,
`ppo_max_views` should be reduced (currently 1, meaning only the most recent visual frame is
used during PPO re-evaluation).

---

## 10. The Full Training Loop

```
for iteration in range(1000):

    ┌── Block 3: Rollout Collection ──────────────────────────────────────┐
    │  model.eval()                                                        │
    │  Collect 32 episodes using 4 parallel Habitat VectorEnv instances   │
    │  Each step:                                                          │
    │    1. Process RGB, depth, pose, intrinsics                           │
    │    2. model(inputs) → logits, value                                  │
    │    3. Apply logit mask → sample action (T=0.6)                       │
    │    4. ref_model(inputs) → ref_log_prob                               │
    │    5. env.step(action) → next_obs, done, metrics                     │
    │    6. reward_fn.step_reward(metrics, step, done, action)             │
    │    7. Store StepData(inputs, log_prob, value, ref_lp, reward, done)  │
    │  After episode: store total_reward, success, spl, dtg                │
    └──────────────────────────────────────────────────────────────────────┘
             ↓
    ┌── Convert to Transition Batch ──────────────────────────────────────┐
    │  trajectories_to_transition_batch(trajectories)                      │
    │  Forces last step of each episode to done=True (GAE boundary)        │
    └──────────────────────────────────────────────────────────────────────┘
             ↓
    ┌── Block 4: PPO Optimization ────────────────────────────────────────┐
    │  model.train()                                                        │
    │                                                                       │
    │  1. old_logprobs, old_values, env_rewards, dones from buffer          │
    │  2. ref_logprobs from cached ref_log_prob (no ref-model forward)      │
    │  3. rewards = env_rewards - kl_coef * (old_logprobs - ref_logprobs)  │
    │  4. values, advantages, returns = compute_advantages(GAE-λ)           │
    │                                                                       │
    │  for epoch in range(4):                                               │
    │      shuffle transitions                                              │
    │      for minibatch in chunks(transitions, size=8):                    │
    │          logprobs, vpreds, entropies = model(minibatch_inputs)        │
    │          loss = pg_loss + 0.5*vf_loss - 0.01*entropy                 │
    │          (loss / 2).backward()                                        │
    │          if accum_count % 2 == 0:                                     │
    │              clip_grad_norm_(params, 0.5)                             │
    │              optimizer.step(); optimizer.zero_grad()                  │
    │                                                                       │
    │  kl_ctl.update(mean_kl, n_steps)                                      │
    └──────────────────────────────────────────────────────────────────────┘
             ↓
    lr_scheduler.step()          # cosine decay with 5% linear warm-up
    log_stats(stats)             # rollout SR, PPO losses, KL, entropy, etc.

    if iteration % 5 == 0:
        save_checkpoint(...)     # LoRA adapters + value head + optimizer state

    if iteration % 25 == 0:
        run_evaluation(...)      # val_unseen split, track best model
        model.reset(num_envs)    # restore KV cache slots after eval
```

---

## 11. Key Hyperparameters and Their Roles

| Parameter | Value | Why This Value |
|-----------|-------|----------------|
| `lora_r` | 64 | High rank = more expressive adapters. r=64 allows the policy to make significant behavioral changes while keeping parameters manageable. |
| `lora_alpha` | 64 | alpha=r gives scaling factor 1.0. Lower alpha (e.g. 16) gives 0.25 scaling — too conservative, slows learning. |
| `episodes_per_update` | 32 | More episodes = more diverse transitions per PPO step = more stable gradient estimates. 32 balances diversity with iteration speed. |
| `ppo_epochs` | 4 | Re-using each rollout for 4 gradient epochs improves sample efficiency. Beyond 4-6, the old_logprob / new_logprob ratio diverges too much. |
| `mini_batch_size` | 8 | Larger batches = more stable gradients. 8 transitions fit in GPU VRAM with `ppo_max_views=1`. |
| `gamma` | 0.99 | High discount for long-horizon tasks (50-500 steps). γ=0.99 means a reward 100 steps away is worth 0.99^100 ≈ 0.37 today. |
| `gae_lambda` | 0.95 | λ=0.95 gives a good bias-variance tradeoff. Lower λ → lower variance but higher bias (relies more on the imperfect value function). |
| `cliprange` | 0.2 | Standard PPO clip. Allows 20% policy change per update. |
| `vf_coef` | 0.5 | Value loss coefficient. Balances policy and critic learning — critic should not dominate. |
| `entropy_coef` | 0.01 | Small entropy bonus keeps the policy from collapsing to deterministic behavior too early. |
| `init_kl_coef` | 0.1 | Initial KL penalty weight. Low enough to allow the policy to learn, high enough to prevent immediate catastrophic forgetting. |
| `target_kl` | 6.0 | Target KL between policy and reference. KL=6 allows significant policy change while staying in the reference model's neighborhood. |
| `temperature` | 0.6 | Sharper-than-uniform sampling. Encourages the policy to exploit its best guess while still exploring. |
| `progress_scale` | 0.1 | Dense shaping magnitude. Scaled so total shaping ≈ terminal magnitude over a typical episode. |
| `early_stop_penalty` | -0.5 | Strong enough to override STOP-bias, small enough not to destabilize early training. |
| `warmup_ratio` | 0.05 | 5% of 1000 iterations = 50 warm-up steps. Ramps LR from 0 to 1e-5 while the randomly-initialized value head stabilizes. |

---

## 12. GPU Memory Strategy

The model is trained on a single A6000 (49 GB VRAM) with the reference model on a second GPU.

**Policy model (cuda:0):**
- 4-bit NF4 quantization (QLoRA): ~2.3 GB for 4B parameters (vs ~8 GB in bf16)
- LoRA adapters: ~0.6 GB
- Value head: negligible
- Gradient + optimizer states for LoRA: ~2.4 GB (Adam stores 2 moments per trainable param)
- Activation memory during PPO forward/backward: ~6-10 GB per mini-batch

**Reference model (cuda:1):**
- bf16 (no quantization): ~8 GB
- No gradients, no optimizer state
- Reference log-probs are cached during rollout, so the ref-model is not called during PPO

**Rollout buffer (CPU RAM):**
- 32 episodes × ~100 steps × 1 view × (image + depth + pose + intrinsics) per step
- With `buffer_max_views=1`, each step stores only the most recent frame
- Approximate: 32 × 100 × (3×336×336×4 + 336×336×4 + 4×4×8 + 4×4×4) bytes ≈ 15 GB

**`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** prevents CUDA from reserving large
fixed-size memory blocks. Without this, the allocator may fail to find a contiguous block for a
large allocation even when total free memory exceeds the request. Expandable segments allow the
allocator to grow allocations gradually, reducing fragmentation.

---

## 13. Failure Modes and Mitigations

| Failure | Symptom | Mitigation |
|---------|---------|------------|
| STOP-bias | Agent stops on step 1-3, episodes of length 1-3 | Early-STOP penalty (-0.5 within first 5 steps) |
| Policy collapse | All actions converge to one (e.g. always FORWARD) | Entropy bonus (coef=0.01) + KL penalty prevents collapsing too fast |
| Catastrophic forgetting | Navigation quality degrades, model loses language understanding | KL penalty against reference model (frozen IL weights) |
| Value head divergence | Value loss explodes, advantage estimates become huge | Gaussian init (std=0.2), gradient clipping (0.5), clipped value loss |
| KV-cache slot mismatch | IndexError after evaluation run | `finally: model.reset(_cache_slots)` restores correct number of slots |
| CUDA OOM in PPO | RuntimeError during mini-batch forward | OOM-safe skip: zero gradients, empty cache, continue to next mini-batch |
| Cross-episode GAE leakage | Advantages from one episode contaminate the next | `next_nonterminal = 1 - dones[t]` zeros bootstrap at episode boundaries |
| Log-prob underflow | `log(softmax(x))` = -inf for large negative logits | Use `F.log_softmax` instead (numerically stable log-sum-exp) |
| Vision tower on meta device | 4-bit loading silently leaves vision tower uninitialized | Explicit reload of SigLip weights + StreamVLN fine-tuned vision tower after quantized loading |

---

*This document reflects the MindDrive RL implementation as of the current codebase state.*
*All code references are to `StreamVLN/streamvln/rl/`.*
