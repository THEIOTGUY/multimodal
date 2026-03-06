"""
RolloutCollector — Online rollout collection in Habitat simulator.

Implements Block 3 of the MindDrive architecture diagram:

    ┌──────────────────────────────────────────────────────────┐
    │  3. Online Rollouts (Simulation Loop)                   │
    │                                                         │
    │  Habitat Simulator ──► Observation (Video Frames)       │
    │          ├──► Action Generation (LLM + Logit Mask)      │
    │          │         ├──► Selected Action Token            │
    │          │         └──► Buffer Storage                   │
    │          └──► Environment Step (Execute Command)         │
    │                    └──► Critic Value Prediction          │
    └──────────────────────────────────────────────────────────┘

Each rollout episode collects:
  - Per-step observations (RGB, depth, pose)
  - Per-step action log-probabilities
  - Per-step critic value predictions
  - Per-step rewards (from VLNRewardFunction)
  - Query/response token tensors for PPO

The collector mirrors the logic in ``VLNEvaluator.eval_action()`` exactly
but additionally captures the data needed for PPO training.
"""

from __future__ import annotations

import copy
import logging
import random
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from PIL import Image

import torch
import torch.nn.functional as F

from .reward import VLNRewardFunction, RewardConfig

logger = logging.getLogger(__name__)

# Import StreamVLN utilities — these are available when running from the
# StreamVLN/streamvln/ directory (added to sys.path in train_rl.py).
try:
    from utils.utils import (
        IMAGE_TOKEN_INDEX,
        MEMORY_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_MEMORY_TOKEN,
        DEFAULT_VIDEO_TOKEN,
        dict_to_cuda,
    )
except ImportError:
    # Fallback constants matching StreamVLN
    IMAGE_TOKEN_INDEX = -200
    MEMORY_TOKEN_INDEX = -300
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_MEMORY_TOKEN = "<memory>"
    DEFAULT_VIDEO_TOKEN = "<video>"

    def dict_to_cuda(data, device):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)
        return data


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class RolloutConfig:
    """Configuration for the rollout collector."""
    max_episode_steps: int = 500
    num_frames: int = 32
    num_future_steps: int = 1
    num_history: int = 8
    temperature: float = 1.0

    # Logit mask: restrict generation to valid action tokens only
    # (only forward, left, right, stop — from diagram Block 3)
    use_logit_mask: bool = True

    # Maximum visual views stored per transition in the rollout buffer.
    # Clips multimodal tensors to the last N views before CPU storage,
    # preventing CPU OOM on long episodes (PPO re-evaluation only needs
    # the most recent views anyway — see ppo_max_views in trainer).
    buffer_max_views: int = 1


@dataclass
class StepData:
    """Data collected at a single timestep during a rollout."""
    # Token tensors
    query_tensor: Optional[torch.Tensor] = None
    response_tensor: Optional[torch.Tensor] = None

    # Multimodal state used to produce the action at this step
    state_images: Optional[torch.Tensor] = None
    state_depths: Optional[torch.Tensor] = None
    state_poses: Optional[torch.Tensor] = None
    state_intrinsics: Optional[torch.Tensor] = None
    state_time_ids: Optional[List[int]] = None
    state_task_type: int = 0

    # Probabilities
    action_log_prob: float = 0.0

    # Critic
    value: float = 0.0

    # Reference model log-probability (cached during rollout to skip
    # expensive re-evaluation during PPO optimization)
    ref_log_prob: Optional[float] = None

    # Reward
    reward: float = 0.0

    # Environment info
    action_idx: int = 0
    done: bool = False
    metrics: Optional[Dict] = None


@dataclass
class EpisodeTrajectory:
    """Complete trajectory from a single episode rollout."""
    steps: List[StepData] = field(default_factory=list)
    episode_id: str = ""
    scene_id: str = ""
    instruction: str = ""
    total_reward: float = 0.0
    success: bool = False
    spl: float = 0.0
    distance_to_goal: float = float("inf")
    num_steps: int = 0


# ======================================================================
# Rollout Collector
# ======================================================================

class RolloutCollector:
    """
    Collects rollout trajectories by running the actor-critic model in
    the Habitat simulator.

    Mirrors ``VLNEvaluator.eval_action()`` exactly but collects per-step
    log-probs, values, and rewards for PPO training.

    Parameters
    ----------
    model : StreamVLNWithValueHead
        The actor-critic model (value head attached).
    tokenizer : PreTrainedTokenizer
        Tokenizer matching the model.
    image_processor : object
        Vision tower image processor (from model.get_vision_tower().image_processor).
    reward_fn : VLNRewardFunction
        Reward function instance.
    config : RolloutConfig
        Rollout hyperparameters.
    device : torch.device
        CUDA device.
    """

    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        reward_fn: VLNRewardFunction,
        config: Optional[RolloutConfig] = None,
        device: Optional[torch.device] = None,
        ref_model=None,
        ref_device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.reward_fn = reward_fn
        self.config = config or RolloutConfig()
        self.device = device or torch.device("cuda")
        self.ref_model = ref_model
        self.ref_device = ref_device or self.device

        # Action mapping (matching VLNEvaluator exactly)
        self.actions2idx = OrderedDict({
            "STOP": [0],
            "↑": [1],
            "←": [2],
            "→": [3],
        })
        self.idx2action = {0: "STOP", 1: "↑", 2: "←", 3: "→"}

        # Tokenize action tokens for logit masking
        self._action_token_ids = {}
        for action_name in self.actions2idx.keys():
            ids = self.tokenizer.encode(action_name, add_special_tokens=False)
            self._action_token_ids[action_name] = ids[0] if ids else None

        self.valid_token_ids = [
            tid for tid in self._action_token_ids.values()
            if tid is not None
        ]

        # Conversation template (matching VLNEvaluator)
        self.prompt_template = (
            "<video>\nYou are an autonomous navigation assistant. "
            "Your task is to <instruction>. Devise an action sequence "
            "to follow the instruction using the four actions: "
            "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, "
            "MOVE FORWARD (↑) by 25 centimeters, or STOP."
        )

        self.conjunctions = [
            "you can see ",
            "in front of you is ",
            "there is ",
            "you can spot ",
            "you are toward the ",
            "ahead of you is ",
            "in your sight is ",
        ]

        # Pre-build the tokenizer used for RL rollouts ONCE (avoid
        # deepcopy + add_tokens on every single step).
        self._rl_tokenizer = copy.deepcopy(tokenizer)
        self._rl_tokenizer.add_tokens(["<image>"], special_tokens=True)
        self._rl_tokenizer.add_tokens(["<memory>"], special_tokens=True)
        self._rl_image_token_index = self._rl_tokenizer.convert_tokens_to_ids("<image>")
        self._rl_memory_token_index = self._rl_tokenizer.convert_tokens_to_ids("<memory>")
        self._rl_tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )

        # Pre-built constant; avoids recreating this tensor on every rollout step.
        self._axis_align = torch.tensor(
            [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
        ).double()

    # ------------------------------------------------------------------
    # Main rollout method
    # ------------------------------------------------------------------

    def collect_episode(
        self,
        env,
        episode,
        env_idx: int = 0,
        sim_sensors_config=None,
    ) -> EpisodeTrajectory:
        """
        Run a single episode in the Habitat environment, collecting
        trajectory data for PPO training.

        Parameters
        ----------
        env : habitat.Env
            Initialized Habitat environment.
        episode : habitat.Episode
            The episode to run.
        env_idx : int
            Environment index for streaming cache management.
        sim_sensors_config : OmegaConf
            Habitat sensor config for depth/intrinsic processing.

        Returns
        -------
        EpisodeTrajectory
            Complete trajectory with per-step data.
        """
        from depth_camera_filtering import filter_depth
        from transformers.image_utils import to_numpy_array

        cfg = self.config
        self.reward_fn.reset()
        self.model.reset_for_env(env_idx)

        # Grab sensor config
        if sim_sensors_config is None:
            # Habitat config layout differs across versions:
            # - older: env._config.habitat.simulator...
            # - newer: env._config.simulator...
            env_cfg = env._config
            if hasattr(env_cfg, "habitat"):
                sim_sensors_config = (
                    env_cfg.habitat.simulator.agents.main_agent.sim_sensors
                )
            else:
                sim_sensors_config = (
                    env_cfg.simulator.agents.main_agent.sim_sensors
                )

        camera_height = sim_sensors_config.rgb_sensor.position[1]
        min_depth = sim_sensors_config.depth_sensor.min_depth
        max_depth = sim_sensors_config.depth_sensor.max_depth

        intrinsic_matrix = self._get_intrinsic_matrix(sim_sensors_config.rgb_sensor)

        # Set up episode
        env.current_episode = episode
        observations = env.reset()
        instruction = episode.instruction.instruction_text

        trajectory = EpisodeTrajectory(
            episode_id=str(episode.episode_id),
            scene_id=episode.scene_id.split("/")[-2] if "/" in episode.scene_id else episode.scene_id,
            instruction=instruction,
        )

        # Per-episode accumulators (matching VLNEvaluator)
        rgb_list = []
        depth_list = []
        pose_list = []
        intrinsic_list = []
        time_ids = []
        step_id = 0

        initial_height = env.sim.get_agent_state().position[1]

        self.model.eval()
        while not env.episode_over and step_id < cfg.max_episode_steps:
            time_ids.append(step_id)

            # ── 1. Process observation (matching VLNEvaluator exactly) ──
            rgb = observations["rgb"]
            depth = observations["depth"]
            x, y = observations["gps"]
            camera_yaw = observations["compass"][0]

            # Depth processing
            depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
            depth = depth * (max_depth - min_depth) + min_depth
            depth = depth * 1000  # mm

            # Camera pose
            agent_state = env.sim.get_agent_state()
            height = agent_state.position[1] - initial_height
            camera_position = np.array([x, -y, camera_height + height])
            tf_camera_to_episodic = self._xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

            # Process RGB image (matching VLNEvaluator)
            image = Image.fromarray(rgb).convert("RGB")
            image_size = image.size
            image_tensor = self.image_processor.preprocess(
                images=image, return_tensors="pt"
            )["pixel_values"][0]

            # Process depth image
            target_height = self.image_processor.crop_size["height"]
            target_width = self.image_processor.crop_size["width"]
            depth_image = Image.fromarray(depth.astype(np.uint16), mode="I;16")
            resized_depth = depth_image.resize((target_width, target_height), Image.NEAREST)
            depth_array = to_numpy_array(resized_depth) / 1000.0
            resize_shape = (target_width, target_height)

            # Intrinsic
            intrinsic = self._preprocess_intrinsic(
                intrinsic_matrix, image_size, resize_shape
            )
            intrinsic_tensor = torch.from_numpy(intrinsic).float()

            rgb_list.append(image_tensor)
            depth_list.append(torch.from_numpy(depth_array).float())
            pose_list.append(
                torch.from_numpy(tf_camera_to_episodic) @ self._axis_align
            )
            intrinsic_list.append(intrinsic_tensor)

            # ── 2. Build model input (matching VLNEvaluator) ──
            sources = self._build_conversation(
                instruction, step_id, output_ids_exist=False
            )
            add_system = True
            input_ids, _ = self._preprocess_qwen(
                [sources], self.tokenizer, has_image=True, add_system=add_system
            )

            images = rgb_list[-1:]
            depths = depth_list[-1:]
            poses = pose_list[-1:]
            intrinsics = intrinsic_list[-1:]

            # History frames at window boundaries
            if step_id != 0 and step_id % cfg.num_frames == 0:
                if cfg.num_history is None:
                    history_ids = slice(0, time_ids[0], cfg.num_future_steps)
                else:
                    history_ids = slice(
                        0, time_ids[0], max(1, time_ids[0] // cfg.num_history)
                    )
                images = rgb_list[history_ids] + images
                depths = depth_list[history_ids] + depths
                poses = pose_list[history_ids] + poses
                intrinsics = intrinsic_list[history_ids] + intrinsics

            input_dict = {
                "images": torch.stack(images).unsqueeze(0),
                "depths": torch.stack(depths).unsqueeze(0),
                "poses": torch.stack(poses).unsqueeze(0),
                "intrinsics": torch.stack(intrinsics).unsqueeze(0),
                "input_ids": input_ids,
                "time_ids": [time_ids],
                "task_type": [0],
            }
            input_dict = dict_to_cuda(input_dict, self.device)
            for key in ["images", "depths", "poses", "intrinsics"]:
                if key in input_dict and isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].to(torch.bfloat16)

            # Clip multimodal inputs to buffer_max_views BEFORE forward so
            # that old_logprob matches what PPO re-evaluation will see.
            bmax = cfg.buffer_max_views
            if bmax > 0:
                for key in ("images", "depths", "poses", "intrinsics"):
                    if input_dict[key].shape[1] > bmax:
                        input_dict[key] = input_dict[key][:, -bmax:]
                cur_tids = input_dict["time_ids"][0]
                if len(cur_tids) > bmax:
                    input_dict["time_ids"] = [cur_tids[-bmax:]]

            # ── 3. Get logits + value from actor-critic ──
            with torch.no_grad():
                action_data = self._generate_action_with_value(input_dict, step_id=step_id)

            # ── 3b. Cache reference model log-prob ──
            _act_tid = action_data["action_token_id"]
            _act_tid_int = int(_act_tid.item()) if isinstance(_act_tid, torch.Tensor) else int(_act_tid)
            ref_lp = self._compute_ref_logprob(input_dict, _act_tid_int)

            # ── 4. Execute action in environment ──
            action_idx = action_data["action_idx"]
            observations = env.step(action_idx)
            done = env.episode_over

            # ── 5. Compute reward ──
            metrics = env.get_metrics()
            reward = self.reward_fn.step_reward(metrics, step_id, done, action_idx=action_idx)

            # ── 6. Store step data (views already clipped above) ──
            step_data = StepData(
                query_tensor=input_ids.cpu(),
                response_tensor=action_data["action_token_id"],
                state_images=input_dict["images"].detach().cpu(),
                state_depths=input_dict["depths"].detach().cpu(),
                state_poses=input_dict["poses"].detach().cpu(),
                state_intrinsics=input_dict["intrinsics"].detach().cpu(),
                state_time_ids=list(input_dict["time_ids"][0]),
                state_task_type=int(input_dict["task_type"][0]),
                action_log_prob=action_data["log_prob"],
                value=action_data["value"],
                ref_log_prob=ref_lp,
                reward=reward,
                action_idx=action_idx,
                done=done,
                metrics=metrics,
            )
            trajectory.steps.append(step_data)

            step_id += 1

            # ── 7. Reset streaming cache at window boundaries ──
            if step_id % cfg.num_frames == 0:
                self.model.reset_for_env(env_idx)
                time_ids = []

        # ── Episode summary ──
        final_metrics = env.get_metrics()
        trajectory.total_reward = sum(s.reward for s in trajectory.steps)
        trajectory.success = final_metrics.get("success", False)
        trajectory.spl = final_metrics.get("spl", 0.0)
        trajectory.distance_to_goal = final_metrics.get(
            "distance_to_goal", float("inf")
        )
        trajectory.num_steps = step_id

        return trajectory

    # ------------------------------------------------------------------
    # Parallel rollout collection using Habitat VectorEnv (N envs)
    # ------------------------------------------------------------------

    def collect_batch_parallel(
        self,
        vec_env,
        episode_ids: List[str],
        target_n: int,
        sim_sensors_config,
    ) -> List[EpisodeTrajectory]:
        """
        Collect target_n episode trajectories using N parallel Habitat envs.

        All N environments step simultaneously; model forwards are sequential
        on cuda:0.  Typical speedup: ≈ N× over sequential collection because
        Habitat sim.step() (the bottleneck) runs in parallel processes.

        Parameters
        ----------
        vec_env : habitat.core.vector_env.VectorEnv
            Pre-built VectorEnv of N ``VLNEnvForRL`` instances.
        episode_ids : list[str]
            Pool of episode IDs to draw from.
        target_n : int
            Number of complete trajectories to return.
        sim_sensors_config : OmegaConf
            Habitat sensor config (same as used in collect_episode).
        """
        from depth_camera_filtering import filter_depth
        from transformers.image_utils import to_numpy_array

        cfg = self.config
        num_envs = vec_env.num_envs

        # ── Camera / depth constants ──
        camera_height  = sim_sensors_config.rgb_sensor.position[1]
        min_depth      = sim_sensors_config.depth_sensor.min_depth
        max_depth      = sim_sensors_config.depth_sensor.max_depth
        intrinsic_mat  = self._get_intrinsic_matrix(sim_sensors_config.rgb_sensor)
        target_w       = self.image_processor.crop_size["width"]
        target_h       = self.image_processor.crop_size["height"]
        resize_shape   = (target_w, target_h)
        # ── Per-env mutable state ──
        class _ES:
            """Lightweight per-environment state bag."""
            __slots__ = (
                "traj", "rgb", "depth", "pose", "intr",
                "tids", "step_id", "init_h", "instruction",
                "reward_fn", "collecting", "obs",
            )
            def __init__(self, reward_config):
                self.traj        = None
                self.rgb         = []
                self.depth       = []
                self.pose        = []
                self.intr        = []
                self.tids        = []
                self.step_id     = 0
                self.init_h      = 0.0
                self.instruction = ""
                self.reward_fn   = VLNRewardFunction(reward_config)
                self.collecting  = False   # True when actively recording
                self.obs         = None

        es        = [_ES(self.reward_fn.config) for _ in range(num_envs)]
        completed = []
        ep_queue  = list(episode_ids)

        # ── Helper: assign and reset env i with the next episode ──
        def _start(i: int) -> bool:
            """Return True on success, False if no episodes left or env fails."""
            attempts = 0
            while ep_queue and attempts < 5:
                ep_id = ep_queue.pop(0)
                attempts += 1
                try:
                    # call_at uses kwargs dict, not positional list
                    vec_env.call_at(i, "set_episode_by_id", {"episode_id": ep_id})
                    obs_result = vec_env.reset_at(i)          # [obs]
                    es[i].obs        = obs_result[0]
                    es[i].init_h     = float(vec_env.call_at(i, "get_agent_height"))
                    es[i].instruction = vec_env.call_at(i, "get_instruction")
                    es[i].traj       = EpisodeTrajectory(
                        episode_id=ep_id, instruction=es[i].instruction
                    )
                    es[i].rgb        = []
                    es[i].depth      = []
                    es[i].pose       = []
                    es[i].intr       = []
                    es[i].tids       = []
                    es[i].step_id    = 0
                    es[i].reward_fn.reset()
                    self.model.reset_for_env(i)
                    es[i].collecting = True
                    return True
                except AssertionError as exc:
                    msg = str(exc)
                    if (
                        "Likely an invalid scene name" in msg
                        or "Missing (at least) one of scene dataset" in msg
                    ):
                        logger.warning(
                            f"[env {i}] missing scene, skipping ep {ep_id}"
                        )
                        continue
                    raise
                except Exception as exc:
                    logger.warning(
                        f"[env {i}] episode {ep_id} failed to start: {exc}"
                    )
                    continue
            es[i].collecting = False
            return False

        # Initialise all envs
        for i in range(num_envs):
            _start(i)

        # ── Main parallel step loop ──
        self.model.eval()
        while len(completed) < target_n:
            active = [i for i in range(num_envs) if es[i].collecting]
            if not active:
                break

            # Batch-fetch agent heights for all envs in one parallel IPC call
            all_heights = vec_env.call(["get_agent_height"] * num_envs)

            # ── Sequential model forward passes ──
            action_datas = {}          # env_idx → (action_data, input_dict, input_ids, tids_snap)
            for i in active:
                s   = es[i]
                obs = s.obs

                x, y       = obs["gps"]
                camera_yaw = obs["compass"][0]

                # Depth processing
                raw_depth = obs["depth"]
                filt_d = filter_depth(
                    raw_depth.reshape(raw_depth.shape[:2]), blur_type=None
                )
                filt_d = filt_d * (max_depth - min_depth) + min_depth
                filt_d = filt_d * 1000  # mm

                # Camera pose
                height          = all_heights[i] - s.init_h
                cam_pos         = np.array([x, -y, camera_height + height])
                tf_cam          = self._xyz_yaw_to_tf_matrix(cam_pos, camera_yaw)

                # RGB tensor
                image       = Image.fromarray(obs["rgb"]).convert("RGB")
                image_size  = image.size
                img_t       = self.image_processor.preprocess(
                    images=image, return_tensors="pt"
                )["pixel_values"][0]

                # Depth tensor
                dep_img  = Image.fromarray(filt_d.astype(np.uint16), mode="I;16")
                dep_res  = dep_img.resize((target_w, target_h), Image.NEAREST)
                dep_arr  = to_numpy_array(dep_res) / 1000.0
                dep_t    = torch.from_numpy(dep_arr).float()

                # Intrinsic tensor
                intr_np  = self._preprocess_intrinsic(intrinsic_mat, image_size, resize_shape)
                intr_t   = torch.from_numpy(intr_np).float()

                s.rgb.append(img_t)
                s.depth.append(dep_t)
                s.pose.append(torch.from_numpy(tf_cam) @ self._axis_align)
                s.intr.append(intr_t)
                s.tids.append(s.step_id)

                # Build conversation input (mirrors collect_episode exactly)
                sources   = self._build_conversation(
                    s.instruction, s.step_id, output_ids_exist=False
                )
                input_ids, _ = self._preprocess_qwen(
                    [sources], self.tokenizer, has_image=True, add_system=True
                )

                imgs   = s.rgb[-1:]
                deps   = s.depth[-1:]
                poses  = s.pose[-1:]
                intrs  = s.intr[-1:]

                if s.step_id != 0 and s.step_id % cfg.num_frames == 0:
                    if cfg.num_history is None:
                        hist = slice(0, s.tids[0], cfg.num_future_steps)
                    else:
                        hist = slice(
                            0, s.tids[0], max(1, s.tids[0] // cfg.num_history)
                        )
                    imgs  = s.rgb[hist]  + imgs
                    deps  = s.depth[hist] + deps
                    poses = s.pose[hist] + poses
                    intrs = s.intr[hist] + intrs

                inp = {
                    "images":     torch.stack(imgs).unsqueeze(0),
                    "depths":     torch.stack(deps).unsqueeze(0),
                    "poses":      torch.stack(poses).unsqueeze(0),
                    "intrinsics": torch.stack(intrs).unsqueeze(0),
                    "input_ids":  input_ids,
                    "time_ids":   [s.tids],
                    "task_type":  [0],
                }
                inp = dict_to_cuda(inp, self.device)
                for key in ["images", "depths", "poses", "intrinsics"]:
                    if isinstance(inp.get(key), torch.Tensor):
                        inp[key] = inp[key].to(torch.bfloat16)

                # Clip multimodal inputs to buffer_max_views BEFORE forward
                # so that old_logprob matches PPO re-evaluation.
                bmax = cfg.buffer_max_views
                if bmax > 0:
                    for key in ("images", "depths", "poses", "intrinsics"):
                        if inp[key].shape[1] > bmax:
                            inp[key] = inp[key][:, -bmax:]
                    cur_tids = inp["time_ids"][0]
                    if len(cur_tids) > bmax:
                        inp["time_ids"] = [cur_tids[-bmax:]]

                with torch.no_grad():
                    ad = self._generate_action_with_value(inp, step_id=s.step_id)

                # Cache reference model log-prob
                _ptid = ad["action_token_id"]
                _ptid_int = int(_ptid.item()) if isinstance(_ptid, torch.Tensor) else int(_ptid)
                _ref_lp = self._compute_ref_logprob(inp, _ptid_int)

                action_datas[i] = (ad, inp, input_ids, list(inp["time_ids"][0]), _ref_lp)

            # ── Step ALL envs in parallel ──
            actions = [
                action_datas[i][0]["action_idx"] if i in action_datas else 0
                for i in range(num_envs)
            ]
            # vec_env.step returns List[(obs, reward, done, info)] per env
            step_results = vec_env.step(actions)
            metrics_list = vec_env.get_metrics()

            # ── Process results per active env ──
            for i in active:
                if i not in action_datas:
                    continue
                s                           = es[i]
                ad, inp, input_ids, tids_sn, ref_lp_i = action_datas[i]

                obs_i, _rew, done_i, _info = step_results[i]
                done    = bool(done_i)
                metrics = metrics_list[i]
                reward  = s.reward_fn.step_reward(metrics, s.step_id, done, action_idx=ad["action_idx"])

                step_data = StepData(
                    query_tensor     = input_ids.cpu(),
                    response_tensor  = ad["action_token_id"],
                    state_images     = inp["images"].detach().cpu(),
                    state_depths     = inp["depths"].detach().cpu(),
                    state_poses      = inp["poses"].detach().cpu(),
                    state_intrinsics = inp["intrinsics"].detach().cpu(),
                    state_time_ids   = tids_sn,
                    state_task_type  = 0,
                    action_log_prob  = ad["log_prob"],
                    value            = ad["value"],
                    ref_log_prob     = ref_lp_i,
                    reward           = reward,
                    action_idx       = ad["action_idx"],
                    done             = done,
                    metrics          = dict(metrics),
                )
                s.traj.steps.append(step_data)
                s.step_id += 1

                # Update obs for next step (obs_i from step result)
                s.obs = obs_i

                # Window boundary → reset streaming KV cache
                if s.step_id % cfg.num_frames == 0:
                    self.model.reset_for_env(i)
                    s.tids = []

                # Episode terminal?
                if done or s.step_id >= cfg.max_episode_steps:
                    fin              = metrics_list[i]
                    s.traj.total_reward     = sum(st.reward for st in s.traj.steps)
                    s.traj.success          = bool(fin.get("success", False))
                    s.traj.spl              = float(fin.get("spl", 0.0))
                    s.traj.distance_to_goal = float(
                        fin.get("distance_to_goal", float("inf"))
                    )
                    s.traj.num_steps = s.step_id
                    completed.append(s.traj)

                    ok = "✓" if s.traj.success else "✗"
                    logger.info(
                        f"  [P{i}] Episode {len(completed)}/{target_n} {ok}  "
                        f"steps={s.traj.num_steps:3d}  "
                        f"reward={s.traj.total_reward:+.3f}  "
                        f"dtg={s.traj.distance_to_goal:.1f}m"
                    )

                    # Start next episode if still needed
                    if len(completed) < target_n:
                        _start(i)
                    else:
                        es[i].collecting = False

        return completed[:target_n]

    # ------------------------------------------------------------------
    # Action generation with logit masking (Block 3)
    # ------------------------------------------------------------------

    def _generate_action_with_value(self, input_dict: Dict, step_id: int = 0) -> Dict:
        """
        Generate a single action from the model with logit masking.

        For RL training, we generate ONE action at a time (not multi-step
        sequences) to enable per-step credit assignment.

        The logit mask restricts output to valid action tokens only:
        forward (↑), left (←), right (→), stop (STOP).

        Returns dict with: action_idx, action_token_id, log_prob, value
        """
        cfg = self.config

        # Forward pass through actor-critic to get logits + value
        logits, _, value = self.model(
            **input_dict,
            output_hidden_states=True,
        )

        # Get logits for the LAST position (next-token prediction)
        last_logits = logits[:, -1, :]  # [1, vocab_size]

        # Apply logit mask: only allow valid action tokens
        if cfg.use_logit_mask and self.valid_token_ids:
            mask = torch.full_like(last_logits, float("-inf"))
            mask[:, self.valid_token_ids] = 0.0
            masked_logits = last_logits + mask
        else:
            masked_logits = last_logits

        # Apply temperature
        if cfg.temperature != 1.0:
            masked_logits = masked_logits / cfg.temperature

        # Sample action via multinomial (stochastic for exploration)
        probs = F.softmax(masked_logits, dim=-1)
        action_token_id = torch.multinomial(probs, num_samples=1)  # [1, 1]
        # Use log_softmax for numerical stability instead of log(softmax(.))
        log_prob = F.log_softmax(masked_logits, dim=-1).gather(-1, action_token_id)  # [1, 1]

        # Map token ID back to action index
        token_id = action_token_id.item()
        action_idx = 0  # default STOP
        for action_name, tid in self._action_token_ids.items():
            if tid == token_id:
                action_idx = self.actions2idx[action_name][0]
                break

        # Value at last position
        last_value = value[:, -1].item()

        return {
            "action_idx": action_idx,
            "action_token_id": action_token_id.squeeze().cpu(),
            "log_prob": log_prob.item(),
            "value": last_value,
        }

    def _compute_ref_logprob(self, input_dict: Dict, action_token_id: int) -> Optional[float]:
        """Compute reference-model log-probability for a selected action.

        Called during rollout so the PPO trainer can skip the expensive
        ref-model re-evaluation pass.  Returns None when no ref_model is
        available (single-GPU PEFT mode computes ref logprobs in the trainer).
        """
        if self.ref_model is None:
            return None

        cfg = self.config
        ref_inputs = {}
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                ref_inputs[k] = v.to(self.ref_device, non_blocking=True)
            else:
                ref_inputs[k] = v

        with torch.no_grad():
            logits, _, _ = self.ref_model(**ref_inputs, output_hidden_states=True)

        last_logits = logits[:, -1, :]

        # Apply same logit mask as policy
        if cfg.use_logit_mask and self.valid_token_ids:
            mask = torch.full_like(last_logits, float("-inf"))
            mask[:, self.valid_token_ids] = 0.0
            last_logits = last_logits + mask

        # Apply temperature (must match PPO trainer's computation)
        if cfg.temperature != 1.0:
            last_logits = last_logits / cfg.temperature

        tid = torch.tensor([[action_token_id]], device=last_logits.device)
        log_prob = F.log_softmax(last_logits, dim=-1).gather(-1, tid)
        return log_prob.item()

    # ------------------------------------------------------------------
    # Conversation building (matching VLNEvaluator exactly)
    # ------------------------------------------------------------------

    def _build_conversation(
        self, instruction: str, step_id: int, output_ids_exist: bool
    ) -> list:
        """Build conversation sources matching VLNEvaluator.eval_action()."""
        cfg = self.config
        sources = copy.deepcopy(
            [
                {"from": "human", "value": self.prompt_template},
                {"from": "gpt", "value": ""},
            ]
        )

        if step_id != 0:
            sources[0]["value"] += (
                f" These are your historical observations {DEFAULT_MEMORY_TOKEN}."
            )
        sources[0]["value"] = sources[0]["value"].replace(
            DEFAULT_VIDEO_TOKEN + "\n", ""
        )
        sources[0]["value"] = sources[0]["value"].replace(
            "<instruction>.", instruction
        )

        return sources

    # ------------------------------------------------------------------
    # Tokenization (matching VLNEvaluator.preprocess_qwen exactly)
    # ------------------------------------------------------------------

    def _preprocess_qwen(
        self,
        sources,
        tokenizer,
        has_image: bool = False,
        max_len: int = 2048,
        system_message: str = "You are a helpful assistant.",
        add_system: bool = False,
    ):
        """Tokenize conversation sources into input IDs.

        Mirrors ``VLNEvaluator.preprocess_qwen()`` exactly.
        Uses the pre-built ``_rl_tokenizer`` to avoid a deepcopy + add_tokens
        on every call (the original tokenizer arg is kept for API compat but
        ignored when has_image=True).
        """
        roles = {"human": "user", "gpt": "assistant"}

        # Use the pre-built RL tokenizer (created once in __init__) instead
        # of deepcopy + add_tokens on every step.
        if has_image:
            tok = self._rl_tokenizer
            image_token_index = self._rl_image_token_index
            memory_token_index = self._rl_memory_token_index
        else:
            tok = tokenizer
            image_token_index = None
            memory_token_index = None

        conversations = []
        input_ids = []

        for i, source in enumerate(sources):
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."

            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]

            input_id = []

            if add_system:
                input_id += tok.apply_chat_template(
                    [{"role": "system", "content": system_message}]
                )

            for conv in source:
                try:
                    role = conv["role"]
                    content = conv["content"]
                except (KeyError, TypeError):
                    role = conv["from"]
                    content = conv["value"]

                role = roles.get(role, role)
                conv_msg = [{"role": role, "content": content}]
                conversations.append(content)
                encode_id = tok.apply_chat_template(conv_msg)
                input_id += encode_id

            # Replace special token indices
            if image_token_index is not None:
                for idx, encode_id in enumerate(input_id):
                    if encode_id == image_token_index:
                        input_id[idx] = IMAGE_TOKEN_INDEX
                    if encode_id == memory_token_index:
                        input_id[idx] = MEMORY_TOKEN_INDEX

            input_ids.append(input_id)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids, conversations

    # ------------------------------------------------------------------
    # Geometry helpers (matching VLNEvaluator)
    # ------------------------------------------------------------------

    @staticmethod
    def _xyz_yaw_to_tf_matrix(xyz: np.ndarray, yaw: float) -> np.ndarray:
        x, y, z = xyz
        return np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],
                [np.sin(yaw), np.cos(yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )

    @staticmethod
    def _get_intrinsic_matrix(sensor_cfg) -> np.ndarray:
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0
        return np.array(
            [
                [fx, 0.0, cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    @staticmethod
    def _preprocess_intrinsic(intrinsic, ori_size, target_size):
        intrinsic = copy.deepcopy(intrinsic)
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[None, :, :]
        intrinsic[:, 0] /= ori_size[0] / target_size[0]
        intrinsic[:, 1] /= ori_size[1] / target_size[1]
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2
        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)
        return intrinsic


# ======================================================================
# Trajectory → PPO batch conversion
# ======================================================================

def trajectories_to_transition_batch(
    trajectories: List[EpisodeTrajectory],
) -> Dict[str, list]:
    """Convert trajectories into per-step transitions for multimodal PPO.

    IMPORTANT: The last step of every episode is forced to ``done=True`` so
    that GAE in ``compute_advantages`` never bootstraps across episode
    boundaries.  An assertion guards this invariant.
    """
    transitions = []

    for traj in trajectories:
        if not traj.steps:
            continue

        for step_idx, step in enumerate(traj.steps):
            if (
                step.query_tensor is None
                or step.response_tensor is None
                or step.state_images is None
                or step.state_depths is None
                or step.state_poses is None
                or step.state_intrinsics is None
            ):
                continue

            action_token_id = step.response_tensor
            if isinstance(action_token_id, torch.Tensor):
                action_token_id = int(action_token_id.reshape(-1)[0].item())
            else:
                action_token_id = int(action_token_id)

            # Force trajectory boundary to be terminal for advantage computation.
            # This is critical: GAE must not bootstrap values[t+1] across
            # episode boundaries in the flattened transition buffer.
            done = bool(step.done) or (step_idx == len(traj.steps) - 1)

            transitions.append(
                {
                    "input_ids": step.query_tensor,
                    "images": step.state_images,
                    "depths": step.state_depths,
                    "poses": step.state_poses,
                    "intrinsics": step.state_intrinsics,
                    "time_ids": list(step.state_time_ids or []),
                    "task_type": int(step.state_task_type),
                    "action_token_id": action_token_id,
                    "old_logprob": float(step.action_log_prob),
                    "old_value": float(step.value),
                    "ref_logprob": step.ref_log_prob,
                    "reward": float(step.reward),
                    "done": done,
                    "episode_id": traj.episode_id,
                    "step_idx": step_idx,
                }
            )

    # Verify episode-boundary invariant: the last transition of every
    # episode must be terminal so GAE doesn't leak across episodes.
    if transitions:
        # Walk backwards — every transition where the next one has a
        # different episode_id (or the very last transition) must be done.
        for i in range(len(transitions)):
            is_last = (i == len(transitions) - 1)
            is_boundary = is_last or (
                transitions[i]["episode_id"] != transitions[i + 1]["episode_id"]
            )
            if is_boundary:
                assert transitions[i]["done"], (
                    f"Episode-boundary invariant violated at transition {i} "
                    f"(episode {transitions[i]['episode_id']}): done must be True"
                )

    return {
        "transitions": transitions,
    }
