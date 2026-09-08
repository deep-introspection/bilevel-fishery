import hashlib
from typing import Optional

import numpy as np
import torch
from ray.rllib.utils.typing import ResultDict

from core.adaptors.ray.schema import (
    LearnerSchema,
    MechanismID,
    MechanismLearnerSchema,
    MechanismRolloutSchema,
    PerformanceSchema,
    PolicyID,
    PolicyLearnerSchema,
    RolloutSchema,
    SeedID,
    SeedLearnerSchema,
    SeedRolloutSchema,
)
from core.envs.schema import EpisodeRolloutSchema
from core.utils import finite, safe_ratio, to_float


def _get_env(result: dict) -> dict:
    return result.get("env_runners", {}) or {}


def get_episode_return_mean(result: dict) -> float:
    env = _get_env(result)
    v = to_float(env.get("episode_return_mean"))

    if v is not None:
        return v

    v = to_float(result.get("episode_reward_mean")) or to_float(
        env.get("episode_reward_mean")
    )

    return v if v is not None else 0.0


def get_env_steps(result: dict) -> tuple[int, int]:
    env = _get_env(result)
    steps_iter = to_float(env.get("num_env_steps_sampled")) or to_float(
        result.get("timesteps_this_iter")
    )
    steps_life = to_float(env.get("num_env_steps_sampled_lifetime")) or to_float(
        result.get("timesteps_total")
    )

    return int(steps_iter or 0), int(steps_life or 0)


def get_policy_loss_if_present(result: dict) -> float:
    learner_info = (result.get("info") or {}).get("learner") or {}
    losses = []

    if isinstance(learner_info, dict):
        for _, policy_stats in learner_info.items():
            ls = (policy_stats or {}).get("learner_stats") or {}
            v = to_float(ls.get("policy_loss"))

            if v is not None:
                losses.append(v)

    return float(np.mean(losses)) if losses else float("nan")


def hash_weights(weights) -> str:
    h = hashlib.sha256()

    def update(obj, prefix=""):
        if isinstance(obj, dict):
            for key in sorted(obj):
                update(obj[key], f"{prefix}/{key}")
        elif isinstance(obj, torch.Tensor):
            array = obj.detach().cpu().contiguous().numpy()

            h.update(prefix.encode())
            h.update(array.tobytes())
        elif isinstance(obj, np.ndarray):
            h.update(prefix.encode())
            h.update(np.ascontiguousarray(obj).tobytes())
        else:
            h.update(prefix.encode())
            h.update(repr(obj).encode())

    update(weights)

    return h.hexdigest()


def parse_learner_id(
    learner_id: str,
) -> tuple[PolicyID, MechanismID, SeedID]:
    try:
        policy_and_mechanism, policy_seed = learner_id.rsplit("_s", 1)
        policy_id, mechanism_id = policy_and_mechanism.rsplit("_m", 1)
    except ValueError:
        raise ValueError(
            "Expected learner ID of the form "
            f"'<policy>_m<mechanism>_s<seed>', got {learner_id!r}."
        ) from None

    return (
        policy_id,
        mechanism_id,
        policy_seed,
    )


# TODO remove finite
def build_episode_aggregate(results: ResultDict) -> EpisodeRolloutSchema:
    env = results.get("env_runners", {}) or {}

    return EpisodeRolloutSchema(
        reward_total=None,
        reward_mean=finite(env.get("episode_return_mean")),
        reward_min=finite(env.get("episode_return_min")),
        reward_max=finite(env.get("episode_return_max")),
        reward_terminal=None,
        value_terminal=None,
        value_penultimate=None,
        episode_len_mean=finite(env.get("episode_len_mean")),
        episode_len_min=finite(env.get("episode_len_min")),
        episode_len_max=finite(env.get("episode_len_max")),
        num_episodes=finite(env.get("num_episodes")),
        num_episodes_lifetime=finite(env.get("num_episodes_lifetime")),
    )


def build_performance(results: ResultDict) -> PerformanceSchema:
    env = results.get("env_runners", {}) or {}
    timers = results.get("timers", {}) or {}
    throughput_data = env.get("num_env_steps_sampled_lifetime_throughput")
    throughput = None

    if isinstance(throughput_data, dict):
        throughput = finite(
            throughput_data.get("throughput_since_last_reduce")
        ) or finite(throughput_data.get("throughput_since_last_restore"))

    # TODO refactor this to get data from env
    agent_steps = env.get("num_agent_steps_sampled")
    agent_steps_lifetime = env.get("num_agent_steps_sampled_lifetime")
    agent_steps_sum = None
    agent_steps_lifetime_sum = None

    if isinstance(agent_steps, dict):
        agent_steps_sum = finite(
            sum((to_float(value) or 0.0) for value in agent_steps.values())
        )

    if isinstance(agent_steps_lifetime, dict):
        agent_steps_lifetime_sum = finite(
            sum((to_float(value) or 0.0) for value in agent_steps_lifetime.values())
        )

    return PerformanceSchema(
        env_steps_this_iter=finite(env.get("num_env_steps_sampled")),
        env_steps_lifetime=finite(env.get("num_env_steps_sampled_lifetime")),
        agent_steps_this_iter_sum=agent_steps_sum,
        agent_steps_lifetime_sum=agent_steps_lifetime_sum,
        env_steps_throughput=throughput,
        training_iteration_s=finite(timers.get("training_iteration")),
        training_step_s=finite(timers.get("training_step")),
        sample_s=finite(timers.get("sample")),
        learner_update_s=finite(timers.get("learner_update_timer")),
        weights_seq_no=finite(env.get("weights_seq_no")),
    )


def build_rollout(results: ResultDict) -> RolloutSchema:
    env = results.get("env_runners", {}) or {}
    episodes = env.get("by_episode", {}) or {}

    by_mechanism: dict[MechanismID, MechanismRolloutSchema] = {}

    for episode_id, episode in episodes.items():
        mechanism_id = str(episode.mechanism_id)
        seed = str(episode.seed)
        mechanism = by_mechanism.setdefault(mechanism_id, MechanismRolloutSchema())
        seed_rollout = mechanism.by_seed.setdefault(seed, SeedRolloutSchema())
        seed_rollout.by_episode[episode_id] = episode

    return RolloutSchema(
        aggregate=build_episode_aggregate(results),
        by_mechanism=by_mechanism,
    )


def build_learner(results: ResultDict) -> LearnerSchema:
    learners = results.get("learners", {}) or {}
    learner_group = results.get("learner_group", {}) or {}
    mean_training_calls_since_sync = finite(
        results.get("mean_num_training_step_calls_since_last_synch_worker_weights")
    )
    outstanding_async_reqs = finite(
        learner_group.get("actor_manager_num_outstanding_async_reqs")
    )
    all_modules_stats = learners.get("__all_modules__", {}) or {}
    learner_queue_wait = finite(
        all_modules_stats.get("learner_thread_in_queue_wait_timer")
    )

    by_mechanism: dict[MechanismID, MechanismLearnerSchema] = {}

    for learner_id, stats in learners.items():
        if learner_id == "__all_modules__":
            continue

        policy_id, mechanism_id, policy_seed = parse_learner_id(learner_id)

        m: dict[str, Optional[float]] = {}

        for key, value in stats.items():
            value = finite(value)

            if value is None:
                continue

            m[str(key)] = value

        # Legacy optionally unpacked this throughput dict.
        throughput = stats.get("num_module_steps_trained_lifetime_throughput")

        if isinstance(throughput, dict):
            m["module_steps_throughput_since_last_reduce"] = finite(
                throughput.get("throughput_since_last_reduce")
            )
            m["module_steps_throughput_since_last_restore"] = finite(
                throughput.get("throughput_since_last_restore")
            )

        entropy = m.get("entropy")
        entropy_coeff = m.get("curr_entropy_coeff")
        m["policy_relative_entropy"] = safe_ratio(entropy, entropy_coeff)

        if entropy is not None and entropy_coeff is not None:
            m["entropy_pressure"] = float(entropy) * float(entropy_coeff)

        lag1 = m.get("diff_num_grad_updates_vs_sampler_policy")
        lag2 = mean_training_calls_since_sync
        lag3 = outstanding_async_reqs
        lag4 = learner_queue_wait
        parts = [value for value in (lag1, lag2, lag3, lag4) if value is not None]
        m["sample_staleness"] = float(sum(parts)) if parts else None
        policy_metrics = PolicyLearnerSchema(
            batch_size=m.get("module_train_batch_size_mean"),
            total_loss=m.get("total_loss"),
            residual_variance=None,
            sample_staleness=m.get("sample_staleness"),
            policy_loss=m.get("policy_loss"),
            policy_entropy=m.get("entropy"),
            policy_entropy_coeff=m.get("curr_entropy_coeff"),
            policy_relative_entropy=m.get("policy_relative_entropy"),
            entropy_pressure=m.get("entropy_pressure"),
            policy_kl=m.get("kl"),
            policy_kl_coeff=m.get("curr_kl_coeff"),
            value_loss=m.get("vf_loss"),
            value_mean=m.get("value_mean"),
            value_target=m.get("value_target"),
            gradient_norm=(
                m.get("gradients_default_optimizer_global_norm") or m.get("grad_gnorm")
            ),
            gradient_noise=m.get("gradient_noise"),
        )
        mechanism = by_mechanism.setdefault(
            mechanism_id,
            MechanismLearnerSchema(),
        )
        seed = mechanism.by_seed.setdefault(
            policy_seed,
            SeedLearnerSchema(),
        )
        seed.by_policy[policy_id] = policy_metrics

    return LearnerSchema(by_mechanism=by_mechanism)
