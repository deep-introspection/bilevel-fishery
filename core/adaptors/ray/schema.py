from typing import Optional, TypeAlias

from pydantic import Field

from core.envs.schema import EpisodeRolloutSchema
from core.metrics.enums import ReduceProtocol
from core.metrics.schemas import MetricSchema

PolicyID: TypeAlias = str
EpisodeID: TypeAlias = str
MechanismID: TypeAlias = str
SeedID: TypeAlias = str


class PolicyLearnerSchema(MetricSchema):
    batch_size: Optional[int] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )

    # Value, Q, advantage debugging
    total_loss: Optional[float] = Field(
        default=None, json_schema_extra={"source": "total_loss"}
    )
    residual_variance: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )

    # TODO this is a callable
    sample_staleness: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )

    # Policy (π) debugging
    policy_loss: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    policy_entropy: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    policy_entropy_coeff: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    policy_relative_entropy: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    entropy_pressure: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    policy_kl: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    policy_kl_coeff: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )

    # TODO what is total loss ?
    # TODO kl vs kl loss
    # TODO curr_kl_coeff
    # TODO entropy vs entropy coeff

    # Value (V) debgging
    value_loss: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    value_mean: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    value_target: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )

    # TODO Q-statistics
    # TODO Advantage statistics
    # TODO advantage statistics

    # TODO Reward (R) debugging
    # Reward metrics. N.B. episode == trajectory

    # Gradient debugging
    gradient_norm: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    gradient_noise: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )


class PerformanceSchema(MetricSchema):
    env_steps_this_iter: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},
    )
    env_steps_lifetime: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},
    )
    agent_steps_this_iter_sum: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},
    )
    agent_steps_lifetime_sum: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},
    )
    env_steps_throughput: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    training_iteration_s: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    training_step_s: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    sample_s: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    learner_update_s: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    weights_seq_no: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},
    )


class SeedRolloutSchema(MetricSchema):
    by_episode: dict[EpisodeID, EpisodeRolloutSchema] = Field(default_factory=dict)


class MechanismRolloutSchema(MetricSchema):
    by_seed: dict[SeedID, SeedRolloutSchema] = Field(default_factory=dict)


class RolloutSchema(MetricSchema):
    aggregate: EpisodeRolloutSchema
    by_mechanism: dict[MechanismID, MechanismRolloutSchema] = Field(
        default_factory=dict
    )


class SeedLearnerSchema(MetricSchema):
    by_policy: dict[PolicyID, PolicyLearnerSchema] = Field(default_factory=dict)


class MechanismLearnerSchema(MetricSchema):
    """Training/policy initialization seed, NOT evaluation environment seed."""

    by_seed: dict[SeedID, SeedLearnerSchema] = Field(default_factory=dict)


class LearnerSchema(MetricSchema):
    by_mechanism: dict[MechanismID, MechanismLearnerSchema] = Field(
        default_factory=dict
    )


class TrainSchema(MetricSchema):
    rollout: RolloutSchema
    learner: LearnerSchema
    performance: PerformanceSchema


class EvalSchema(MetricSchema):
    rollout: RolloutSchema
    performance: PerformanceSchema


class RaySchema(MetricSchema):
    train: Optional[TrainSchema] = Field(
        default=None,
        json_schema_extra={"subtree_reduce": ReduceProtocol.LAST},
    )
    eval: Optional[EvalSchema] = Field(
        default=None,
        json_schema_extra={"subtree_reduce": ReduceProtocol.LAST},
    )


# TODO
# num_env_steps_sampled_lifetime_throughput
# timers
# throughput_since_last_restore
# num_agent_steps_sampled
# num_agent_steps_sampled_lifetime
# prevent non terminal leaves to have reduce objects
