import logging
from typing import SupportsFloat, Tuple

import numpy as np
from gymnasium.core import ActType
from ray.rllib.utils.typing import AgentID, MultiAgentDict

from core.annotations import override
from core.envs.marl_regulated import MultiAgentRegulatedEnv
from core.utils import sigmoid, smooth_positive_zero_at_origin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

EPS = 1e-8


def pella_tomlinson_step(
    B: float,
    H: float,
    r: float,
    K: float,
    p: float = 1.0,
    noise: float = 0.0,
    restoration: float = 0.0,
) -> tuple[float, float, float]:
    """
    Single-stock surplus production step.

    Returns:
        B_next
        H_realized
        growth
    """

    B = max(float(B), EPS)
    K = max(float(K), EPS)
    p = max(float(p), EPS)

    biological_growth = (r / p) * B * (1.0 - (B / K) ** p)
    growth = biological_growth + noise + restoration
    available = max(B + growth, 0.0)

    H_realized = min(float(H), available)
    B_next = available - H_realized
    B_next = float(np.clip(B_next, 0.0, K))

    return B_next, H_realized, growth


def reference_points(r: float, K: float, p: float = 1.0) -> dict[str, float]:
    p = max(float(p), EPS)

    B_msy = K * (1.0 / (p + 1.0)) ** (1.0 / p)
    MSY = r * K / (p + 1.0) ** ((p + 1.0) / p)
    F_msy = MSY / max(B_msy, EPS)

    return {
        "B_msy": float(B_msy),
        "MSY": float(MSY),
        "F_msy": float(F_msy),
    }


class FisheryRegulatedEnv(MultiAgentRegulatedEnv):
    def __init__(
        self,
        *,
        ecology_cfg: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.r = ecology_cfg.get("r", 0.3)
        self.K = ecology_cfg.get("K", ecology_cfg.get("max_fish", 1000.0))
        self.p = ecology_cfg.get("p", 1.0)
        self.sigma = ecology_cfg.get("sigma", 0.05)

        self.max_fish = self.K
        self.fish_init = ecology_cfg.get("fish_init", ecology_cfg.get("B0", self.K))
        self.restoration_effectiveness = float(
            ecology_cfg.get(
                "restoration_effectiveness",
                0.005,
            )
        )
        self.restoration_effort_cost = float(
            ecology_cfg.get(
                "restoration_effort_cost",
                0.25,
            )
        )

        self.collapse_stock_frac = ecology_cfg.get("collapse_stock_frac", 0.20)
        self.collapse_transition_width = ecology_cfg.get(
            "collapse_transition_width", 0.03
        )
        self.unregulated_f_multiplier = ecology_cfg.get("unregulated_f_multiplier", 2.0)
        self.initial_stock_log_sigma = float(
            ecology_cfg.get("initial_stock_log_sigma", 0.05)
        )

        if self.initial_stock_log_sigma < 0.0:
            raise ValueError("initial_stock_log_sigma must be non-negative")

        self.quota_transition_width = ecology_cfg.get("quota_transition_width", 0.03)
        self.harvest_transition_width = ecology_cfg.get(
            "harvest_transition_width", 0.005
        )
        self.violation_transition_width = ecology_cfg.get(
            "violation_transition_width", 0.03
        )

        rp = reference_points(self.r, self.K, self.p)
        self.B_msy = rp["B_msy"]
        self.MSY = rp["MSY"]
        self.F_msy = rp["F_msy"]

        self.full_required_harvest = 0.0

        self.obs_map = [
            "fish_norm",
            "effective_quota",
            "total_usage_norm",
        ]

    @override(MultiAgentRegulatedEnv)
    def _reset(self):
        if self.initial_stock_log_sigma == 0.0:
            initial_fish = float(self.fish_init)
        else:
            initial_fish = self.rng.lognormal(
                mean=np.log(max(self.fish_init, EPS)),
                sigma=self.initial_stock_log_sigma,  # sigma around sampling from lognormal distribution
            )

        # worth investigating freezing
        self.S_t = {
            "fish": np.clip(
                initial_fish,
                EPS,
                self.K,
            ),
            "last_usage": 0.0,
        }

        # TODO n.b. we do not log metrics at reset ! must be done in subclass ! this could end up
        # skewing the count

        return {
            agent_id: self.observation(agent_id, self.S_t) for agent_id in self.agents
        }

    @override(MultiAgentRegulatedEnv)
    def _is_truncated(self) -> bool:
        return self.horizon is not None and (self._t + 1) >= self.horizon

    def _action_components(
        self,
        action: ActType,
    ) -> tuple[float, float]:
        z = np.asarray(action, dtype=np.float32).reshape(-1)

        if z.size != 2:
            raise ValueError(f"Expected action with 2 elements, got shape {z.shape}")

        temperature = 4.0

        harvest_fraction = float(sigmoid(float(z[0]) / temperature))

        restoration_effort = float(sigmoid(float(z[1]) / temperature))

        return harvest_fraction, restoration_effort

    def _quota_stress(self, fish_norm: float) -> float:
        fish_norm = float(np.clip(fish_norm, 0.0, 1.0))
        width = max(self.quota_transition_width, EPS)
        lower = sigmoid((0.0 - self.mechanism.fixed_quota) / width)
        upper = sigmoid((1.0 - self.mechanism.fixed_quota) / width)
        current = sigmoid((fish_norm - self.mechanism.fixed_quota) / width)
        return (current - lower) / max(upper - lower, EPS)

    def _allowed_frac(self, fish_norm: float) -> float:
        stress = self._quota_stress(fish_norm)

        return float(stress * self.mechanism.max_demand_frac)

    def _allowed_harvest(self, fish_norm: float) -> float:
        return self._allowed_frac(fish_norm) * self.full_required_harvest

    def intrinsic_utility(
        self,
        A_t: dict[AgentID, ActType],
    ) -> MultiAgentDict:
        fish = float(self.S_t["fish"])
        fish_norm = fish / max(self.max_fish, EPS)

        # self.full_required_harvest = fish / len(self.agents)
        # self.full_required_harvest = (
        #     self.F_msy * fish / len(self.agents)
        # )
        self.full_required_harvest = (
            self.unregulated_f_multiplier * self.F_msy * fish / len(self.agents)
        )

        utilities = {}

        for agent_id, action in A_t.items():
            harvest_frac, _ = self._action_components(action)

            requested_harvest = harvest_frac * self.full_required_harvest
            allowed_harvest = self._allowed_harvest(fish_norm)
            delivered_harvest = requested_harvest - (
                smooth_positive_zero_at_origin(
                    requested_harvest - allowed_harvest,
                    self.harvest_transition_width * self.full_required_harvest,
                )
            )
            requested_frac_norm = requested_harvest / max(
                EPS, self.full_required_harvest
            )
            utility = delivered_harvest / max(EPS, self.full_required_harvest)
            utilities[agent_id] = utility

            self.logger.push(
                key=("by_agent", agent_id, "requested_harvest"), value=requested_harvest
            )
            self.logger.push(
                key=("by_agent", agent_id, "delivered_harvest"), value=delivered_harvest
            )
            self.logger.push(
                key=("by_agent", agent_id, "requested_frac"), value=requested_frac_norm
            )
            # TODO move to base class
            self.logger.push(
                key=("by_agent", agent_id, "intrinsic_utility"), value=utility
            )

        return utilities

    def violation_signal(
        self,
        u_i: SupportsFloat,
        agent_id: AgentID,
        *,
        A_t: MultiAgentDict,
    ) -> SupportsFloat:
        fish = float(self.S_t["fish"])
        fish_norm = fish / max(self.max_fish, EPS)

        harvest_frac, _ = self._action_components(A_t[agent_id])

        requested_harvest = harvest_frac * self.full_required_harvest

        allowed_frac = self._allowed_frac(fish_norm)

        requested_frac_norm = requested_harvest / max(EPS, self.full_required_harvest)

        delivered_frac_norm = float(np.clip(u_i, 0.0, 1.0))

        violation_frac = smooth_positive_zero_at_origin(
            requested_frac_norm - allowed_frac,
            self.violation_transition_width,
        )

        quota_violation = violation_frac * self.full_required_harvest

        quota_penalty = min(
            1.0,
            self.mechanism.fine_amount * violation_frac,
        )

        stock_pressure = max(0.0, 1.0 - fish_norm)

        risk_penalty = (
            self.mechanism.risk_penalty_scale
            * stock_pressure
            * (delivered_frac_norm**self.mechanism.risk_penalty_power)
        )

        total_penalty = min(1.0, quota_penalty + risk_penalty)

        self.logger.push(
            key=("by_agent", agent_id, "quota_violation"), value=quota_violation
        )
        self.logger.push(
            key=("by_agent", agent_id, "quota_penalty"), value=quota_penalty
        )
        self.logger.push(key=("by_agent", agent_id, "risk_penalty"), value=risk_penalty)

        return total_penalty

    def penalty(self, u_i: SupportsFloat, **kwargs) -> SupportsFloat:
        return 1.0

    def transition_kernel(
        self,
        *,
        A_t: MultiAgentDict,
        S_t: dict,
    ) -> dict[str, float]:
        fish = float(S_t["fish"])
        fish_norm = fish / max(self.max_fish, EPS)
        allowed_harvest = self._allowed_harvest(fish_norm)
        quota_stress = self._quota_stress(fish_norm)

        self.full_required_harvest = (
            self.unregulated_f_multiplier * self.F_msy * fish / len(self.agents)
        )

        delivered_harvest = {}

        restoration_efforts = {}

        for agent_id, action in A_t.items():
            harvest_frac, restoration_effort = self._action_components(action)

            restoration_efforts[agent_id] = restoration_effort

            requested_harvest = harvest_frac * self.full_required_harvest

            delivered_harvest[agent_id] = (
                requested_harvest
                - smooth_positive_zero_at_origin(
                    requested_harvest - allowed_harvest,
                    self.harvest_transition_width * self.full_required_harvest,
                )
            )

        mean_restoration_effort = float(np.mean(list(restoration_efforts.values())))

        restoration_gain = (
            self.restoration_effectiveness * mean_restoration_effort * self.K
        )

        H_attempted = float(sum(delivered_harvest.values()))

        noise = self.sigma * self.rng.normal() * fish

        fish_next, H_realized, growth = pella_tomlinson_step(
            B=fish,
            H=H_attempted,
            r=self.r,
            K=self.K,
            p=self.p,
            noise=noise,
            restoration=restoration_gain,
        )

        new_state = {
            "fish": fish_next,
            "last_usage": H_realized,
        }

        # TODO fallback when user forgets a comms
        self.logger.push(key=("quota_stress",), value=quota_stress)
        self.logger.push(key=("allowed_harvest",), value=allowed_harvest)
        self.logger.push(key=("fish_stock",), value=fish)
        self.logger.push(key=("growth",), value=growth)
        self.logger.push(key=("growth_noise",), value=noise)
        self.logger.push(key=("H_attempted",), value=H_attempted)
        self.logger.push(key=("H_realized",), value=H_realized)
        self.logger.push(
            key=("total_usage_norm",), value=H_realized / max(EPS, self.max_fish)
        )
        self.logger.push(key=("B_msy",), value=self.B_msy)
        self.logger.push(key=("MSY",), value=self.MSY)
        self.logger.push(key=("F_msy",), value=self.F_msy)

        # TODO not necessary  yo have next and now?
        self.logger.push(key=("fish_stock_next",), value=fish_next)
        self.logger.push(key=("fish_norm",), value=fish_norm)
        self.logger.push(
            key=("fish_norm_next_mean",), value=fish_next / max(self.max_fish, EPS)
        )
        self.logger.push(
            key=("fish_norm_next_min",), value=fish_next / max(self.max_fish, EPS)
        )
        self.logger.push(
            key=("fish_norm_next_max",), value=fish_next / max(self.max_fish, EPS)
        )
        self.logger.push(
            key=("fish_norm_next_last",), value=fish_next / max(self.max_fish, EPS)
        )
        

        # Move this to mechanism logging
        # self.logger.push(key=("by_agent", agent_id, "max_demand_frac"), value=self.mechanism.max_demand_frac)

        self.S_t = new_state
        return self.S_t

    def _observation(
        self,
        agent_id: AgentID,
        S_t: dict,
    ):
        fish_norm = float(S_t["fish"] / max(self.max_fish, EPS))
        effective_quota = self._allowed_frac(fish_norm)
        total_usage_norm = float(S_t.get("last_usage", 0.0) / max(EPS, self.max_fish))

        return np.array(
            [
                fish_norm,
                effective_quota,
                total_usage_norm,
            ],
            dtype=np.float32,
        )

    def _step(
        self,
        action_dict: dict[AgentID, ActType],
    ) -> Tuple[
        MultiAgentDict,
        MultiAgentDict,
        MultiAgentDict,
        MultiAgentDict,
        MultiAgentDict,
    ]:
        S_t = self.S_t.copy()
        fish = float(S_t["fish"])
        fish_norm = fish / max(self.max_fish, EPS)

        # Same parent-style logic as water:
        # utility first, violation second, then transition
        utilities = self.intrinsic_utility(action_dict)

        violations = {
            agent_id: self.violation_signal(
                utilities[agent_id],
                agent_id,
                A_t=action_dict,
            )
            for agent_id in self.agents
        }
        restoration_efforts = {
            agent_id: self._action_components(action_dict[agent_id])[1]
            for agent_id in self.agents
        }
        restoration_costs = {
            agent_id: (
                self.restoration_effort_cost * restoration_efforts[agent_id] ** 2
            )
            for agent_id in self.agents
        }

        restoration_subsidies = {
            agent_id: (
                self.mechanism.restoration_subsidy * restoration_efforts[agent_id]
            )
            for agent_id in self.agents
        }

        S_next = self.transition_kernel(
            A_t=action_dict,
            S_t=S_t,
        )
        H_realized = float(S_next["last_usage"])
        harvest_to_msy = H_realized / max(self.MSY, EPS)

        fish_norm_next = float(S_next["fish"] / max(self.max_fish, EPS))

        collapse_penalty = 0.1 / (
            1.0
            + np.exp(
                np.clip(
                    (fish_norm_next - self.collapse_stock_frac)
                    / self.collapse_transition_width,
                    -60.0,
                    60.0,
                )
            )
        )

        rewards = {
            agent_id: float(
                utilities[agent_id]
                - violations[agent_id]
                - collapse_penalty
                - restoration_costs[agent_id]
                + restoration_subsidies[agent_id]
            )
            for agent_id in self.agents
        }

        obs = {agent_id: self.observation(agent_id, S_next) for agent_id in self.agents}

        time_limit = self._is_truncated()

        terminated = {agent_id: False for agent_id in self.agents}
        terminated["__all__"] = False

        truncated = {agent_id: time_limit for agent_id in self.agents}
        truncated["__all__"] = time_limit

        infos = {}

        return obs, rewards, terminated, truncated, infos
