"""Per-environment sensor timestamps; actuator filtering has a separate clock."""
from dataclasses import dataclass
import math
import torch

@dataclass(frozen=True)
class PerceptionDelayConfig:
    mode: str = "discrete_history_sample_and_hold"
    policy_dt_s: float = .02
    acquisition_period_s: float = .02
    refresh_period_s: float = .1
    latency_min_s: float = .04
    latency_max_s: float = .08

    def __post_init__(self):
        if self.mode not in ("discrete_history_sample_and_hold", "paper_diagnostic"):
            raise ValueError("unsupported perception mode")
        for key in ("policy_dt_s", "acquisition_period_s", "refresh_period_s", "latency_min_s", "latency_max_s"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(key + " must be finite and positive")
        if self.latency_max_s < self.latency_min_s:
            raise ValueError("latency bounds reversed")
        for value in (self.acquisition_period_s, self.refresh_period_s):
            ratio = value / self.policy_dt_s
            if abs(ratio - round(ratio)) > 1e-7 or ratio < 1:
                raise ValueError("perception periods must be integral policy ticks")
        if self.mode == "discrete_history_sample_and_hold" and self.acquisition_period_s != self.policy_dt_s:
            raise ValueError("upstream acquisition must happen every policy tick")

@dataclass(frozen=True)
class PerceptionObservation:
    rays: torch.Tensor
    goals: torch.Tensor
    sample_timestamp: torch.Tensor
    sampled_latency: torch.Tensor
    actual_age: torch.Tensor
    synthetic_bootstrap: torch.Tensor

class TimestampedPerception:
    """Bounded ring with masked noise-free synthetic step-0 reconstruction.

    Upstream samples 2/3 ticks back at each 100ms refresh. The separately named
    diagnostic model acquires at 10Hz, samples transport latency on acquisition,
    and publishes the latest arrived packet on a policy tick, holding otherwise.
    """
    def __init__(self, config, num_envs, num_rays, device="cpu"):
        self.config = config
        self.capacity = max(5, int(math.ceil((config.latency_max_s + config.acquisition_period_s) / config.policy_dt_s)) + 3)
        shape = (num_envs, self.capacity)
        self.rays = torch.zeros(*shape, num_rays, device=device)
        self.goals = torch.zeros(*shape, 2, device=device)
        self.times = torch.full(shape, -float("inf"), dtype=torch.float64, device=device)
        self.latencies = torch.zeros(shape, dtype=torch.float64, device=device)
        self.write = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.count = self.write.clone()
        self.epoch = torch.zeros(num_envs, dtype=torch.float64, device=device)
        self.next_acquisition = self.epoch.clone()
        self.next_refresh = self.epoch.clone()
        self.held_rays = torch.zeros(num_envs, num_rays, device=device)
        self.held_goals = torch.zeros(num_envs, 2, device=device)
        self.held_time = self.epoch.clone()
        self.held_latency = self.epoch.clone()
        self.synthetic = torch.ones(num_envs, dtype=torch.bool, device=device)

    def _time(self, ids, value):
        return torch.as_tensor(value, dtype=torch.float64, device=self.times.device).expand(ids.numel())

    def reset(self, ids, now, rays, goals):
        now = self._time(ids, now)
        self.times[ids] = -float("inf")
        self.write[ids] = 0
        self.count[ids] = 0
        self.epoch[ids] = now
        self.next_acquisition[ids] = now + self.config.acquisition_period_s
        self.next_refresh[ids] = now + self.config.refresh_period_s
        self.held_rays[ids] = rays
        self.held_goals[ids] = goals
        self.held_time[ids] = now
        self.held_latency[ids] = 0
        self.synthetic[ids] = True
        self._store(ids, now, rays, goals, torch.zeros_like(now))

    def _store(self, ids, now, rays, goals, latency):
        slots = self.write[ids]
        self.rays[ids, slots] = rays
        self.goals[ids, slots] = goals
        self.times[ids, slots] = now
        self.latencies[ids, slots] = latency
        self.write[ids] = (slots + 1) % self.capacity
        self.count[ids] = (self.count[ids] + 1).clamp(max=self.capacity)

    def push(self, ids, now, rays, goals, generator=None):
        now = self._time(ids, now)
        due = now + 1e-8 >= self.next_acquisition[ids]
        active = ids[due]
        times = now[due]
        cfg = self.config
        if cfg.mode == "paper_diagnostic":
            latency = torch.rand(active.numel(), generator=generator, device=self.times.device,
                                 dtype=torch.float64) * (cfg.latency_max_s - cfg.latency_min_s) + cfg.latency_min_s
        else:
            latency = torch.zeros_like(times)
        self._store(active, times, rays[due], goals[due], latency)
        self.next_acquisition[active] = times + cfg.acquisition_period_s
        # Startup has no historical packet: publish actual current samples.
        if cfg.mode == "discrete_history_sample_and_hold":
            startup = self.count[active] < 4
            first = active[startup]
            self.held_rays[first] = rays[due][startup]
            self.held_goals[first] = goals[due][startup]
            self.held_time[first] = times[startup]
            self.synthetic[first] = False

    def observe(self, ids, now, generator=None):
        now = self._time(ids, now)
        cfg = self.config
        due = now + 1e-8 >= self.next_refresh[ids]
        active = ids[due]
        if cfg.mode == "discrete_history_sample_and_hold":
            ready = self.count[active] >= 4
            active = active[ready]
            offsets = torch.randint(2, 4, (active.numel(),), generator=generator, device=self.times.device)
            slots = (self.write[active] - 1 - offsets) % self.capacity
            self._hold(active, slots)
            self.held_latency[active] = offsets.to(torch.float64) * cfg.policy_dt_s
        else:
            candidate_time = self.times[active]
            arrived = candidate_time + self.latencies[active] <= now[due, None] + 1e-8
            candidates = torch.where(arrived, candidate_time, torch.full_like(candidate_time, -float("inf")))
            latest, slots = candidates.max(dim=1)
            changed = latest > self.held_time[active]
            self._hold(active[changed], slots[changed])
        self.next_refresh[ids[due]] = now[due] + cfg.refresh_period_s
        return PerceptionObservation(self.held_rays[ids], self.held_goals[ids], self.held_time[ids],
                                     self.held_latency[ids], now - self.held_time[ids], self.synthetic[ids])

    def _hold(self, ids, slots):
        self.held_rays[ids] = self.rays[ids, slots]
        self.held_goals[ids] = self.goals[ids, slots]
        self.held_time[ids] = self.times[ids, slots]
        self.held_latency[ids] = self.latencies[ids, slots]
        self.synthetic[ids] = False
