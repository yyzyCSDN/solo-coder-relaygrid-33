"""Unified fault-location fusion.

故障指示器、阻抗测距、拓扑位置和遥测质量各自给出结论时，可能把班组引向
不同区段。本模块把四类证据统一为候选区段：

* 按证据时间与可信度加权，缩小候选范围；
* 证据冲突（无一致交集且无多数）时保留多个候选；
* 同一来源只采用最新证据（序号优先，时间次之），旧遥测不得覆盖新判断；
* 每个候选都列出依据（支持/反对/被取代/已过期）。
"""
from __future__ import annotations

from dataclasses import dataclass

from relaygrid.faults.indicators import confidence, infer_fault_zone
from relaygrid.faults.location import ImpedanceEstimate, rank_sections
from relaygrid.models import FeederSnapshot, TelemetrySample
from relaygrid.telemetry.quality import quality_score

INDICATOR = "indicator"
IMPEDANCE = "impedance"
TOPOLOGY = "topology"
TELEMETRY = "telemetry"

SOURCE_LABELS = {
    INDICATOR: "故障指示器",
    IMPEDANCE: "阻抗测距",
    TOPOLOGY: "拓扑位置",
    TELEMETRY: "遥测质量",
}

DEFAULT_MAX_AGE_S = 300.0


@dataclass(frozen=True)
class Evidence:
    """A single piece of fault-location evidence.

    sections: 该证据指向的候选区段名称（可多个）。
    sequence: 单调序号，新证据优先于旧证据（旧遥测不得覆盖新判断）。
    credibility: 可信度 0..1，由遥测质量 / 指示器置信度等给出。
    """

    source: str
    observed_at: float
    sequence: int
    credibility: float
    sections: tuple[str, ...]
    basis: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.credibility <= 1.0:
            raise ValueError("credibility must be in [0, 1]")
        if not self.sections:
            raise ValueError("evidence must name at least one section")
        if self.source not in SOURCE_LABELS:
            raise ValueError(f"unknown source: {self.source!r}")


@dataclass(frozen=True)
class FusedCandidate:
    """A candidate section with the evidence basis behind it."""

    section: str
    score: float  # 0..1 share of effective weight supporting this section
    support: tuple[str, ...]  # source labels supporting
    opposition: tuple[str, ...]  # authoritative source labels not supporting
    reasons: tuple[str, ...]  # human-readable basis, one line per item


@dataclass(frozen=True)
class FusionResult:
    """Unified positioning result."""

    candidates: tuple[FusedCandidate, ...]
    narrowed: bool  # candidate range reduced by agreement or majority
    conflict: bool  # evidence conflicted -> multiple candidates kept
    authoritative: tuple[str, ...]  # sources whose newest evidence was applied
    superseded: tuple[str, ...]  # sources whose older evidence was dropped
    stale: tuple[str, ...]  # sources whose evidence aged to zero


def _freshness(observed_at: float, anchor_t: float, max_age_s: float | None) -> float:
    """Recency of evidence relative to the newest judgment (anchor).

    Newer evidence anchors at 1.0; older evidence decays linearly to 0 at
    max_age_s. This is what stops old telemetry from overriding new judgments.
    """
    if max_age_s is None:
        return 1.0
    if max_age_s <= 0:
        return 1.0 if observed_at == anchor_t else 0.0
    age = max(0.0, anchor_t - observed_at)
    return max(0.0, 1.0 - age / max_age_s)


def _dedupe(sections: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for name in sections:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def fuse(
    evidence: list[Evidence],
    *,
    all_sections: list[str] | tuple[str, ...] | None = None,
    max_age_s: float | None = DEFAULT_MAX_AGE_S,
    quorum: float = 0.5,
) -> FusionResult:
    """Fuse fault-location evidence into a unified candidate list.

    Rule order:
      1. per-source newest evidence wins by (sequence, observed_at); older
         evidence is marked superseded and never applied;
      2. effective weight = credibility x recency (relative to newest
         judgment); evidence aged to zero is marked stale and excluded;
      3. unanimous intersection narrows first; otherwise a section with more
         than ``quorum`` of the weight narrows; otherwise evidence conflicts
         and every viable candidate is kept.
    """
    if not 0.0 < quorum <= 1.0:
        raise ValueError("quorum must be in (0, 1]")

    # 1. per-source newest evidence by (sequence, observed_at), mirroring
    #    telemetry.quality.accept_sample: higher sequence wins, then later time.
    newest_idx: dict[str, int] = {}
    newest_key: dict[str, tuple[int, float]] = {}
    for idx, ev in enumerate(evidence):
        key = (ev.sequence, ev.observed_at)
        cur = newest_key.get(ev.source)
        if cur is None or key > cur:
            newest_key[ev.source] = key
            newest_idx[ev.source] = idx

    authoritative: list[tuple[Evidence, int]] = []
    superseded: list[tuple[str, int, float, int, float]] = []
    for idx, ev in enumerate(evidence):
        if newest_idx.get(ev.source) == idx:
            authoritative.append((ev, idx))
        else:
            new_ev = evidence[newest_idx[ev.source]]
            superseded.append(
                (ev.source, ev.sequence, ev.observed_at, new_ev.sequence, new_ev.observed_at)
            )

    # 2. recency anchored at the newest judgment so old telemetry cannot
    #    override newer conclusions.
    anchor_t = max((ev.observed_at for ev, _ in authoritative), default=0.0)
    weight_by_idx: dict[int, float] = {}
    stale_sources: list[str] = []
    for ev, idx in authoritative:
        w = ev.credibility * _freshness(ev.observed_at, anchor_t, max_age_s)
        weight_by_idx[idx] = w
        if w <= 0.0:
            stale_sources.append(ev.source)

    applied = [(ev, idx) for ev, idx in authoritative if weight_by_idx[idx] > 0.0]
    total_weight = sum(weight_by_idx[idx] for _, idx in applied)

    section_weight: dict[str, float] = {}
    section_sources: dict[str, list[str]] = {}
    for ev, idx in applied:
        w = weight_by_idx[idx]
        for name in ev.sections:
            if all_sections is not None and name not in all_sections:
                continue
            section_weight[name] = section_weight.get(name, 0.0) + w
            section_sources.setdefault(name, []).append(ev.source)

    viable = sorted(section_weight, key=lambda n: (-section_weight[n], n))

    # 3. narrowing: unanimous intersection first, then majority, else conflict.
    narrowed = False
    conflict = False
    narrowed_set: set[str] = set(viable)
    if applied and viable:
        common = set(applied[0][0].sections)
        for ev, _ in applied[1:]:
            common &= set(ev.sections)
        if all_sections is not None:
            common &= set(all_sections)
        if common:
            narrowed = True
            narrowed_set = set(common)
        elif total_weight > 0 and any(
            section_weight[n] > total_weight * quorum for n in viable
        ):
            narrowed = True
            narrowed_set = {
                n for n in viable if section_weight[n] > total_weight * quorum
            }
        else:
            conflict = True

    candidates: list[FusedCandidate] = []
    for name in viable:
        if name not in narrowed_set:
            continue
        score = section_weight[name] / total_weight if total_weight > 0 else 0.0
        support = tuple(section_sources.get(name, ()))
        opposition = tuple(
            ev.source for ev, _ in applied if ev.source not in support
        )
        reasons: list[str] = []
        for ev, idx in applied:
            label = SOURCE_LABELS[ev.source]
            w = weight_by_idx[idx]
            if ev.source in support:
                reasons.append(
                    f"{label} 支持（观测 t={ev.observed_at:g}，序号 {ev.sequence}，"
                    f"可信度 {ev.credibility:.2f}，权重 {w:.2f}）"
                )
            else:
                reasons.append(
                    f"{label} 不支持（观测 t={ev.observed_at:g}，"
                    f"可信度 {ev.credibility:.2f}）"
                )
        for source, old_seq, old_t, new_seq, new_t in superseded:
            reasons.append(
                f"{SOURCE_LABELS[source]} 旧证据（序号 {old_seq}，t={old_t:g}）"
                f"已被新证据（序号 {new_seq}，t={new_t:g}）取代，不参与定位"
            )
        for source in stale_sources:
            reasons.append(f"{SOURCE_LABELS[source]} 证据已过期，权重衰减为 0，不参与定位")
        candidates.append(
            FusedCandidate(
                section=name,
                score=round(score, 3),
                support=tuple(SOURCE_LABELS[s] for s in support),
                opposition=tuple(SOURCE_LABELS[s] for s in opposition),
                reasons=tuple(reasons),
            )
        )

    return FusionResult(
        candidates=tuple(candidates),
        narrowed=narrowed,
        conflict=conflict,
        authoritative=tuple(SOURCE_LABELS[ev.source] for ev, _ in authoritative),
        superseded=tuple(SOURCE_LABELS[s] for s, _, _, _, _ in superseded),
        stale=tuple(SOURCE_LABELS[s] for s in stale_sources),
    )


# ---------------------------------------------------------------------------
# Builders: feed the four existing conclusions into Evidence.
# ---------------------------------------------------------------------------

def evidence_from_indicators(
    indicators: list[tuple[str, int, bool]],
    breaker_tripped: bool,
    telemetry_quality: float,
    observed_at: float,
    sequence: int,
    *,
    credibility: float | None = None,
) -> Evidence | None:
    """Fault-zone conclusion from fault indicators (faults.indicators)."""
    zone = infer_fault_zone(indicators, breaker_tripped)
    if not zone:
        return None
    asserted = sum(1 for _, _, state in indicators if state)
    cred = (
        credibility
        if credibility is not None
        else confidence(asserted, len(indicators), telemetry_quality)
    )
    sections = _dedupe(tuple(zone))
    return Evidence(
        source=INDICATOR,
        observed_at=observed_at,
        sequence=sequence,
        credibility=cred,
        sections=sections,
        basis=f"故障指示器判定区段 {sections[0]}（动作 {asserted}/{len(indicators)}）",
    )


def evidence_from_impedance(
    estimate: ImpedanceEstimate,
    sections: list[tuple[str, float, float]],
    observed_at: float,
    sequence: int,
    *,
    credibility: float = 0.8,
) -> Evidence:
    """Distance-based conclusion from impedance reactance (faults.location)."""
    containing = [
        name for name, start_km, end_km in sections
        if start_km <= estimate.distance_km <= end_km
    ]
    if containing:
        names = _dedupe(tuple(containing))
        basis = f"阻抗测距 {estimate.distance_km:.2f} km 位于区段 {names[0]} 内"
    else:
        ranked = rank_sections(estimate, sections)
        names = (ranked[0],)
        basis = f"阻抗测距 {estimate.distance_km:.2f} km 最近区段 {names[0]}"
    return Evidence(
        source=IMPEDANCE,
        observed_at=observed_at,
        sequence=sequence,
        credibility=credibility,
        sections=names,
        basis=basis,
    )


def topology_candidate_sections(
    snapshot: FeederSnapshot,
    indicators: list[tuple[str, int, bool]],
    breaker_tripped: bool,
) -> tuple[str, ...]:
    """Sections the fault can occupy given indicator states and breaker trip.

    The fault is on or downstream of the last asserted indicator and does
    not cross the first cleared indicator or an open point. With no asserted
    indicator the whole energized feeder is candidate.
    """
    if not breaker_tripped:
        return ()
    branches = list(snapshot.closed_branches())
    adj: dict[str, list[tuple[str, str]]] = {}
    for br in branches:
        adj.setdefault(br.source, []).append((br.branch_id, br.target))

    asserted = [(name, order) for name, order, state in indicators if state]
    cleared = {name for name, _, state in indicators if not state}

    if asserted:
        last_name, _ = max(asserted, key=lambda item: item[1])
        last_br = next((br for br in branches if br.branch_id == last_name), None)
        if last_br is None:
            return ()
        start = last_br.target
        out = [last_br.branch_id]  # fault is on or beyond the last asserted section
    else:
        targets = {br.target for br in branches}
        heads = sorted({br.source for br in branches} - targets)
        if not heads:
            return ()
        start = heads[0]
        out = []

    seen: set[str] = set()
    stack = [start]
    while stack:
        bus = stack.pop()
        if bus in seen:
            continue
        seen.add(bus)
        for branch_id, target in adj.get(bus, []):
            if branch_id in cleared:
                continue  # do not cross a cleared indicator
            out.append(branch_id)
            stack.append(target)
    return tuple(out)


def evidence_from_topology(
    snapshot: FeederSnapshot,
    indicators: list[tuple[str, int, bool]],
    breaker_tripped: bool,
    observed_at: float,
    sequence: int,
    *,
    credibility: float = 0.7,
) -> Evidence | None:
    """Topology-position conclusion (topology walk bounded by indicators)."""
    sections = topology_candidate_sections(snapshot, indicators, breaker_tripped)
    if not sections:
        return None
    return Evidence(
        source=TOPOLOGY,
        observed_at=observed_at,
        sequence=sequence,
        credibility=credibility,
        sections=sections,
        basis=f"拓扑位置可达区段 {len(sections)} 个（限于动作指示器与开断点之间）",
    )


def evidence_from_telemetry(
    sample: TelemetrySample,
    section: str,
    sequence: int | None = None,
    *,
    now_s: float | None = None,
    max_age_s: float = DEFAULT_MAX_AGE_S,
) -> Evidence | None:
    """Telemetry-quality conclusion for a section the sample refers to."""
    if sample.quality not in {"good", "suspect"}:
        return None
    if now_s is not None:
        cred = quality_score(sample, now_s, max_age_s)
    else:
        cred = 1.0 if sample.quality == "good" else 0.6
    if cred <= 0.0:
        return None
    return Evidence(
        source=TELEMETRY,
        observed_at=sample.timestamp_s,
        sequence=sequence if sequence is not None else sample.sequence,
        credibility=cred,
        sections=(section,),
        basis=f"遥测质量 {sample.quality}，指向区段 {section}",
    )
