from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict

from relaygrid.models import TelemetrySample
from relaygrid.faults.indicators import infer_fault_zone, confidence as fi_confidence
from relaygrid.faults.location import locate_reactance, rank_sections
from relaygrid.topology.radiality import energized_edges
from relaygrid.telemetry.quality import quality_score

# Source weights encode how decisive each evidence family can be.  Telemetry is
# deliberately the weakest: it supports physical judgments but must not lead
# crews to a different section on the strength of an old reading.
SOURCE_WEIGHT={"indicator":0.9,"impedance":0.8,"topology":0.75,"telemetry":0.6}
UNKNOWN_SOURCE_WEIGHT=0.2

# Indicator, impedance and topology produce one section conclusion per fault
# round, so a newer statement from them supersedes older statements on any
# section.  Telemetry can legitimately speak about several sections at once.
SINGLE_VALUED={"indicator","impedance","topology"}

STATUS_ACCEPTED="accepted"
STATUS_BELOW_THRESHOLD="below_threshold"
STATUS_SUPERSEDED="superseded"
STATUS_STALE_BLOCKED="stale_blocked"


@dataclass(frozen=True)
class LocationEvidence:
    source: str
    section: str
    timestamp_s: float
    confidence: float
    detail: str=""


@dataclass(frozen=True)
class BasisEntry:
    source: str
    section: str
    timestamp_s: float
    confidence: float
    weight: float
    score: float
    status: str
    detail: str
    note: str=""


@dataclass(frozen=True)
class Candidate:
    section: str
    score: float
    rank: int
    conflict: bool
    basis_indices: tuple[int,...]


@dataclass(frozen=True)
class UnifiedLocation:
    primary: str|None
    candidates: tuple[Candidate,...]
    excluded_sections: tuple[str,...]
    ledger: tuple[BasisEntry,...]

    def bases_for(self, section: str) -> tuple[BasisEntry,...]:
        keep=set()
        for c in self.candidates:
            if c.section==section: keep.update(c.basis_indices)
        return tuple(self.ledger[i] for i in sorted(keep))


def _clamp(x: float) -> float:
    return min(1.0,max(0.0,x))


def fuse_location(evidence: list[LocationEvidence], *, min_confidence: float=0.2,
                  prune_gap: float=0.15) -> UnifiedLocation:
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be within [0,1]")
    if prune_gap < 0.0:
        raise ValueError("prune_gap cannot be negative")
    n=len(evidence)
    status: list[str|None]=[None]*n
    note: list[str]=[""]*n

    # --- same-source supersession, independent of arrival order -------------
    winner: dict[tuple, int]={}
    def dedup_key(i: int):
        e=evidence[i]
        return (e.source,) if e.source in SINGLE_VALUED else (e.source,e.section)
    for i,e in enumerate(evidence):
        k=dedup_key(i)
        if k not in winner:
            winner[k]=i; continue
        j=winner[k]; prev=evidence[j]
        if (e.timestamp_s,e.confidence) > (prev.timestamp_s,prev.confidence):
            status[j]=STATUS_SUPERSEDED
            note[j]=f"superseded by newer {e.source} evidence @t={e.timestamp_s} (section {e.section})"
            winner[k]=i
        else:
            status[i]=STATUS_SUPERSEDED
            note[i]=f"superseded by newer {prev.source} evidence @t={prev.timestamp_s} (section {prev.section})"

    # --- confidence threshold ------------------------------------------------
    for i,e in enumerate(evidence):
        if status[i] is None and _clamp(e.confidence) < min_confidence:
            status[i]=STATUS_BELOW_THRESHOLD
            note[i]=f"confidence {_clamp(e.confidence):.3f} below minimum {min_confidence:.3f}"

    # --- old telemetry must not override a newer physical judgment -----------
    accepted_physical=[i for i,e in enumerate(evidence)
                       if status[i] is None and e.source!="telemetry"]
    newest_judgment=max((evidence[i].timestamp_s for i in accepted_physical), default=None)
    judged_sections={evidence[i].section for i in accepted_physical}
    for i,e in enumerate(evidence):
        if status[i] is not None or e.source!="telemetry" or newest_judgment is None:
            continue
        if e.timestamp_s < newest_judgment and e.section not in judged_sections:
            status[i]=STATUS_STALE_BLOCKED
            note[i]=(f"telemetry @t={e.timestamp_s} is older than newest physical judgment "
                     f"@t={newest_judgment} and points outside it")
        else:
            status[i]=STATUS_ACCEPTED
    for i in range(n):
        if status[i] is None: status[i]=STATUS_ACCEPTED

    # --- aggregate accepted evidence per section ----------------------------
    scores: dict[str,float]=defaultdict(float)
    times: dict[str,float]={}
    bases: dict[str,list[int]]=defaultdict(list)
    physical: set[str]=set()
    fresh_telemetry: set[str]=set()
    for i,e in enumerate(evidence):
        if status[i]!=STATUS_ACCEPTED: continue
        w=SOURCE_WEIGHT.get(e.source,UNKNOWN_SOURCE_WEIGHT)
        scores[e.section]+=w*_clamp(e.confidence)
        times[e.section]=max(times.get(e.section,e.timestamp_s),e.timestamp_s)
        bases[e.section].append(i)
        if e.source!="telemetry":
            physical.add(e.section)
        elif newest_judgment is None or e.timestamp_s > newest_judgment:
            # Only telemetry newer than the newest physical judgment can raise
            # a genuine new conflict; older telemetry was already blocked.
            fresh_telemetry.add(e.section)

    ordered=sorted(scores, key=lambda s:(-scores[s],-times[s],s))
    if not ordered:
        retained=[]; excluded=[]
    else:
        best=scores[ordered[0]]
        def keep(s):
            if best-scores[s] <= prune_gap: return True
            if s in physical: return True            # physical conflict never narrowed away
            return s in fresh_telemetry             # newer telemetry may show new development
        retained=[s for s in ordered if keep(s)]
        excluded=[s for s in ordered if not keep(s)]
    conflicting=len(retained)>1

    candidates=tuple(Candidate(s,scores[s],rank+1,conflicting,tuple(bases[s]))
                     for rank,s in enumerate(retained))
    ledger=tuple(
        BasisEntry(e.source,e.section,e.timestamp_s,_clamp(e.confidence),
                   SOURCE_WEIGHT.get(e.source,UNKNOWN_SOURCE_WEIGHT),
                   SOURCE_WEIGHT.get(e.source,UNKNOWN_SOURCE_WEIGHT)*_clamp(e.confidence),
                   status[i],e.detail,note[i])
        for i,e in enumerate(evidence))
    return UnifiedLocation(retained[0] if retained else None,
                           candidates,tuple(excluded),ledger)


# ---------------------------------------------------------------------------
# Adapters turning the four separately-computed conclusions into evidence.
# ---------------------------------------------------------------------------

def indicator_evidence(indicators: list[tuple[str,int,bool]], breaker_tripped: bool,
                       timestamp_s: float, telemetry_quality: float=1.0
                       ) -> list[LocationEvidence]:
    if not breaker_tripped or not indicators:
        return []
    zones=infer_fault_zone(indicators,breaker_tripped)
    asserted=sum(1 for _,_,state in indicators if state)
    conf=fi_confidence(asserted,len(indicators),telemetry_quality)
    detail=f"fault indicators asserted {asserted}/{len(indicators)}"
    return [LocationEvidence("indicator",z,timestamp_s,conf,detail) for z in zones]


def impedance_evidence(measured_x_ohm: float, source_x_ohm: float,
                       line_x_ohm_per_km: float, line_length_km: float,
                       sections: list[tuple[str,float,float]], timestamp_s: float,
                       confidence: float=0.9) -> LocationEvidence|None:
    estimate=locate_reactance(measured_x_ohm,source_x_ohm,line_x_ohm_per_km,line_length_km)
    ranked=rank_sections(estimate,sections)
    if not ranked:
        return None
    detail=(f"reactance estimate distance={estimate.distance_km:.3f}km "
            f"fraction={estimate.fraction:.3f} residual={estimate.residual_ohm:.3f}ohm")
    return LocationEvidence("impedance",ranked[0],timestamp_s,confidence,detail)


def _energized_from_roots(snapshot, roots):
    """Energized edges when power flows from the given source/root buses.
    energized_edges alone is insufficient downstream of an open device."""
    adj: dict[str,list[tuple[str,str]]]={b:[] for b in snapshot.buses}
    for eid,a,b in energized_edges(snapshot):
        adj[a].append((b,eid)); adj[b].append((a,eid))
    seen=set(roots); stack=list(roots); live_edges=set()
    while stack:
        cur=stack.pop()
        for nxt,eid in adj.get(cur,()):
            live_edges.add(eid)
            if nxt not in seen:
                seen.add(nxt); stack.append(nxt)
    return live_edges,seen


def topology_evidence(before, after, section_edges: dict[str,str], timestamp_s: float,
                      roots, confidence: float=0.75) -> list[LocationEvidence]:
    """An edge energized before the opening, dead afterwards, whose source bus
    (relative to the before snapshot) remains energized is the fault boundary
    and maps to its protected section."""
    before_live,_=_energized_from_roots(before,roots)
    after_live,after_buses=_energized_from_roots(after,roots)
    out=[]
    for eid in sorted(before_live-after_live):
        # Orient the edge from the side that stays energized (the protected
        # side facing the source), so a mid-feeder device is the boundary.
        a,b=before.branches[eid].source,before.branches[eid].target
        source_bus,target_bus=(a,b) if a in after_buses and b not in after_buses else (b,a)
        if source_bus not in after_buses or target_bus in after_buses:
            continue
        section=section_edges.get(eid)
        if section is not None:
            out.append(LocationEvidence("topology",section,timestamp_s,confidence,
                                        f"boundary edge {eid} de-energized, source bus {source_bus} alive"))
    return out


def telemetry_evidence(section: str, sample: TelemetrySample|None, now_s: float,
                       max_age_s: float, dead_threshold_v: float=0.0
                       ) -> LocationEvidence|None:
    if sample is None or sample.value > dead_threshold_v:
        return None
    conf=quality_score(sample,now_s,max_age_s)
    detail=(f"point {sample.point_id} dead-telemetry value={sample.value} "
            f"quality={sample.quality}")
    return LocationEvidence("telemetry",section,sample.timestamp_s,conf,detail)
