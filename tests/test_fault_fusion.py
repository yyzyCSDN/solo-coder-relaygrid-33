from __future__ import annotations

from relaygrid.models import Bus, Branch, FeederSnapshot, SwitchState, TelemetrySample
from relaygrid.faults.fusion import (
    LocationEvidence, fuse_location,
    indicator_evidence, impedance_evidence, topology_evidence, telemetry_evidence,
    STATUS_ACCEPTED, STATUS_BELOW_THRESHOLD, STATUS_SUPERSEDED, STATUS_STALE_BLOCKED,
)


def E(source, section, t, conf=0.9, detail=""):
    return LocationEvidence(source, section, t, conf, detail)


# --------------------------------------------------------------------------
# Fusion rules
# --------------------------------------------------------------------------

def test_consistent_evidence_narrows_to_single_candidate():
    ev=[E("indicator","s2",100,.9,"FPI"),E("impedance","s2",99,.8,"X"),
        E("topology","s2",98,.75,"topo")]
    r=fuse_location(ev)
    assert r.primary=="s2"
    assert len(r.candidates)==1
    assert r.candidates[0].conflict is False
    assert r.excluded_sections==()


def test_conflicting_evidence_keeps_multiple_candidates():
    ev=[E("indicator","s2",100,.9),E("impedance","s3",99,.85)]
    r=fuse_location(ev)
    assert r.primary=="s2"
    assert {c.section for c in r.candidates}=={"s2","s3"}
    assert all(c.conflict for c in r.candidates)


def test_low_scoring_candidate_is_narrowed_out():
    ev=[E("indicator","s2",100,.95),E("impedance","s2",100,.9),
        E("telemetry","s4",100,.5)]
    r=fuse_location(ev,prune_gap=.15)
    assert r.primary=="s2"
    assert [c.section for c in r.candidates]==["s2"]
    assert r.excluded_sections==("s4",)


def test_old_telemetry_cannot_redirect_to_other_section():
    ev=[E("indicator","s2",100,.9),E("telemetry","s5",50,.9)]
    r=fuse_location(ev)
    assert r.primary=="s2"
    assert {c.section for c in r.candidates}=={"s2"}
    blocked=[b for b in r.ledger if b.status==STATUS_STALE_BLOCKED]
    assert len(blocked)==1 and blocked[0].section=="s5"
    assert "older than newest physical judgment" in blocked[0].note


def test_new_telemetry_conflicting_keeps_both_candidates():
    ev=[E("indicator","s2",100,.9),E("telemetry","s5",150,.9)]
    r=fuse_location(ev)
    assert {c.section for c in r.candidates}=={"s2","s5"}
    assert all(b.status!=STATUS_STALE_BLOCKED for b in r.ledger)


def test_old_telemetry_supporting_same_section_is_accepted():
    ev=[E("indicator","s2",100,.9),E("telemetry","s2",50,.9)]
    r=fuse_location(ev)
    tel=[b for b in r.ledger if b.source=="telemetry"]
    assert tel[0].status==STATUS_ACCEPTED
    assert r.primary=="s2"


def test_same_source_newer_round_supersedes_old_round_any_section():
    ev=[E("impedance","s5",50,.9),E("impedance","s2",100,.9)]
    r=fuse_location(ev)
    old=[b for b in r.ledger if b.status==STATUS_SUPERSEDED]
    assert len(old)==1 and old[0].section=="s5"
    assert r.primary=="s2"


def test_arrival_order_independent_dedup():
    ev=[E("impedance","s2",100,.9),E("impedance","s5",50,.9)]  # newest first
    r=fuse_location(ev)
    assert r.primary=="s2"
    assert {b.section for b in r.ledger if b.status==STATUS_SUPERSEDED}=={"s5"}


def test_multi_section_telemetry_not_deduped():
    ev=[E("telemetry","s2",50,.9),E("telemetry","s3",50,.9)]
    r=fuse_location(ev)
    assert {c.section for c in r.candidates}=={"s2","s3"}


def test_below_threshold_dropped_but_listed_in_ledger():
    ev=[E("indicator","s2",100,.9),E("telemetry","s9",100,.1)]
    r=fuse_location(ev,min_confidence=.2)
    dropped=[b for b in r.ledger if b.status==STATUS_BELOW_THRESHOLD]
    assert len(dropped)==1 and dropped[0].section=="s9"
    assert r.primary=="s2"


def test_ledger_lists_every_input_with_basis_for_each_candidate():
    ev=[E("indicator","s2",100,.9,"FPI detail"),E("impedance","s3",99,.8,"X detail"),
        E("telemetry","s9",90,.05)]
    r=fuse_location(ev)
    assert len(r.ledger)==len(ev)
    cand_s2=next(c for c in r.candidates if c.section=="s2")
    bases=r.bases_for("s2")
    assert len(bases)==1 and bases[0].source=="indicator" and bases[0].detail=="FPI detail"
    assert all(0.0<=b.confidence<=1.0 for b in r.ledger)
    assert cand_s2.rank==1
    # scores = weight * confidence
    b=bases[0]
    assert abs(b.score-b.weight*b.confidence)<1e-12


def test_no_credible_evidence_returns_empty_location():
    r=fuse_location([E("indicator","s2",100,.05)],min_confidence=.2)
    assert r.primary is None and r.candidates==()
    assert r.ledger[0].status==STATUS_BELOW_THRESHOLD


def test_tie_break_newest_section_then_name():
    ev=[E("telemetry","old_s",90,.9),E("telemetry","new_s",100,.9)]
    r=fuse_location(ev)
    assert r.candidates[0].section=="new_s"
    ev2=[E("telemetry","a",100,.9),E("telemetry","b",100,.9)]
    assert fuse_location(ev2).candidates[0].section=="a"


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------

def test_indicator_adapter():
    inds=[("fpi1",1,True),("fpi2",2,True),("fpi3",3,False)]
    out=indicator_evidence(inds,True,timestamp_s=100,telemetry_quality=1.0)
    assert len(out)==1 and out[0].source=="indicator" and out[0].section=="fpi3"
    assert out[0].confidence<1.0  # 2/3 asserted
    assert indicator_evidence(inds,False,100)==[]


def test_impedance_adapter_picks_containing_section():
    sections=[("s1",0,5),("s2",5,15),("s3",15,20)]
    ev=impedance_evidence(1.5,.5,.1,20,sections,timestamp_s=60)
    assert ev.section=="s2" and ev.source=="impedance"
    assert "residual" in ev.detail


def feeder():
    buses={x:Bus(x,12.47) for x in "ABCDE"}
    branches={
        "ab":Branch("ab","A","B",.1,.1,300),
        "bc":Branch("bc","B","C",.1,.1,300,switchable=True),
        "cd":Branch("cd","C","D",.1,.1,300),
        "de":Branch("de","D","E",.1,.1,300),
    }
    return FeederSnapshot(buses,branches,{})


def test_topology_adapter_finds_boundary_edge():
    before=feeder()
    after=feeder().copy()
    after.switch_state["bc"]=SwitchState.OPEN
    out=topology_evidence(before,after,{"bc":"section-bc","cd":"section-cd"},
                          timestamp_s=80,roots=["A"])
    assert [e.section for e in out]==["section-bc"]
    assert out[0].source=="topology"


def test_telemetry_adapter_uses_freshness_quality():
    good=TelemetrySample("mD",90,1,0.0,"good")
    assert telemetry_evidence("s2",good,now_s=100,max_age_s=120).confidence>0
    stale=telemetry_evidence("s2",good,now_s=300,max_age_s=120)
    assert stale is None or stale.confidence==0.0
    alive=TelemetrySample("mD",90,1,7.2,"good")
    assert telemetry_evidence("s2",alive,100,120) is None


def test_end_to_end_unified_result_for_crew():
    sections=[("s1",0,5),("s2",5,15),("s3",15,20)]
    inds=[("s1",1,True),("s2",2,True),("s3",3,False)]
    ev=[]
    ev+=indicator_evidence(inds,True,100,telemetry_quality=1.0)
    ev.append(impedance_evidence(1.5,.5,.1,20,sections,99))
    before=feeder()
    after=feeder().copy(); after.switch_state["bc"]=SwitchState.OPEN
    ev+=topology_evidence(before,after,{"bc":"s2"},98,roots=["A"])
    # stale telemetry pointing at another section must be ignored
    ev.append(telemetry_evidence("s1",TelemetrySample("mA",20,1,0.0,"good"),
                                 now_s=100,max_age_s=120))
    r=fuse_location(ev)
    # impedance + topology agree on s2; indicators infer the un-asserted FPI
    # zone s3, a genuine physical conflict that must remain for the crew.
    assert r.primary=="s2"
    assert {c.section for c in r.candidates}=={"s2","s3"}
    assert all(c.conflict for c in r.candidates)
    blocked=[b for b in r.ledger if b.status==STATUS_STALE_BLOCKED]
    assert blocked and blocked[0].section=="s1"
