from __future__ import annotations

from relaygrid.faults.fusion import (
    DEFAULT_MAX_AGE_S,
    Evidence,
    evidence_from_impedance,
    evidence_from_indicators,
    evidence_from_telemetry,
    evidence_from_topology,
    fuse,
    topology_candidate_sections,
    IMPEDANCE,
    INDICATOR,
    TELEMETRY,
    TOPOLOGY,
)
from relaygrid.faults.location import ImpedanceEstimate
from relaygrid.models import Branch, Bus, FeederSnapshot, SwitchState, TelemetrySample


def ev(source, t, seq, cred, *sections):
    return Evidence(source, t, seq, cred, tuple(sections))


def test_unanimous_intersection_narrows_to_single_candidate():
    result = fuse(
        [
            ev(INDICATOR, 100, 1, 0.9, "bc"),
            ev(IMPEDANCE, 100, 1, 0.8, "bc"),
            ev(TOPOLOGY, 100, 1, 0.7, "ab", "bc"),
        ]
    )
    assert result.narrowed is True
    assert result.conflict is False
    assert [c.section for c in result.candidates] == ["bc"]
    assert result.candidates[0].score == 1.0


def test_majority_narrows_when_intersection_empty():
    result = fuse(
        [
            ev(INDICATOR, 100, 1, 0.9, "bc"),
            ev(IMPEDANCE, 100, 1, 0.8, "bc"),
            ev(TOPOLOGY, 100, 1, 0.7, "cd"),
        ]
    )
    assert result.narrowed is True
    assert [c.section for c in result.candidates] == ["bc"]


def test_conflict_keeps_multiple_candidates():
    result = fuse(
        [
            ev(INDICATOR, 100, 1, 0.9, "bc"),
            ev(IMPEDANCE, 100, 1, 0.9, "cd"),
        ]
    )
    assert result.conflict is True
    assert result.narrowed is False
    assert {c.section for c in result.candidates} == {"bc", "cd"}
    assert all(0 < c.score < 1 for c in result.candidates)


def test_old_telemetry_does_not_override_new_judgment():
    # old telemetry (t=0) points at ab; new indicator (t=100) points at bc.
    # Recency decay must make the old telemetry lose, not override.
    result = fuse(
        [
            ev(TELEMETRY, 0, 1, 1.0, "ab"),
            ev(INDICATOR, 100, 2, 0.9, "bc"),
        ],
        max_age_s=DEFAULT_MAX_AGE_S,
    )
    assert result.narrowed is True
    assert [c.section for c in result.candidates] == ["bc"]
    # the old telemetry still appears in the basis trail as opposition
    bc = result.candidates[0]
    assert any("遥测质量" in r and "不支持" in r for r in bc.reasons)


def test_same_source_old_sequence_is_superseded():
    result = fuse(
        [
            ev(TELEMETRY, 90, 1, 1.0, "ab"),
            ev(TELEMETRY, 100, 5, 1.0, "bc"),
        ]
    )
    assert result.superseded == ("遥测质量",)
    assert result.authoritative == ("遥测质量",)
    assert [c.section for c in result.candidates] == ["bc"]
    # superseded evidence must not appear as a supporting reason
    bc = result.candidates[0]
    assert not any("支持" in r and "ab" in r for r in bc.reasons)


def test_each_candidate_lists_basis():
    result = fuse(
        [
            ev(INDICATOR, 100, 1, 0.9, "bc"),
            ev(IMPEDANCE, 100, 1, 0.8, "cd"),
        ]
    )
    for cand in result.candidates:
        assert cand.reasons  # every candidate has a basis trail
        support = [r for r in cand.reasons if "支持" in r]
        oppose = [r for r in cand.reasons if "不支持" in r]
        assert support and oppose


def test_stale_evidence_is_excluded_and_recorded():
    result = fuse(
        [
            ev(TELEMETRY, 0, 1, 1.0, "ab"),
            ev(INDICATOR, 1000, 2, 0.9, "bc"),
        ],
        max_age_s=100.0,
    )
    # telemetry is 1000s older than the newest judgment -> freshness 0
    assert result.stale == ("遥测质量",)
    assert [c.section for c in result.candidates] == ["bc"]
    assert any("已过期" in r for r in result.candidates[0].reasons)


def test_credibility_weights_candidates():
    # 3-way conflict: no single source exceeds quorum, so all are kept and
    # credibility ordering shows up in the scores.
    result = fuse(
        [
            ev(INDICATOR, 100, 1, 0.5, "bc"),
            ev(IMPEDANCE, 100, 1, 0.4, "cd"),
            ev(TOPOLOGY, 100, 1, 0.3, "ad"),
        ]
    )
    assert result.conflict is True
    scores = {c.section: c.score for c in result.candidates}
    assert scores["bc"] > scores["cd"] > scores["ad"]


def test_evidence_validation():
    import pytest

    with pytest.raises(ValueError):
        ev(INDICATOR, 100, 1, 1.5, "bc")  # credibility out of range
    with pytest.raises(ValueError):
        Evidence(INDICATOR, 100, 1, 0.9, ())  # no sections
    with pytest.raises(ValueError):
        Evidence("bogus", 100, 1, 0.9, ("bc",))  # unknown source


def test_builders_feed_fuse():
    # indicators: ab and bc asserted, no cleared downstream -> fault past bc
    indicators = [("ab", 1, True), ("bc", 2, True)]
    snapshot = FeederSnapshot(
        buses={x: Bus(x, 12.47) for x in "ABCD"},
        branches={
            "ab": Branch("ab", "A", "B", 0.1, 0.1, 300),
            "bc": Branch("bc", "B", "C", 0.1, 0.1, 300),
            "cd": Branch("cd", "C", "D", 0.1, 0.1, 300, True),
        },
        switch_state={"cd": SwitchState.OPEN},
    )
    ind = evidence_from_indicators(indicators, True, 0.9, 100, 1)
    imp = evidence_from_impedance(
        ImpedanceEstimate(15.0, 0.75, 0.0),
        [("ab", 0.0, 10.0), ("bc", 10.0, 20.0), ("cd", 20.0, 30.0)],
        100,
        1,
    )
    topo = evidence_from_topology(snapshot, indicators, True, 100, 1)
    tel = evidence_from_telemetry(TelemetrySample("p1", 100, 1, 1.0, "good"), "bc")
    assert ind and imp and topo and tel
    result = fuse([ind, imp, topo, tel])
    assert result.narrowed is True
    assert result.candidates[0].section == "bc"


def test_topology_bounded_by_cleared_indicator():
    snapshot = FeederSnapshot(
        buses={x: Bus(x, 12.47) for x in "ABCD"},
        branches={
            "ab": Branch("ab", "A", "B", 0.1, 0.1, 300),
            "bc": Branch("bc", "B", "C", 0.1, 0.1, 300),
            "cd": Branch("cd", "C", "D", 0.1, 0.1, 300),
        },
    )
    # bc asserted, cd cleared -> fault on bc (do not cross cd)
    sections = topology_candidate_sections(
        snapshot, [("bc", 1, True), ("cd", 2, False)], True
    )
    assert sections == ("bc",)


def test_topology_no_asserted_indicator_whole_feeder():
    snapshot = FeederSnapshot(
        buses={x: Bus(x, 12.47) for x in "ABCD"},
        branches={
            "ab": Branch("ab", "A", "B", 0.1, 0.1, 300),
            "bc": Branch("bc", "B", "C", 0.1, 0.1, 300),
        },
    )
    sections = topology_candidate_sections(snapshot, [], True)
    assert sections == ("ab", "bc")


def test_topology_open_point_stops_downstream():
    snapshot = FeederSnapshot(
        buses={x: Bus(x, 12.47) for x in "ABCD"},
        branches={
            "ab": Branch("ab", "A", "B", 0.1, 0.1, 300),
            "bc": Branch("bc", "B", "C", 0.1, 0.1, 300),
            "cd": Branch("cd", "C", "D", 0.1, 0.1, 300, True),
        },
        switch_state={"cd": SwitchState.OPEN},
    )
    # bc asserted, cd open -> fault on bc (open point stops reach)
    sections = topology_candidate_sections(
        snapshot, [("bc", 1, True)], True
    )
    assert sections == ("bc",)


def test_topology_breaker_not_tripped():
    snapshot = FeederSnapshot(
        buses={"A": Bus("A", 12.47)},
        branches={"ab": Branch("ab", "A", "B", 0.1, 0.1, 300)},
    )
    assert topology_candidate_sections(snapshot, [], False) == ()
