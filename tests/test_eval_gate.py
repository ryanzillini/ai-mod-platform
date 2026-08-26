"""CI routing gate: 20/20 on replay, and a real policy regression fails it."""

from src.eval_gate import load_golden, load_replay_records, run_routing_eval
from src.policy import DecisionPolicy


def test_replay_fixture_covers_every_golden_id():
    _version, examples = load_golden()
    records = load_replay_records()
    assert {ex["id"] for ex in examples} == set(records)


def test_ci_eval_passes_on_current_policy():
    report = run_routing_eval()
    assert report.n == 20
    assert report.hits == 20
    assert report.passed
    assert report.action_misses == []
    assert report.interrupt_misses == []
    escalate_ids = {
        row.example_id for row in report.rows if row.expected_action == "ESCALATE"
    }
    interrupted = {row.example_id for row in report.rows if row.interrupted}
    assert escalate_ids == interrupted


def test_eval_fails_if_medical_policy_is_removed():
    policy = DecisionPolicy(always_escalate_if_medical_advice=False)
    report = run_routing_eval(policy=policy)
    assert report.passed is False
    assert "gd-015" in report.action_misses
    gd015 = next(row for row in report.rows if row.example_id == "gd-015")
    assert gd015.predicted_action == "BLOCK"
    assert gd015.expected_action == "ESCALATE"


def test_eval_fails_if_low_confidence_threshold_is_zero():
    policy = DecisionPolicy(confidence_threshold=0.0)
    report = run_routing_eval(policy=policy)
    low_conf_only = {"gd-012", "gd-014", "gd-016", "gd-020"}
    assert report.passed is False
    assert low_conf_only <= set(report.action_misses)
    for example_id in low_conf_only:
        row = next(r for r in report.rows if r.example_id == example_id)
        assert row.predicted_action == "BLOCK"
        assert row.interrupted is False


def test_eval_fails_if_operational_pii_rule_is_removed():
    policy = DecisionPolicy(always_escalate_if_operational_pii=False)
    report = run_routing_eval(policy=policy)
    assert "gd-013" in report.action_misses
    gd013 = next(row for row in report.rows if row.example_id == "gd-013")
    assert gd013.predicted_action == "ALLOW"
