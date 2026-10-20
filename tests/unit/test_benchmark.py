from __future__ import annotations

from benchmark.run_experiment import run_benchmark


def test_benchmark_compares_three_conditions_and_excludes_scope_mismatches():
    report = run_benchmark()

    assert set(report["metrics"]) == {"baseline", "vector-only", "full-engine"}
    assert report["metrics"]["baseline"]["mrr"] == 0
    assert report["metrics"]["full-engine"]["top1_relevance_rate"] == 1
    assert report["metrics"]["full-engine"]["harmful_advice_rate_at_1"] == 0
    assert report["metrics"]["full-engine"]["scenario_count"] == 24
    assert report["limitations"]
