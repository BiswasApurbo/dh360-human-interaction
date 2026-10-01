"""Unit tests for offline face verification evaluation and threshold calibration."""

import numpy as np
import pytest

from dh360_hi.vision.evaluation import (
    PairComparison,
    compute_distribution,
    evaluate_pairs,
    format_evaluation_report,
)


def test_compute_distribution_statistics() -> None:
    """Verify compute_distribution accurately computes statistical moments."""
    data = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
    dist = compute_distribution(data)

    assert dist.count == 5
    assert dist.mean == pytest.approx(0.6, abs=1e-4)
    assert dist.min == pytest.approx(0.2, abs=1e-4)
    assert dist.max == pytest.approx(1.0, abs=1e-4)
    assert dist.p50 == pytest.approx(0.6, abs=1e-4)


def test_compute_distribution_empty() -> None:
    """Verify compute_distribution handles empty arrays without error."""
    dist = compute_distribution(np.array([], dtype=np.float32))
    assert dist.count == 0
    assert dist.mean == 0.0


def test_evaluate_pairs_separable_distributions() -> None:
    """Verify evaluate_pairs identifies 0% EER for perfectly separated genuine and impostor sets."""
    comparisons = [
        # Genuine pairs (high similarity: 0.85 - 0.95)
        PairComparison(
            pair_id="g1",
            label1="a",
            label2="a",
            is_genuine=True,
            cosine_similarity=0.90,
            cosine_distance=0.10,
        ),
        PairComparison(
            pair_id="g2",
            label1="a",
            label2="a",
            is_genuine=True,
            cosine_similarity=0.95,
            cosine_distance=0.05,
        ),
        PairComparison(
            pair_id="g3",
            label1="b",
            label2="b",
            is_genuine=True,
            cosine_similarity=0.85,
            cosine_distance=0.15,
        ),
        # Impostor pairs (low similarity: 0.10 - 0.20)
        PairComparison(
            pair_id="i1",
            label1="a",
            label2="b",
            is_genuine=False,
            cosine_similarity=0.15,
            cosine_distance=0.85,
        ),
        PairComparison(
            pair_id="i2",
            label1="a",
            label2="c",
            is_genuine=False,
            cosine_similarity=0.20,
            cosine_distance=0.80,
        ),
        PairComparison(
            pair_id="i3",
            label1="b",
            label2="c",
            is_genuine=False,
            cosine_similarity=0.10,
            cosine_distance=0.90,
        ),
    ]

    report = evaluate_pairs(comparisons, model_name="TestModel", model_version="v1.0")

    assert report.total_pairs == 6
    assert report.genuine_scores.count == 3
    assert report.impostor_scores.count == 3
    assert report.genuine_scores.mean == pytest.approx(0.90, abs=1e-2)
    assert report.impostor_scores.mean == pytest.approx(0.15, abs=1e-2)
    assert report.separation_gap > 0.70
    assert report.eer == pytest.approx(0.0, abs=1e-3)
    assert 0.20 < report.optimal_threshold <= 0.85


def test_evaluate_pairs_empty_list() -> None:
    """Verify evaluate_pairs handles empty input gracefully."""
    report = evaluate_pairs([])
    assert report.total_pairs == 0
    assert report.eer == 0.0
    assert len(report.operating_points) == 0


def test_format_evaluation_report() -> None:
    """Verify format_evaluation_report returns well-formed markdown."""
    comparisons = [
        PairComparison(
            pair_id="g1",
            label1="a",
            label2="a",
            is_genuine=True,
            cosine_similarity=0.88,
            cosine_distance=0.12,
        ),
        PairComparison(
            pair_id="i1",
            label1="a",
            label2="b",
            is_genuine=False,
            cosine_similarity=0.22,
            cosine_distance=0.78,
        ),
    ]
    report = evaluate_pairs(comparisons)
    formatted = format_evaluation_report(report)

    assert "OFFLINE VERIFICATION & CALIBRATION" in formatted
    assert "Genuine Similarity" in formatted
    assert "Impostor Similarity" in formatted
    assert "Equal Error Rate (EER)" in formatted
