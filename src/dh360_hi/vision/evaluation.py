"""Offline evaluation harness for face verification and threshold calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dh360_hi.vision.embedder import compute_cosine_distance, compute_cosine_similarity

__all__ = [
    "EvaluationReport",
    "PairComparison",
    "ScoreDistribution",
    "ThresholdPoint",
    "compute_cosine_distance",
    "compute_cosine_similarity",
    "compute_distribution",
    "evaluate_pairs",
    "format_evaluation_report",
]


@dataclass(frozen=True)
class PairComparison:
    """Represents a pairwise similarity comparison between two face embeddings."""

    pair_id: str
    label1: str
    label2: str
    is_genuine: bool
    cosine_similarity: float
    cosine_distance: float


@dataclass(frozen=True)
class ThresholdPoint:
    """Verification metrics at a specific decision threshold."""

    threshold: float
    far: float  # False Accept Rate: Impostors accepted / Total Impostors
    frr: float  # False Reject Rate: Genuine rejected / Total Genuine
    tpr: float  # True Positive Rate: 1.0 - FRR
    accuracy: float
    f1_score: float


@dataclass(frozen=True)
class ScoreDistribution:
    """Statistical summary of similarity scores."""

    count: int
    mean: float
    std: float
    min: float
    max: float
    p25: float
    p50: float
    p75: float


@dataclass(frozen=True)
class EvaluationReport:
    """Comprehensive offline evaluation report across genuine and impostor pairs."""

    model_name: str
    model_version: str
    total_pairs: int
    genuine_scores: ScoreDistribution
    impostor_scores: ScoreDistribution
    eer: float  # Equal Error Rate
    eer_threshold: float
    optimal_threshold: float  # Max accuracy / balanced threshold
    operating_points: list[ThresholdPoint]

    @property
    def separation_gap(self) -> float:
        """Distance between genuine mean and impostor mean."""
        return self.genuine_scores.mean - self.impostor_scores.mean


def compute_distribution(scores: np.ndarray) -> ScoreDistribution:
    """Compute summary statistics for a collection of similarity scores."""
    if len(scores) == 0:
        return ScoreDistribution(
            count=0, mean=0.0, std=0.0, min=0.0, max=0.0, p25=0.0, p50=0.0, p75=0.0
        )
    return ScoreDistribution(
        count=len(scores),
        mean=float(np.mean(scores)),
        std=float(np.std(scores)),
        min=float(np.min(scores)),
        max=float(np.max(scores)),
        p25=float(np.percentile(scores, 25)),
        p50=float(np.percentile(scores, 50)),
        p75=float(np.percentile(scores, 75)),
    )


def evaluate_pairs(
    comparisons: list[PairComparison],
    model_name: str = "SFace",
    model_version: str = "2021dec",
    threshold_step: float = 0.01,
) -> EvaluationReport:
    """Evaluate pairwise verification metrics across thresholds to calibrate FAR/FRR and EER."""
    if not comparisons:
        empty_dist = ScoreDistribution(
            count=0, mean=0.0, std=0.0, min=0.0, max=0.0, p25=0.0, p50=0.0, p75=0.0
        )
        return EvaluationReport(
            model_name=model_name,
            model_version=model_version,
            total_pairs=0,
            genuine_scores=empty_dist,
            impostor_scores=empty_dist,
            eer=0.0,
            eer_threshold=0.0,
            optimal_threshold=0.0,
            operating_points=[],
        )

    genuine_sims = np.array([c.cosine_similarity for c in comparisons if c.is_genuine])
    impostor_sims = np.array([c.cosine_similarity for c in comparisons if not c.is_genuine])

    gen_dist = compute_distribution(genuine_sims)
    imp_dist = compute_distribution(impostor_sims)

    thresholds = np.arange(0.0, 1.0 + threshold_step / 2.0, threshold_step)
    operating_points: list[ThresholdPoint] = []

    best_acc = -1.0
    best_thresh = 0.5
    min_diff_far_frr = float("inf")
    eer_val = 0.0
    eer_thresh = 0.5

    n_gen = max(len(genuine_sims), 1)
    n_imp = max(len(impostor_sims), 1)

    for thresh in thresholds:
        # Prediction: similarity >= thresh -> Same Person (Genuine)
        tp = np.sum(genuine_sims >= thresh) if len(genuine_sims) > 0 else 0
        fn = n_gen - tp
        fp = np.sum(impostor_sims >= thresh) if len(impostor_sims) > 0 else 0
        tn = n_imp - fp

        far = float(fp / n_imp)
        frr = float(fn / n_gen)
        tpr = float(tp / n_gen)
        accuracy = float((tp + tn) / (n_gen + n_imp))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = float(2 * precision * recall / max(precision + recall, 1e-6))

        point = ThresholdPoint(
            threshold=float(thresh),
            far=far,
            frr=frr,
            tpr=tpr,
            accuracy=accuracy,
            f1_score=f1,
        )
        operating_points.append(point)

        if accuracy > best_acc:
            best_acc = accuracy
            best_thresh = float(thresh)

        diff = abs(far - frr)
        if diff < min_diff_far_frr:
            min_diff_far_frr = diff
            eer_val = (far + frr) / 2.0
            eer_thresh = float(thresh)

    return EvaluationReport(
        model_name=model_name,
        model_version=model_version,
        total_pairs=len(comparisons),
        genuine_scores=gen_dist,
        impostor_scores=imp_dist,
        eer=eer_val,
        eer_threshold=eer_thresh,
        optimal_threshold=best_thresh,
        operating_points=operating_points,
    )


def format_evaluation_report(report: EvaluationReport) -> str:
    """Format an EvaluationReport into human-readable markdown table."""
    gen_str = (
        f"Mean: {report.genuine_scores.mean:.4f} ± {report.genuine_scores.std:.4f} "
        f"[Min: {report.genuine_scores.min:.4f}, Max: {report.genuine_scores.max:.4f}]"
    )
    imp_str = (
        f"Mean: {report.impostor_scores.mean:.4f} ± {report.impostor_scores.std:.4f} "
        f"[Min: {report.impostor_scores.min:.4f}, Max: {report.impostor_scores.max:.4f}]"
    )
    title = f"OFFLINE VERIFICATION & CALIBRATION: {report.model_name} (v{report.model_version})"
    lines = [
        "================================================================================",
        title,
        "================================================================================",
        (
            f"  • Total Pairs Evaluated:   {report.total_pairs} "
            f"({report.genuine_scores.count} Genuine, {report.impostor_scores.count} Impostor)"
        ),
        f"  • Genuine Similarity:      {gen_str}",
        f"  • Impostor Similarity:     {imp_str}",
        f"  • Score Separation Gap:    {report.separation_gap:.4f}",
        "--------------------------------------------------------------------------------",
        (
            f"  • Equal Error Rate (EER):  {report.eer * 100.0:.2f}% "
            f"at threshold = {report.eer_threshold:.2f}"
        ),
        f"  • Optimal Acc Threshold:   {report.optimal_threshold:.2f}",
        "================================================================================",
        "Operating Points (FAR / FRR / Accuracy curve sampling):",
        (
            f"{'Threshold':<10} | {'FAR (%)':<10} | {'FRR (%)':<10} | "
            f"{'TPR (%)':<10} | {'Accuracy (%)':<12} | {'F1-Score':<10}"
        ),
        "-" * 72,
    ]

    # Sample key operating points
    sample_threshs = [0.20, 0.30, 0.36, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
    for pt in report.operating_points:
        if any(abs(pt.threshold - t) < 0.005 for t in sample_threshs):
            lines.append(
                f"{pt.threshold:<10.2f} | {pt.far * 100:<10.2f} | {pt.frr * 100:<10.2f} | "
                f"{pt.tpr * 100:<10.2f} | {pt.accuracy * 100:<12.2f} | {pt.f1_score:<10.4f}"
            )

    lines.append("================================================================================")
    return "\n".join(lines)
