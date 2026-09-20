"""Custom Evidently metrics for churn drift monitoring."""

from __future__ import annotations

from typing import List, Optional

from evidently.core.datasets import Dataset
from evidently.core.metric_types import BoundTest, SingleValueCalculation, SingleValueMetric
from evidently.core.report import Context
from evidently.tests import lt


class MonthlyChargesMeanAbsDiff(SingleValueMetric):
    """Absolute difference between current and reference mean MonthlyCharges."""

    column: str = "MonthlyCharges"
    threshold: float = 5.0

    def _default_tests_with_reference(self, context: Context) -> List[BoundTest]:
        # Fail (drift) when abs mean diff >= threshold → test that value < threshold
        return [lt(self.threshold).bind_single(self.get_fingerprint())]


class MonthlyChargesMeanAbsDiffCalculation(SingleValueCalculation[MonthlyChargesMeanAbsDiff]):
    def calculate(
        self,
        context: Context,
        current_data: Dataset,
        reference_data: Optional[Dataset],
    ):
        if reference_data is None:
            raise ValueError("MonthlyChargesMeanAbsDiff requires reference data")
        col = self.metric.column
        cur_mean = float(current_data.column(col).data.mean())
        ref_mean = float(reference_data.column(col).data.mean())
        value = abs(cur_mean - ref_mean)
        result = self.result(value)
        result.display_name = (
            f"|mean({col})_current - mean({col})_reference| = {value:.3f} "
            f"(threshold={self.metric.threshold})"
        )
        return result

    def display_name(self) -> str:
        return f"Abs mean diff of '{self.metric.column}' (current vs reference)"
