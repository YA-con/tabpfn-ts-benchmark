"""Dashboard specification for final benchmark reporting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FigureSpec:
    """Specification for one report figure."""

    name: str
    section: str
    purpose: str
    required_columns: tuple[str, ...]


DASHBOARD_FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec(
        name="cross_domain_leaderboard",
        section="overview",
        purpose="Rank models by normalized forecasting metrics across datasets and domains.",
        required_columns=("model", "dataset", "domain", "mse", "mae", "smape", "mase"),
    ),
    FigureSpec(
        name="model_domain_heatmap",
        section="overview",
        purpose="Show model strength and weakness by domain in a compact matrix.",
        required_columns=("model", "domain", "dataset", "smape"),
    ),
    FigureSpec(
        name="context_degradation_fan",
        section="low_data",
        purpose="Visualize how each model degrades as context length shrinks.",
        required_columns=("model", "dataset", "context_length", "smape"),
    ),
    FigureSpec(
        name="forecast_gallery",
        section="diagnostics",
        purpose="Show representative actual-versus-forecast trajectories.",
        required_columns=("unique_id", "ds", "y", "y_hat", "model", "dataset"),
    ),
    FigureSpec(
        name="finance_risk_return_bubble",
        section="finance",
        purpose="Compare annual return, drawdown, and stability for tradable signals.",
        required_columns=("model", "annual_return", "max_drawdown", "stability_pass", "sharpe"),
    ),
    FigureSpec(
        name="finance_signal_decay",
        section="finance",
        purpose="Measure IC and Rank IC decay across forecast horizons.",
        required_columns=("model", "horizon", "ic", "rank_ic"),
    ),
)


def list_dashboard_figures() -> tuple[FigureSpec, ...]:
    """Return planned dashboard figures."""

    return DASHBOARD_FIGURES
