"""Shared visual design tokens for fancy experiment dashboards."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReportTheme:
    """One reusable visual theme shared by HTML, CSS, JS, and Matplotlib."""

    name: str
    primary_color: str
    accent_color: str
    success_color: str
    warning_color: str
    danger_color: str
    background_color: str
    hero_background: str
    card_background: str
    card_background_alt: str
    text_color: str
    muted_text_color: str
    border_color: str
    chart_background: str
    grid_color: str
    chart_palette: tuple[str, ...]
    font_family: tuple[str, ...]
    font_sizes: dict[str, int]
    chart_dimensions: dict[str, tuple[float, float]]
    border_radius: int
    shadow_style: str
    grid_spacing: int
    dark_mode: bool
    use_svg: bool = True
    enable_interactive_charts: bool = True


COMMON_FONTS = (
    "Inter",
    "PingFang SC",
    "Hiragino Sans GB",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
)


THEMES: dict[str, ReportTheme] = {
    "dark_premium": ReportTheme(
        name="dark_premium",
        primary_color="#7dd3fc",
        accent_color="#f472b6",
        success_color="#34d399",
        warning_color="#fbbf24",
        danger_color="#fb7185",
        background_color="#070b16",
        hero_background="radial-gradient(circle at 12% 18%, rgba(125,211,252,.24), transparent 28%), "
        "radial-gradient(circle at 80% 8%, rgba(244,114,182,.18), transparent 30%), "
        "linear-gradient(135deg, #070b16 0%, #0f172a 52%, #111827 100%)",
        card_background="rgba(15, 23, 42, .78)",
        card_background_alt="rgba(30, 41, 59, .64)",
        text_color="#e5edf8",
        muted_text_color="#94a3b8",
        border_color="rgba(148, 163, 184, .22)",
        chart_background="#0b1220",
        grid_color="#263244",
        chart_palette=("#7dd3fc", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#fb7185", "#2dd4bf", "#c084fc"),
        font_family=COMMON_FONTS,
        font_sizes={"title": 15, "label": 10, "tick": 9, "annotation": 9},
        chart_dimensions={
            "wide": (12.8, 6.0),
            "medium": (9.2, 5.8),
            "square": (7.2, 6.8),
            "compact": (7.2, 4.8),
        },
        border_radius=18,
        shadow_style="0 24px 90px rgba(0, 0, 0, .35)",
        grid_spacing=18,
        dark_mode=True,
    ),
    "clean_premium": ReportTheme(
        name="clean_premium",
        primary_color="#2563eb",
        accent_color="#db2777",
        success_color="#059669",
        warning_color="#d97706",
        danger_color="#dc2626",
        background_color="#f5f7fb",
        hero_background="radial-gradient(circle at 8% 20%, rgba(37,99,235,.16), transparent 28%), "
        "radial-gradient(circle at 82% 0%, rgba(219,39,119,.12), transparent 28%), "
        "linear-gradient(135deg, #ffffff 0%, #eef4ff 55%, #fdf2f8 100%)",
        card_background="rgba(255, 255, 255, .86)",
        card_background_alt="#f8fafc",
        text_color="#0f172a",
        muted_text_color="#64748b",
        border_color="#dde6f2",
        chart_background="#ffffff",
        grid_color="#e6edf6",
        chart_palette=("#2563eb", "#db2777", "#059669", "#d97706", "#7c3aed", "#dc2626", "#0891b2", "#9333ea"),
        font_family=COMMON_FONTS,
        font_sizes={"title": 15, "label": 10, "tick": 9, "annotation": 9},
        chart_dimensions={
            "wide": (12.8, 6.0),
            "medium": (9.2, 5.8),
            "square": (7.2, 6.8),
            "compact": (7.2, 4.8),
        },
        border_radius=16,
        shadow_style="0 18px 50px rgba(15, 23, 42, .10)",
        grid_spacing=18,
        dark_mode=False,
    ),
}


def get_theme(name: str) -> ReportTheme:
    """Return a named theme, falling back to dark premium."""

    return THEMES.get(name, THEMES["dark_premium"])
