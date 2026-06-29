"""
charts.py — Plotly visualisation helpers.

All chart functions accept data in the shapes returned by DatabaseManager,
so the main app never has to wrestle with dataframes.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import CLASS_COLORS_HEX, CLASS_NAMES

# ─── Shared theme ─────────────────────────────────────────────────────────────

_BG        = "#0e1117"
_PLOT_BG   = "#161b22"
_TEXT      = "#e0e0e0"
_GRID      = "#2a2d3a"

_BASE_LAYOUT = dict(
    paper_bgcolor = _BG,
    plot_bgcolor  = _PLOT_BG,
    font          = dict(color=_TEXT, family="monospace"),
    margin        = dict(l=16, r=16, t=48, b=16),
    legend        = dict(
        bgcolor      = "rgba(0,0,0,0.4)",
        bordercolor  = "#444",
        borderwidth  = 1,
    ),
)


# ─── Hourly bar chart ─────────────────────────────────────────────────────────

def build_hourly_chart(
    rows: list[dict[str, Any]],
    hours: int = 24,
) -> go.Figure:
    """Grouped bar chart: counts by hour and class."""
    if not rows:
        fig = go.Figure()
        fig.add_annotation(
            text       = "No production data for the selected period.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow  = False,
            font       = dict(size=14, color="#888"),
        )
        fig.update_layout(
            title  = f"Production Log — Last {hours} h",
            **_BASE_LAYOUT,
            height = 380,
        )
        return fig

    df = pd.DataFrame(rows)
    df["hour"] = pd.to_datetime(df["hour"])

    fig = px.bar(
        df,
        x             = "hour",
        y             = "count",
        color         = "class_name",
        barmode       = "group",
        title         = f"Production Log — Last {hours} h",
        color_discrete_map = CLASS_COLORS_HEX,
        labels        = {"hour": "Time", "count": "Pieces", "class_name": "Class"},
        template      = "plotly_dark",
        category_orders = {"class_name": CLASS_NAMES},
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        height             = 380,
        legend_title_text  = "Detection Class",
        xaxis = dict(gridcolor=_GRID, title="Time (UTC)"),
        yaxis = dict(gridcolor=_GRID, title="Count"),
    )
    return fig


# ─── Session pie / donut ──────────────────────────────────────────────────────

def build_session_donut(counts: dict[str, int]) -> go.Figure:
    """Donut chart of current-session class distribution."""
    labels  = list(counts.keys())
    values  = list(counts.values())
    colors  = [CLASS_COLORS_HEX.get(lbl, "#888") for lbl in labels]

    if sum(values) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No objects counted yet.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="#888"),
        )
        fig.update_layout(title="Session Distribution", **_BASE_LAYOUT, height=240)
        return fig

    fig = go.Figure(go.Pie(
        labels          = labels,
        values          = values,
        hole            = 0.55,
        marker_colors   = colors,
        textinfo        = "percent+label",
        textfont        = dict(size=13),
        hovertemplate   = "<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title      = "Session Distribution",
        **_BASE_LAYOUT,
        height     = 240,
        showlegend = False,
    )
    return fig


# ─── Defect rate trend (line chart) ──────────────────────────────────────────

def build_defect_trend_chart(
    rows: list[dict[str, Any]],
    hours: int = 24,
) -> go.Figure:
    """Line chart showing defect rate % per hour over time.

    Computed as: (paper_defect + wrap_defect) / total_in_hour * 100
    Also shows good-unit throughput as a secondary area to give context.
    """
    fig = go.Figure()

    if not rows:
        fig.add_annotation(
            text      = "No data yet — start processing to see the trend.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#888"),
        )
        fig.update_layout(
            title  = f"📉 Defect Rate % / Hour — Last {hours} h",
            **_BASE_LAYOUT,
            height = 280,
        )
        return fig

    df = pd.DataFrame(rows)
    df["hour"] = pd.to_datetime(df["hour"])

    # Pivot so each class is a column
    pivot = (
        df.pivot_table(index="hour", columns="class_name", values="count", aggfunc="sum")
        .fillna(0)
        .reset_index()
    )

    # Ensure all class columns exist
    for cls in CLASS_NAMES:
        if cls not in pivot.columns:
            pivot[cls] = 0

    pivot["total"]       = pivot[CLASS_NAMES].sum(axis=1)
    pivot["defect_rate"] = (
        (pivot.get("paper_defect", 0) + pivot.get("wrap_defect", 0))
        / pivot["total"].replace(0, pd.NA)
        * 100
    ).fillna(0)

    # Area: good throughput (right y-axis)
    fig.add_trace(go.Scatter(
        x          = pivot["hour"],
        y          = pivot.get("good", pd.Series(dtype=float)),
        name       = "Good units",
        fill       = "tozeroy",
        mode       = "lines",
        line       = dict(color="#00cc44", width=1.5),
        fillcolor  = "rgba(0,204,68,0.12)",
        yaxis      = "y2",
        hovertemplate = "<b>Good</b>: %{y}<extra></extra>",
    ))

    # Line: defect rate % (left y-axis)
    fig.add_trace(go.Scatter(
        x          = pivot["hour"],
        y          = pivot["defect_rate"],
        name       = "Defect rate %",
        mode       = "lines+markers",
        line       = dict(color="#ff4444", width=2.5),
        marker     = dict(size=6, color="#ff4444"),
        hovertemplate = "<b>Defect rate</b>: %{y:.1f} %<extra></extra>",
    ))

    # Alert threshold reference line
    fig.add_hline(
        y          = 5,
        line_dash  = "dot",
        line_color = "#ff8800",
        annotation_text      = "Alert 5 %",
        annotation_position  = "bottom right",
        annotation_font_color= "#ff8800",
    )

    # Удаляем или комментируем старый блок и вставляем этот:
    fig.update_layout(
        **_BASE_LAYOUT,
        title=f"📉 Defect Rate % / Hour — Last {hours} h",
        height=280,
        yaxis=dict(
            title="Defect Rate (%)",
            gridcolor=_GRID,
            rangemode="tozero",
        ),
        yaxis2=dict(
            title="Good Units",
            overlaying="y",
            side="right",
            showgrid=False,
            rangemode="tozero",
            tickfont=dict(color="#00cc44"),
        ),
        xaxis=dict(gridcolor=_GRID),
    )

    # Применяем специфичные настройки легенды отдельным вызовом,
    # чтобы не было конфликта с _BASE_LAYOUT
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="#444",
            borderwidth=1,
        )
    )
    return fig