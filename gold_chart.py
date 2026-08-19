import base64
import json
import re
import uuid
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from dash import (
    Dash,
    Input,
    Output,
    State,
    ctx,
    dcc,
    html,
    no_update,
)

# ============================================================
# 1. LOAD CSV
# ============================================================
# This reads the CSV produced by gold_myr_tracker.py, which
# already contains real Date, Open, High, Low, Close columns —
# unlike the Dinar CSV this app was originally built for, no
# synthetic Open/Close needs to be derived here.
# ============================================================

FILE_PATH = "gold_myr_ohlc.csv"

df = pd.read_csv(FILE_PATH)


# ============================================================
# 2. CONVERT DATE
# ============================================================

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# 3. CANDLE HIGH/LOW ALIASES
# ============================================================
# The rest of this app (create_figure, support/resistance,
# Fibonacci, etc.) references df["CandleHigh"] / df["CandleLow"]
# so shapes always draw against the true daily wick — kept here
# purely so the drawing-tool code below doesn't need changes.
# ============================================================

df["CandleHigh"] = df["High"]
df["CandleLow"] = df["Low"]


# ============================================================
# 3B. HELPER — HEX COLOR TO RGBA STRING
# ============================================================
# Converts whatever hex color the user picked into a translucent
# rgba() fill so the Zone tool's fill always matches its outline
# color.
# ============================================================


def hex_to_rgba(hex_color, alpha=0.10):
    hex_color = hex_color.lstrip("#")

    if len(hex_color) != 6:
        # Fallback to the original default blue if something unexpected
        # is passed in (defensive — dropdown values are always valid hex).
        return "rgba(37,99,235,0.10)"

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return f"rgba({r},{g},{b},{alpha})"


# ============================================================
# 3C. HELPER — NORMALIZE FIBONACCI X VALUES
# ============================================================
# The clicked point's x-value could come from either
# click_data["points"][0]["x"] (Plotly's own date-string format)
# or, as a fallback, str(df.loc[index, "Date"]) (pandas Timestamp
# string format). These two formats don't always match, and the
# Fibonacci rectangle later compares x-values as strings via
# min()/max(), which can order the box incorrectly if the formats
# differ. Routing both paths through this helper guarantees a
# single consistent ISO format.
# ============================================================


def normalize_x(value):
    try:
        return pd.to_datetime(value).isoformat()
    except Exception:
        return str(value)


# ============================================================
# 3D. HELPERS — FIBONACCI RETRACEMENT BUILDING BLOCKS
# ============================================================
# The two points a user clicks to place a Fibonacci retracement
# are drawn as small, fixed-pixel-size, draggable circle markers
# (name "fibpointN:<fib_id>"). Dragging either marker fires
# relayout keys for its xanchor/yanchor (because
# xsizemode/ysizemode="pixel" keeps its on-screen size constant
# and only its anchor position changes). handle_shape_relayout
# listens for that and rebuilds the retracement using these same
# helpers, so the whole thing recomputes live as a point is moved.
# The 7 level lines and the shaded rectangle are set
# editable=False — they're purely derived from the two points and
# are no longer meant to be dragged directly.
# ============================================================


def build_fib_levels(first, second):
    first_price = float(first["y"])
    second_price = float(second["y"])

    swing_high = max(first_price, second_price)
    swing_low = min(first_price, second_price)
    difference = swing_high - swing_low

    levels = {
        "0.0%": swing_high,
        "23.6%": swing_high - difference * 0.236,
        "38.2%": swing_high - difference * 0.382,
        "50.0%": swing_high - difference * 0.500,
        "61.8%": swing_high - difference * 0.618,
        "78.6%": swing_high - difference * 0.786,
        "100%": swing_low,
    }

    return swing_high, swing_low, levels


def build_fib_point_marker(fib_id, point_index, x, y):
    """A small circle marking one of the two Fibonacci anchor points.

    NOTE: this deliberately uses ordinary data-unit sizing (x0/x1/y0/y1
    expressed in date/price units) rather than Plotly's
    xsizemode/ysizemode="pixel" mode. Pixel-sized shapes look like the
    obvious choice for a fixed-size "drag handle" — but Plotly.js's
    interactive shape-drag logic does not actually support translating
    pixel-anchored shapes: the circle renders fine but silently
    refuses to move no matter how it's dragged. Data-unit sizing is
    the same mechanism already used for support/resistance and
    trendline shapes in this app, and it drags reliably — the only
    cost is the marker's on-screen size shifting slightly as you
    zoom in/out.
    """
    x_center = pd.to_datetime(x)
    x_radius = (df["Date"].max() - df["Date"].min()) * 0.008
    y_radius = (df["High"].max() - df["Low"].min()) * 0.012

    return dict(
        type="circle",
        xref="x",
        yref="y",
        x0=x_center - x_radius,
        x1=x_center + x_radius,
        y0=y - y_radius,
        y1=y + y_radius,
        line=dict(color="#8b5cf6", width=2),
        fillcolor="#ffffff",
        opacity=1,
        editable=True,
        name=f"fibpoint{point_index}:{fib_id}",
    )


def fib_marker_center(shape):
    """Given a fib point marker's shape dict, return its center as
    {'x': <normalized date string>, 'y': <float price>}, or None if
    the shape doesn't have usable coordinates."""
    try:
        center_x = pd.to_datetime([shape.get("x0"), shape.get("x1")]).mean()
        center_y = (float(shape.get("y0")) + float(shape.get("y1"))) / 2
        return {"x": normalize_x(center_x), "y": center_y}
    except Exception:
        return None


def build_fib_shapes_and_annotations(fib_id, first, second, x_range):
    swing_high, swing_low, levels = build_fib_levels(first, second)

    line_shapes = []
    fib_annotations = []

    for label, level in levels.items():
        line_shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=x_range[0],
                x1=x_range[1],
                y0=level,
                y1=level,
                line=dict(color="#8b5cf6", width=2, dash="dash"),
                editable=False,
                name=f"fibonacci:{fib_id}",
            )
        )

        fib_annotations.append(
            dict(
                x=x_range[1],
                y=level,
                text=f"FIB:{label} RM{level:.2f}",
                name=f"fibonacci:{fib_id}",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=11, color="#8b5cf6"),
                bgcolor="rgba(255,255,255,0.85)",
            )
        )

    rect_shape = dict(
        type="rect",
        xref="x",
        yref="y",
        x0=min(first["x"], second["x"]),
        x1=max(first["x"], second["x"]),
        y0=swing_low,
        y1=swing_high,
        fillcolor="#8b5cf6",
        opacity=0.08,
        line=dict(color="#8b5cf6", width=1, dash="dot"),
        editable=False,
        name=f"fibonacci:{fib_id}",
    )

    return line_shapes, rect_shape, fib_annotations


# ============================================================
# 4. CREATE BASE FIGURE
# ============================================================


def create_figure():

    fig = go.Figure()

    # ========================================================
    # CANDLESTICK
    # ========================================================

    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["CandleHigh"],
            low=df["CandleLow"],
            close=df["Close"],
            # ------------------------------------------------
            # BULLISH
            # ------------------------------------------------
            increasing=dict(line=dict(color="#16a34a", width=1), fillcolor="#16a34a"),
            # ------------------------------------------------
            # BEARISH
            # ------------------------------------------------
            decreasing=dict(line=dict(color="#dc2626", width=1), fillcolor="#dc2626"),
            name="Gold",
        )
    )

    # ========================================================
    # LAYOUT
    # ========================================================

    fig.update_layout(
        title="Gold Price Technical Analysis (MYR per gram)",
        yaxis_title="Price (RM/gram)",
        xaxis_title="Date",
        template="plotly_white",
        height=700,
        hovermode="x unified",
        dragmode="pan",
        showlegend=False,
        margin=dict(l=60, r=180, t=60, b=60),
        xaxis=dict(rangeslider=dict(visible=False), type="date"),
        newshape=dict(
            line=dict(color="#2563eb", width=2), fillcolor=("rgba(37,99,235,0.10)")
        ),
    )

    return fig


# ============================================================
# 5. DASH APP
# ============================================================

app = Dash(__name__)

# ------------------------------------------------------------
# BUTTON STYLING
#
# Each button gets a color tied to what it does: drawing-tool
# buttons match the color of the shape they draw (green support,
# red resistance, blue trendline, purple fibonacci), and the
# save/load/clear actions use a neutral action-color scheme
# (teal = save, slate = load, amber = partial clear, deep red =
# clear everything). Hover/active states live in index_string
# below since inline styles can't express ":hover".
# ------------------------------------------------------------

BASE_BUTTON_STYLE = {
    "border": "none",
    "borderRadius": "8px",
    "padding": "10px 18px",
    "fontSize": "14px",
    "fontWeight": "700",
    "fontFamily": '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    "color": "#ffffff",
    "cursor": "pointer",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.15)",
}


def button_style(bg_color):
    return {**BASE_BUTTON_STYLE, "backgroundColor": bg_color}


TOOL_BUTTON_COLORS = {
    "support": "#16a34a",     # green — matches the support line
    "resistance": "#dc2626",  # red — matches the resistance line
    "trendline": "#2563eb",   # blue — matches the default trendline color
    "zone": "#0ea5e9",        # sky blue — visually distinct drawing tool
    "fibonacci": "#8b5cf6",   # purple — matches the fibonacci color
}

ACTION_BUTTON_COLORS = {
    "save": "#0d9488",     # teal — positive / save action
    "load": "#64748b",     # slate — neutral, non-destructive
    "clear_fib": "#f59e0b",  # amber — partial clear, mild warning
    "clear_all": "#b91c1c",  # deep red — destructive, clears everything
}

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .gold-btn {
                transition: transform 0.12s ease, box-shadow 0.12s ease,
                    filter 0.12s ease;
            }
            .gold-btn:hover {
                filter: brightness(1.08);
                transform: translateY(-1px);
                box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            }
            .gold-btn:active {
                transform: translateY(0);
                filter: brightness(0.94);
                box-shadow: 0 1px 2px rgba(0,0,0,0.15);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


# ============================================================
# 6. APP LAYOUT
# ============================================================

app.layout = html.Div(
    [
        # ====================================================
        # TITLE
        # ====================================================
        html.H2("Gold Price Technical Analysis"),
        html.Div(
            "Interactive technical-analysis chart (MYR per gram)",
            style={"color": "#666", "marginBottom": "15px"},
        ),
        # ====================================================
        # DRAWING TOOLS
        # ====================================================
        html.Div(
            [
                html.Div(
                    "DRAWING TOOLS",
                    style={
                        "fontSize": "11px",
                        "fontWeight": "700",
                        "letterSpacing": "0.06em",
                        "color": "#94a3b8",
                        "marginBottom": "8px",
                    },
                ),
                html.Div(
                    [
                        html.Button(
                            "Support",
                            id="support-button",
                            n_clicks=0,
                            className="gold-btn",
                            style=button_style(TOOL_BUTTON_COLORS["support"]),
                        ),
                        html.Button(
                            "Resistance",
                            id="resistance-button",
                            n_clicks=0,
                            className="gold-btn",
                            style=button_style(TOOL_BUTTON_COLORS["resistance"]),
                        ),
                        html.Button(
                            "Trendline",
                            id="trendline-button",
                            n_clicks=0,
                            className="gold-btn",
                            style=button_style(TOOL_BUTTON_COLORS["trendline"]),
                        ),
                        html.Button(
                            "Zone",
                            id="zone-button",
                            n_clicks=0,
                            className="gold-btn",
                            style=button_style(TOOL_BUTTON_COLORS["zone"]),
                        ),
                        html.Button(
                            "Fibonacci",
                            id="fib-button",
                            n_clicks=0,
                            className="gold-btn",
                            style=button_style(TOOL_BUTTON_COLORS["fibonacci"]),
                        ),
                    ],
                    style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
                ),
            ],
            style={
                "backgroundColor": "#f8fafc",
                "border": "1px solid #e2e8f0",
                "borderRadius": "10px",
                "padding": "12px 14px",
                "marginBottom": "10px",
            },
        ),
        # ====================================================
        # SAVE / LOAD / CLEAR
        # ====================================================
        html.Div(
            [
                html.Div(
                    "FILE & CLEANUP",
                    style={
                        "fontSize": "11px",
                        "fontWeight": "700",
                        "letterSpacing": "0.06em",
                        "color": "#94a3b8",
                        "marginBottom": "8px",
                    },
                ),
                html.Div(
                    [
                        html.Button(
                            "Save Drawing",
                            id="save-drawing-button",
                            n_clicks=0,
                            className="gold-btn",
                            style=button_style(ACTION_BUTTON_COLORS["save"]),
                        ),
                        dcc.Upload(
                            id="load-drawing-upload",
                            children=html.Button(
                                "Load Drawing",
                                id="load-drawing-btn",
                                className="gold-btn",
                                style=button_style(ACTION_BUTTON_COLORS["load"]),
                            ),
                            multiple=False,
                            accept=".json",
                        ),
                        html.Button(
                            "Clear Fibonacci",
                            id="clear-fib-button",
                            n_clicks=0,
                            className="gold-btn",
                            style=button_style(ACTION_BUTTON_COLORS["clear_fib"]),
                        ),
                        html.Button(
                            "Clear All",
                            id="clear-button",
                            n_clicks=0,
                            className="gold-btn",
                            style=button_style(ACTION_BUTTON_COLORS["clear_all"]),
                        ),
                    ],
                    style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
                ),
            ],
            style={
                "backgroundColor": "#f8fafc",
                "border": "1px solid #e2e8f0",
                "borderRadius": "10px",
                "padding": "12px 14px",
                "marginBottom": "15px",
            },
        ),
        # ====================================================
        # DOWNLOAD
        # ====================================================
        dcc.Download(id="download-drawing"),
        # ====================================================
        # SETTINGS
        # ====================================================
        html.Div(
            [
                # ------------------------------------------------
                # LINE THICKNESS
                # ------------------------------------------------
                html.Div(
                    [
                        html.Label("Line thickness"),
                        dcc.Dropdown(
                            id="line-width",
                            options=[
                                {"label": "1 px", "value": 1},
                                {"label": "2 px", "value": 2},
                                {"label": "3 px", "value": 3},
                                {"label": "4 px", "value": 4},
                                {"label": "5 px", "value": 5},
                            ],
                            value=2,
                            clearable=False,
                            style={"width": "130px"},
                        ),
                    ]
                ),
                # ------------------------------------------------
                # COLOR
                # ------------------------------------------------
                html.Div(
                    [
                        html.Label("Trendline / Zone color"),
                        dcc.Dropdown(
                            id="line-color",
                            options=[
                                {"label": "Blue", "value": "#2563eb"},
                                {"label": "Green", "value": "#16a34a"},
                                {"label": "Red", "value": "#dc2626"},
                                {"label": "Purple", "value": "#8b5cf6"},
                                {"label": "Orange", "value": "#f97316"},
                                {"label": "Black", "value": "#111827"},
                            ],
                            value="#2563eb",
                            clearable=False,
                            style={"width": "180px"},
                        ),
                    ]
                ),
            ],
            style={
                "display": "flex",
                "gap": "20px",
                "alignItems": "end",
                "marginBottom": "15px",
            },
        ),
        # ====================================================
        # STATUS
        # ====================================================
        html.Div(
            id="tool-info",
            children="Pan mode active.",
            style={"fontWeight": "bold", "marginBottom": "8px"},
        ),
        html.Div(
            id="selection-info",
            children="",
            style={"color": "#555", "marginBottom": "10px"},
        ),
        # ====================================================
        # CHART
        # ====================================================
        dcc.Graph(
            id="price-chart",
            figure=create_figure(),
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "modeBarButtonsToAdd": [
                    "drawline",
                    "drawopenpath",
                    "drawrect",
                    "drawcircle",
                    "eraseshape",
                ],
            },
        ),
        # ====================================================
        # ACTIVE TOOL
        # ====================================================
        dcc.Store(id="active-tool", data="pan"),
        # ====================================================
        # FIBONACCI FIRST POINT
        # ====================================================
        dcc.Store(id="selected-points", data=[]),
    ],
    style={"maxWidth": "1250px", "margin": "30px auto", "padding": "20px"},
)


# ============================================================
# 7. SELECT TOOL
# ============================================================


@app.callback(
    Output("price-chart", "figure", allow_duplicate=True),
    # added allow_duplicate=True — handle_graph_click and
    # handle_shape_relayout also write to active-tool.data, so every
    # callback sharing this Output must declare it or Dash raises
    # DuplicateCallbackOutput at startup.
    Output("active-tool", "data", allow_duplicate=True),
    # same reasoning — tool-info.children is also written
    # by handle_graph_click and handle_shape_relayout.
    Output("tool-info", "children", allow_duplicate=True),
    # clear any stale Fibonacci first-click point whenever
    # a tool button is pressed, so switching tools mid-selection can't
    # cause the next click to be misread as a second Fibonacci point.
    Output("selected-points", "data", allow_duplicate=True),
    Input("support-button", "n_clicks"),
    Input("resistance-button", "n_clicks"),
    Input("trendline-button", "n_clicks"),
    Input("zone-button", "n_clicks"),
    Input("fib-button", "n_clicks"),
    State("price-chart", "figure"),
    State("line-color", "value"),
    State("line-width", "value"),
    prevent_initial_call=True,
)
def select_tool(
    support_clicks,
    resistance_clicks,
    trendline_clicks,
    zone_clicks,
    fib_clicks,
    current_figure,
    color,
    width,
):

    triggered = ctx.triggered_id

    fig = go.Figure(current_figure)

    # ========================================================
    # SUPPORT
    # ========================================================

    if triggered == "support-button":
        fig.update_layout(dragmode="pan")

        return (fig, "support", "SUPPORT — click the desired price level.", [])

    # ========================================================
    # RESISTANCE
    # ========================================================

    if triggered == "resistance-button":
        fig.update_layout(dragmode="pan")

        return (fig, "resistance", "RESISTANCE — click the desired price level.", [])

    # ========================================================
    # TRENDLINE
    # ========================================================

    if triggered == "trendline-button":
        fig.update_layout(
            dragmode="drawline", newshape=dict(line=dict(color=color, width=width))
        )

        return (fig, "trendline", "TRENDLINE — drag between two points.", [])

    # ========================================================
    # ZONE
    # ========================================================

    if triggered == "zone-button":
        fig.update_layout(
            dragmode="drawrect",
            newshape=dict(
                line=dict(color=color, width=width),
                # fill derived from the selected color instead of
                # being hardcoded to blue.
                fillcolor=hex_to_rgba(color, 0.10),
            ),
        )

        return (fig, "zone", "ZONE — drag to create a zone.", [])

    # ========================================================
    # FIBONACCI
    # ========================================================

    if triggered == "fib-button":
        fig.update_layout(dragmode="pan")

        return (fig, "fibonacci", "FIBONACCI — click two points.", [])

    return (fig, "pan", "Pan mode active.", [])


# ============================================================
# 8. HANDLE GRAPH CLICK
# ============================================================


@app.callback(
    Output("price-chart", "figure", allow_duplicate=True),
    Output("active-tool", "data", allow_duplicate=True),
    Output("selected-points", "data", allow_duplicate=True),
    Output("selection-info", "children"),
    Output("tool-info", "children", allow_duplicate=True),
    Input("price-chart", "clickData"),
    State("price-chart", "figure"),
    State("active-tool", "data"),
    State("selected-points", "data"),
    State("line-width", "value"),
    prevent_initial_call=True,
)
def handle_graph_click(click_data, current_figure, active_tool, selected_points, width):

    if not click_data:
        return (no_update, no_update, no_update, no_update, no_update)

    # ========================================================
    # PAN MODE
    # ========================================================

    if active_tool == "pan":
        return (no_update, no_update, no_update, no_update, no_update)

    # ========================================================
    # GET POINT
    # ========================================================

    try:
        point = click_data["points"][0]

    except (KeyError, IndexError, TypeError):
        return (
            no_update,
            no_update,
            no_update,
            "Unable to read clicked point.",
            no_update,
        )

    # ========================================================
    # GET PRICE
    # ========================================================

    clicked_price = point.get("y")

    if clicked_price is None:
        try:
            index = int(point["pointNumber"])

            clicked_price = float(df.loc[index, "Close"])

        except Exception:
            return (
                no_update,
                no_update,
                no_update,
                "Unable to determine price.",
                no_update,
            )

    try:
        price = float(clicked_price)

    except (TypeError, ValueError):
        return (no_update, no_update, no_update, "Invalid price.", no_update)

    fig = go.Figure(current_figure)

    # ========================================================
    # SUPPORT
    # ========================================================

    if active_tool == "support":
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=df["Date"].iloc[0],
            x1=df["Date"].iloc[-1],
            y0=price,
            y1=price,
            line=dict(color="#16a34a", width=width),
            label=dict(
                texttemplate="S %{y0:.2f}",
                textposition="end",
                xanchor="right",
                font=dict(
                    color="#16a34a",
                    size=12,
                ),
                padding=4,
            ),
            editable=True,
            name="support",
        )

        fig.update_layout(dragmode="pan")

        return (
            fig,
            "pan",
            [],
            "",
            "Pan mode — click the support line to select it.",
        )

    # ========================================================
    # RESISTANCE
    # ========================================================

    if active_tool == "resistance":
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=df["Date"].iloc[0],
            x1=df["Date"].iloc[-1],
            y0=price,
            y1=price,
            line=dict(color="#dc2626", width=width),
            label=dict(
                texttemplate="R %{y0:.2f}",
                textposition="end",
                xanchor="right",
                font=dict(
                    color="#dc2626",
                    size=12,
                ),
                padding=4,
            ),
            editable=True,
            name="resistance",
        )

        fig.update_layout(dragmode="pan")

        return (
            fig,
            "pan",
            [],
            "",
            "Pan mode — click the resistance line to select it.",
        )

    # ========================================================
    # FIBONACCI
    # ========================================================

    if active_tool == "fibonacci":
        x_value = point.get("x")

        if x_value is None:
            try:
                index = int(point["pointNumber"])

                x_value = df.loc[index, "Date"]

            except Exception:
                return (
                    no_update,
                    no_update,
                    no_update,
                    "Unable to determine date.",
                    no_update,
                )

        # route both the normal click-provided x-value and the
        # pointNumber fallback through the same normalizer so the
        # two Fibonacci points always share one consistent date
        # format (otherwise they could differ, causing the shaded
        # Fibonacci rectangle's min()/max() string comparison to
        # pick the wrong left/right edge).
        selected_point = {"x": normalize_x(x_value), "y": price}

        # ====================================================
        # FIRST CLICK
        # ====================================================

        if len(selected_points) == 0:
            return (
                no_update,
                "fibonacci",
                [selected_point],
                f"First Fibonacci point: RM{price:.2f}",
                "FIBONACCI — click the second point.",
            )

        # ====================================================
        # SECOND CLICK — BUILD THE RETRACEMENT
        # ====================================================

        first = selected_points[0]

        second = selected_point

        if float(first["y"]) == float(second["y"]):
            return (
                no_update,
                "pan",
                [],
                "Invalid Fibonacci range.",
                "Pan mode active.",
            )

        # A unique id ties this retracement's lines, rectangle and
        # labels together so they can be found and rebuilt as a group
        # whenever either point marker below gets dragged.
        fib_id = uuid.uuid4().hex[:8]

        x_range = (df["Date"].iloc[0], df["Date"].iloc[-1])

        line_shapes, rect_shape, fib_annotations = build_fib_shapes_and_annotations(
            fib_id, first, second, x_range
        )

        for shape in line_shapes:
            fig.add_shape(**shape)

        fig.add_shape(**rect_shape)

        for annotation in fib_annotations:
            fig.add_annotation(**annotation)

        # ----------------------------------------------------
        # DRAGGABLE ANCHOR POINTS
        #
        # These two circles sit exactly on the points the user
        # clicked. They're the only part of a Fibonacci retracement
        # left draggable — dragging either one automatically
        # recomputes the whole retracement (see
        # handle_shape_relayout, Part A, below).
        # ----------------------------------------------------

        fig.add_shape(
            **build_fib_point_marker(fib_id, 0, first["x"], float(first["y"]))
        )

        fig.add_shape(
            **build_fib_point_marker(fib_id, 1, second["x"], float(second["y"]))
        )

        fig.update_layout(dragmode="pan")

        return (
            fig,
            "pan",
            [],
            "Fibonacci retracement added — drag either point to adjust it.",
            "Pan mode active.",
        )

    return (no_update, no_update, no_update, no_update, no_update)


# ============================================================
# 8B. HANDLE SHAPE RELAYOUT EVENTS
# ============================================================
#
# This single callback covers two things that both react to the
# same event (the user dragging/drawing something on the chart):
#
# 1. RESET DRAGMODE AFTER TRENDLINE / ZONE IS DRAWN
#    Trendline and Zone shapes are drawn using Plotly's native
#    "drawline" / "drawrect" dragmodes. While one of those
#    dragmodes is active, clicking on an existing shape starts a
#    NEW shape instead of selecting it — which means "Erase
#    active shape" never has anything to act on. Once a shape has
#    actually been drawn, switch back to "pan" so the shape can
#    be clicked/selected and then removed with the "Erase active
#    shape" modebar button.
#
# 2. KEEP FIBONACCI LABELS IN SYNC WHILE DRAGGING
#    Each Fibonacci level is drawn as an independent, editable
#    line shape (so it can be dragged), plus a separate text
#    annotation showing its price. When a Fibonacci line is
#    dragged, Plotly only moves the shape — the annotation is
#    left behind. This watches for a dragged Fibonacci line and
#    moves + relabels its matching annotation so the price text
#    always tracks the line.
#
# NOTE: both behaviors are combined into one callback (rather
# than two separate ones) because they share the same
# Input/State signature — Dash's allow_duplicate matching would
# otherwise treat two callbacks with an identical Input/State
# pair as the same registration and collide.
# ============================================================


@app.callback(
    Output("price-chart", "figure", allow_duplicate=True),
    Output("active-tool", "data", allow_duplicate=True),
    Output("tool-info", "children", allow_duplicate=True),
    Input("price-chart", "relayoutData"),
    State("price-chart", "figure"),
    prevent_initial_call=True,
)
def handle_shape_relayout(relayout_data, current_figure):

    if not relayout_data or not current_figure:
        return (no_update, no_update, no_update)

    layout = current_figure.get("layout", {}) or {}

    shapes = layout.get("shapes", []) or []
    annotations = layout.get("annotations", []) or []

    new_shapes = [dict(shape) for shape in shapes]
    new_annotations = [dict(annotation) for annotation in annotations]

    changed = False

    # ========================================================
    # PART A — RECOMPUTE FIBONACCI WHEN A POINT IS DRAGGED
    #
    # The two point markers placed at a Fibonacci retracement's
    # original clicks (see handle_graph_click, build_fib_point_marker)
    # are ordinary data-unit-sized circles, so dragging one moves its
    # x0/x1/y0/y1 corners — that's what shows up here as
    # "shapes[i].x0" / ".x1" / ".y0" / ".y1" relayout keys. When that
    # happens, we look up which Fibonacci group ("fib_id", embedded
    # in the marker's name) the point belongs to, work out that
    # point's new center, and rebuild the group's 7 level lines +
    # rectangle + labels + both point markers from scratch.
    # Everything else on the chart — other retracements,
    # support/resistance lines, trendlines, zones — is untouched.
    # ========================================================

    fib_shape_updates = {}  # shape_index -> {"x0":.., "x1":.., "y0":.., "y1":..}

    for key, value in relayout_data.items():

        match = re.match(r"^shapes\[(\d+)\]\.(x0|x1|y0|y1)$", key)

        if not match:
            continue

        shape_index = int(match.group(1))
        prop = match.group(2)

        if shape_index >= len(shapes):
            continue

        shape_name = str(shapes[shape_index].get("name", ""))

        if not re.match(r"^fibpoint[01]:[0-9a-fA-F]+$", shape_name):
            continue

        fib_shape_updates.setdefault(shape_index, {})[prop] = value

    fib_point_updates = {}  # fib_id -> {0: {"x":..,"y":..}, 1: {...}}

    for shape_index, updated_props in fib_shape_updates.items():

        shape_name = str(shapes[shape_index].get("name", ""))

        point_match = re.match(r"^fibpoint([01]):([0-9a-fA-F]+)$", shape_name)

        if not point_match:
            continue

        point_index = int(point_match.group(1))
        fib_id = point_match.group(2)

        merged_shape = dict(shapes[shape_index])
        merged_shape.update(updated_props)

        center = fib_marker_center(merged_shape)

        if center is None:
            continue

        fib_point_updates.setdefault(fib_id, {})[point_index] = center

    for fib_id, points in fib_point_updates.items():

        # A drag only reports the point that actually moved — fetch
        # the other point's current center straight from its shape.
        for point_index in (0, 1):

            if point_index in points:
                continue

            marker_name = f"fibpoint{point_index}:{fib_id}"

            marker_shape = next(
                (s for s in shapes if s.get("name") == marker_name), None
            )

            points[point_index] = (
                fib_marker_center(marker_shape) if marker_shape else None
            )

        if points.get(0) is None or points.get(1) is None:
            continue

        first = points[0]
        second = points[1]

        if first["y"] == second["y"]:
            continue

        # Drop this group's old level lines / rectangle / labels /
        # point markers — everything gets rebuilt fresh below.
        group_name = f"fibonacci:{fib_id}"
        point0_name = f"fibpoint0:{fib_id}"
        point1_name = f"fibpoint1:{fib_id}"

        new_shapes = [
            s
            for s in new_shapes
            if str(s.get("name", "")) not in (group_name, point0_name, point1_name)
        ]

        new_annotations = [
            a for a in new_annotations if str(a.get("name", "")) != group_name
        ]

        x_range = (df["Date"].iloc[0], df["Date"].iloc[-1])

        line_shapes, rect_shape, fib_annotations = build_fib_shapes_and_annotations(
            fib_id, first, second, x_range
        )

        new_shapes.extend(line_shapes)
        new_shapes.append(rect_shape)
        new_shapes.append(build_fib_point_marker(fib_id, 0, first["x"], first["y"]))
        new_shapes.append(build_fib_point_marker(fib_id, 1, second["x"], second["y"]))
        new_annotations.extend(fib_annotations)

        changed = True

    # ========================================================
    # PART B — RESET DRAGMODE AFTER TRENDLINE / ZONE IS DRAWN
    # ========================================================

    new_dragmode = None
    new_active_tool = no_update
    new_tool_info = no_update

    if "shapes" in relayout_data:

        current_dragmode = layout.get("dragmode")

        if current_dragmode in ("drawline", "drawrect"):
            new_dragmode = "pan"
            new_active_tool = "pan"
            new_tool_info = (
                "Pan mode — click the shape to select it, "
                "then use Erase Active Shape."
            )
            changed = True

    if not changed:
        return (no_update, no_update, no_update)

    fig = go.Figure(current_figure)

    fig.update_layout(shapes=new_shapes, annotations=new_annotations)

    if new_dragmode:
        fig.update_layout(dragmode=new_dragmode)

    return (fig, new_active_tool, new_tool_info)


# ============================================================
# 9. CLEAR FIBONACCI — CLIENT SIDE
# ============================================================
#
# IMPORTANT:
#
# There is NO Python @app.callback for Clear Fibonacci.
#
# This runs directly in the browser.
#
# ============================================================

app.clientside_callback(
    """
    function(n_clicks, figure) {

        // ----------------------------------------------------
        // Nothing to do
        // ----------------------------------------------------

        if (!n_clicks) {

            return window.dash_clientside.no_update;

        }


        if (!figure) {

            return window.dash_clientside.no_update;

        }


        // ----------------------------------------------------
        // Make a deep copy
        // ----------------------------------------------------

        var newFigure =
            JSON.parse(
                JSON.stringify(figure)
            );


        // ----------------------------------------------------
        // Make sure layout exists
        // ----------------------------------------------------

        if (!newFigure.layout) {

            newFigure.layout = {};

        }


        // ====================================================
        // REMOVE FIBONACCI SHAPES
        // ====================================================

        if (
            Array.isArray(
                newFigure.layout.shapes
            )
        ) {

            newFigure.layout.shapes =
                newFigure.layout.shapes.filter(

                    function(shape) {

                        // ------------------------------------
                        // Explicit Fibonacci name
                        //
                        // Covers: legacy exact "fibonacci" name,
                        // id-tagged "fibonacci:<fib_id>" lines/
                        // rectangle, and the two draggable point
                        // markers "fibpoint0:<fib_id>" /
                        // "fibpoint1:<fib_id>".
                        // ------------------------------------

                        if (
                            typeof shape.name === "string"
                            &&
                            (
                                shape.name === "fibonacci"
                                ||
                                shape.name.indexOf("fibonacci:") === 0
                                ||
                                shape.name.indexOf("fibpoint0:") === 0
                                ||
                                shape.name.indexOf("fibpoint1:") === 0
                            )
                        ) {

                            return false;

                        }


                        // ------------------------------------
                        // Line information
                        // ------------------------------------

                        var line =
                            shape.line || {};


                        var color =
                            String(
                                line.color || ""
                            ).toLowerCase();


                        var dash =
                            String(
                                line.dash || ""
                            ).toLowerCase();


                        // ------------------------------------
                        // Normalize RGB spacing
                        // ------------------------------------

                        color = color.replace(
                            /\\s+/g,
                            " "
                        );


                        // ------------------------------------
                        // Purple Fibonacci color
                        // ------------------------------------

                        var isPurple = (

                            color === "#8b5cf6"

                            ||

                            color ===
                            "rgb(139, 92, 246)"

                            ||

                            color ===
                            "rgba(139, 92, 246, 1)"

                            ||

                            color ===
                            "rgba(139,92,246,1)"

                        );


                        // ------------------------------------
                        // Fibonacci dashed line
                        // ------------------------------------

                        var isFibLine = (

                            shape.type === "line"

                            &&

                            isPurple

                            &&

                            dash === "dash"

                        );


                        // ------------------------------------
                        // Rectangle
                        // ------------------------------------

                        var fill =
                            String(
                                shape.fillcolor || ""
                            ).toLowerCase();


                        fill = fill.replace(
                            /\\s+/g,
                            " "
                        );


                        var isFibRectangle = (

                            shape.type === "rect"

                            &&

                            (

                                fill === "#8b5cf6"

                                ||

                                fill ===
                                "rgb(139, 92, 246)"

                                ||

                                fill ===
                                "rgba(139, 92, 246, 1)"

                                ||

                                fill ===
                                "rgba(139,92,246,1)"

                            )

                        );


                        // ------------------------------------
                        // Delete Fibonacci
                        // ------------------------------------

                        if (
                            isFibLine ||
                            isFibRectangle
                        ) {

                            return false;

                        }


                        // ------------------------------------
                        // Keep everything else
                        // ------------------------------------

                        return true;

                    }

                );

        }


        // ====================================================
        // REMOVE FIBONACCI ANNOTATIONS
        // ====================================================

        if (
            Array.isArray(
                newFigure.layout.annotations
            )
        ) {

            newFigure.layout.annotations =
                newFigure.layout.annotations.filter(

                    function(annotation) {

                        var text =
                            String(
                                annotation.text || ""
                            );


                        // Remove:

                        // FIB:0.0%
                        // FIB:23.6%
                        // FIB:38.2%
                        // etc.

                        if (
                            text.indexOf("FIB:") === 0
                        ) {

                            return false;

                        }


                        return true;

                    }

                );

        }


        // ====================================================
        // RETURN TO PAN
        // ====================================================

        newFigure.layout.dragmode =
            "pan";


        return newFigure;

    }
    """,
    # added allow_duplicate=True — select_tool, handle_graph_click,
    # handle_shape_relayout, load_drawing, and clear_all all also
    # write to price-chart.figure, so this clientside callback must
    # declare it too or Dash raises DuplicateCallbackOutput at
    # startup.
    Output("price-chart", "figure", allow_duplicate=True),
    Input("clear-fib-button", "n_clicks"),
    State("price-chart", "figure"),
    prevent_initial_call=True,
)


# ============================================================
# 10. SAVE DRAWING
# ============================================================


@app.callback(
    Output("download-drawing", "data"),
    Input("save-drawing-button", "n_clicks"),
    State("price-chart", "figure"),
    State("price-chart", "relayoutData"),
    prevent_initial_call=True,
)
def save_drawing(clicks, current_figure, relayout_data):

    # ========================================================
    # START WITH THE CURRENT FIGURE
    # ========================================================

    fig = go.Figure(current_figure)

    # ========================================================
    # KEEP ALL CURRENT SHAPES
    #
    # Do NOT replace them with relayoutData["shapes"].
    # That can remove programmatically-created Fibonacci shapes.
    # ========================================================

    shapes = list(fig.layout.shapes or [])

    # ========================================================
    # APPLY INDIVIDUAL INTERACTIVE SHAPE CHANGES
    #
    # This preserves Fibonacci shapes while still saving moved
    # or resized support, resistance, trendline and zone shapes.
    #
    # relayoutData keys can be nested, e.g. "shapes[0].line.color"
    # or "shapes[0].line.width", not just flat keys like
    # "shapes[0].x0". We walk dotted property paths and apply the
    # value to the correct nested attribute rather than naively
    # setattr-ing the dotted string itself (which silently does
    # nothing on a plotly shape object).
    # ========================================================

    if relayout_data:

        for key, value in relayout_data.items():

            if not key.startswith("shapes["):
                continue

            try:
                remainder = key[len("shapes[") :]
                index_string, property_path = remainder.split("].", 1)
                index = int(index_string)
            except (ValueError, IndexError):
                continue

            if index >= len(shapes):
                continue

            shape = shapes[index]

            path_parts = property_path.split(".")

            try:
                if len(path_parts) == 1:
                    setattr(shape, path_parts[0], value)
                else:
                    # Walk down to the second-to-last part (e.g. "line"
                    # for "line.color"), then set the final attribute
                    # on that nested object.
                    target = shape
                    for part in path_parts[:-1]:
                        target = getattr(target, part)
                    setattr(target, path_parts[-1], value)
            except Exception:
                pass

    # ========================================================
    # PUT UPDATED SHAPES BACK INTO FIGURE
    # ========================================================

    fig.update_layout(shapes=shapes)

    # ========================================================
    # EXTRACT DRAWINGS
    # ========================================================

    figure_json = fig.to_plotly_json()
    layout = figure_json.get("layout", {})

    saved_shapes = layout.get("shapes", [])
    saved_annotations = layout.get("annotations", [])

    # ========================================================
    # SAVE DATA
    # ========================================================

    drawing_data = {
        "format": "gold-technical-analysis",
        "version": 1,
        "created": datetime.now().isoformat(),
        "csv_file": FILE_PATH,
        "drawings": {
            "shapes": saved_shapes,
            "annotations": saved_annotations,
        },
    }

    # ========================================================
    # FILE NAME
    # ========================================================

    filename = (
        "gold_drawing_"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + ".json"
    )

    return dcc.send_string(
        json.dumps(drawing_data, indent=2, default=str),
        filename,
    )


# ============================================================
# 11. LOAD DRAWING
# ============================================================


@app.callback(
    Output("price-chart", "figure", allow_duplicate=True),
    Output("tool-info", "children", allow_duplicate=True),
    Input("load-drawing-upload", "contents"),
    State("price-chart", "figure"),
    prevent_initial_call=True,
)
def load_drawing(contents, current_figure):

    if contents is None:
        return (no_update, no_update)

    try:
        # ====================================================
        # DECODE
        # ====================================================

        content_type, content_string = contents.split(",", 1)

        decoded = base64.b64decode(content_string)

        drawing_data = json.loads(decoded.decode("utf-8"))

        # ====================================================
        # VALIDATE
        # ====================================================

        if drawing_data.get("format") != "gold-technical-analysis":
            return (no_update, "Invalid drawing file.")

        drawings = drawing_data.get("drawings", {})

        shapes = drawings.get("shapes", [])

        annotations = drawings.get("annotations", [])

        # ====================================================
        # RESTORE
        # ====================================================

        fig = go.Figure(current_figure)

        fig.update_layout(shapes=shapes, annotations=annotations, dragmode="pan")

        return (fig, (f"Drawing loaded successfully. {len(shapes)} shape(s) restored."))

    except Exception as error:
        return (no_update, f"Unable to load drawing: {error}")


# ============================================================
# 12. CLEAR ALL
# ============================================================


@app.callback(
    Output("price-chart", "figure", allow_duplicate=True),
    Input("clear-button", "n_clicks"),
    prevent_initial_call=True,
)
def clear_all(clicks):

    return create_figure()


# ============================================================
# 13. RUN APP
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
