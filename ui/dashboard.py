"""Dashboard data preparation and rendering."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
import sqlite3
from html import escape
from typing import Any

import streamlit as st

import analytics
from ui.styles import render_empty_state, render_kpi_cards, render_page_heading


_SUMMARY_KEYS = ("order_count", "revenue_pkr", "units_sold")
_TREND_KEYS = ("order_date", "order_count", "revenue_pkr", "units_sold")


def _validate_available_range(available_range: tuple[str, str]) -> tuple[date, date]:
    if (
        not isinstance(available_range, tuple)
        or len(available_range) != 2
        or not all(isinstance(value, str) for value in available_range)
    ):
        raise ValueError("Available order date range is invalid.")
    try:
        minimum = date.fromisoformat(available_range[0])
        maximum = date.fromisoformat(available_range[1])
    except ValueError as error:
        raise ValueError("Available order date range is invalid.") from error
    if (
        minimum.isoformat() != available_range[0]
        or maximum.isoformat() != available_range[1]
        or minimum > maximum
    ):
        raise ValueError("Available order date range is invalid.")
    return minimum, maximum


def validate_date_range(
    start_date: str,
    end_date: str,
    available_range: tuple[str, str],
) -> tuple[str, str]:
    """Validate ISO dates, ordering, and inclusion within available order dates."""
    minimum, maximum = _validate_available_range(available_range)
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except (TypeError, ValueError) as error:
        raise ValueError("Dates must use the valid YYYY-MM-DD format.") from error
    if start.isoformat() != start_date or end.isoformat() != end_date:
        raise ValueError("Dates must use the exact YYYY-MM-DD format.")
    if start > end:
        raise ValueError("Start date must be on or before end date.")
    if start < minimum or end > maximum:
        raise ValueError(
            f"Choose dates between {minimum.isoformat()} and {maximum.isoformat()}."
        )
    return start.isoformat(), end.isoformat()


def _validate_numeric_fields(
    record: dict[str, Any],
    required_keys: tuple[str, ...],
    record_name: str,
) -> None:
    for key in required_keys:
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{record_name} data is missing a whole-number {key} value.")
        if value < 0:
            raise ValueError(f"{record_name} data contains a negative {key} value.")


def _validate_rows(
    rows: list[Any],
    required_keys: tuple[str, ...],
    numeric_keys: tuple[str, ...],
    record_name: str,
) -> None:
    for row in rows:
        if not isinstance(row, dict) or any(key not in row for key in required_keys):
            raise ValueError(f"{record_name} data is incomplete.")
        _validate_numeric_fields(row, numeric_keys, record_name)


def _aggregate_trend(
    daily_rows: list[dict[str, Any]],
    granularity: str,
) -> list[dict[str, Any]]:
    """Aggregate observed daily rows without filling missing dates or periods."""
    aggregated: dict[date, dict[str, int]] = defaultdict(
        lambda: {"order_count": 0, "revenue_pkr": 0, "units_sold": 0}
    )
    for row in daily_rows:
        if not isinstance(row, dict) or any(key not in row for key in _TREND_KEYS):
            raise ValueError("Sales trend data is incomplete.")
        try:
            observed_date = date.fromisoformat(row["order_date"])
        except (TypeError, ValueError) as error:
            raise ValueError("Sales trend contains an invalid date.") from error
        if observed_date.isoformat() != row["order_date"]:
            raise ValueError("Sales trend dates must use YYYY-MM-DD format.")
        _validate_numeric_fields(row, _TREND_KEYS[1:], "Sales trend")

        if granularity == "day":
            period = observed_date
        elif granularity == "week":
            period = observed_date - timedelta(days=observed_date.weekday())
        elif granularity == "month":
            period = date(observed_date.year, observed_date.month, 1)
        else:
            raise ValueError("Trend granularity must be day, week, or month.")
        bucket = aggregated[period]
        for key in ("order_count", "revenue_pkr", "units_sold"):
            bucket[key] += row[key]

    return [
        {"period": period.isoformat(), **aggregated[period]}
        for period in sorted(aggregated)
    ]


def _trend_granularity(start_date: str, end_date: str) -> str:
    span_days = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1
    if span_days <= 31:
        return "day"
    if span_days <= 180:
        return "week"
    return "month"


def prepare_dashboard_data(
    start_date: str,
    end_date: str,
    top_limit: int = 10,
) -> dict[str, Any]:
    """Load and validate dashboard values through the read-only analytics API."""
    if isinstance(top_limit, bool) or not isinstance(top_limit, int):
        raise ValueError("Top product/customer limit must be an integer.")
    if top_limit not in (5, 10, 15, 20, 50):
        raise ValueError("Choose a supported dashboard limit.")

    available_range = analytics.order_date_bounds()
    if available_range is None:
        return {
            "available_range": None,
            "summary": {"order_count": 0, "revenue_pkr": 0, "units_sold": 0},
            "average_order_value_pkr": None,
            "granularity": None,
            "trend": [],
            "top_products": [],
            "low_stock_items": [],
            "top_customers": [],
        }

    start, end = validate_date_range(start_date, end_date, available_range)
    summary = analytics.sales_summary(start, end)
    if not isinstance(summary, dict) or any(key not in summary for key in _SUMMARY_KEYS):
        raise ValueError("Sales summary data is incomplete.")
    _validate_numeric_fields(summary, _SUMMARY_KEYS, "Sales summary")
    granularity = _trend_granularity(start, end)
    daily_rows = analytics.daily_sales_summary(start, end)
    if not isinstance(daily_rows, list):
        raise ValueError("Sales trend data must be a list.")
    trend = _aggregate_trend(daily_rows, granularity)
    products = analytics.top_products(start, end, limit=top_limit)
    inventory = analytics.low_stock_items()
    customers = analytics.top_customers(start, end, limit=top_limit)
    if not all(isinstance(rows, list) for rows in (products, inventory, customers)):
        raise ValueError("Dashboard ranking or inventory data is invalid.")
    _validate_rows(
        products,
        ("name", "units_sold", "revenue_pkr"),
        ("units_sold", "revenue_pkr"),
        "Top products",
    )
    _validate_rows(
        inventory,
        ("name", "category", "quantity", "reorder_level"),
        ("quantity", "reorder_level"),
        "Inventory",
    )
    _validate_rows(
        customers,
        ("name", "order_count", "spend_pkr"),
        ("order_count", "spend_pkr"),
        "Top customers",
    )

    average_order_value = (
        summary["revenue_pkr"] / summary["order_count"]
        if summary["order_count"]
        else None
    )
    return {
        "available_range": available_range,
        "summary": summary,
        "average_order_value_pkr": average_order_value,
        "granularity": granularity,
        "trend": trend,
        "top_products": products,
        "low_stock_items": inventory,
        "top_customers": customers,
    }


def render_dashboard_page() -> None:
    """Render the interactive read-only analytics dashboard."""
    render_page_heading(
        "Business overview",
        "Track completed sales, product performance, customer value, and stock health.",
        "DASHBOARD",
    )
    try:
        available_range = analytics.order_date_bounds()
    except (OSError, sqlite3.Error, RuntimeError):
        st.error("Could not load dashboard data. Verify the database is available.")
        return
    except (ValueError, TypeError) as error:
        st.error(f"Could not load dashboard data: {error}")
        return

    if available_range is None:
        st.info("No order data is available yet.")
        return

    try:
        minimum, maximum = _validate_available_range(available_range)
    except ValueError as error:
        st.error(f"Could not load dashboard data: {error}")
        return
    date_column, ranking_column = st.columns((2, 1))
    with date_column:
        selected_range = st.date_input(
            "Order date range",
            value=(minimum, maximum),
            min_value=minimum,
            max_value=maximum,
            format="YYYY-MM-DD",
        )
    if not isinstance(selected_range, (tuple, list)) or len(selected_range) != 2:
        st.info("Select both a start date and an end date.")
        return
    start_date, end_date = (value.isoformat() for value in selected_range)
    with ranking_column:
        product_limit = st.selectbox("Ranking size", (5, 10, 15, 20, 50), index=1)

    try:
        data = prepare_dashboard_data(start_date, end_date, product_limit)
    except (OSError, sqlite3.Error, RuntimeError):
        st.error("Could not load dashboard data. Verify the database is available.")
        return
    except (ValueError, TypeError) as error:
        st.error(f"Could not load dashboard data: {error}")
        return

    summary = data["summary"]
    average_value = data["average_order_value_pkr"]
    render_kpi_cards(
        [
            ("Completed orders", f"{summary['order_count']:,}"),
            ("Revenue (PKR)", f"{summary['revenue_pkr']:,}"),
            ("Units sold", f"{summary['units_sold']:,}"),
            (
                "Average order value (PKR)",
                f"{average_value:,.0f}" if average_value is not None else "—",
            ),
        ]
    )

    st.markdown('<div class="bb-section-label">Sales overview</div>', unsafe_allow_html=True)
    trend = data["trend"]
    if trend:
        import pandas as pd
        import plotly.express as px

        chart_data = pd.DataFrame(trend)
        chart_data["period"] = pd.to_datetime(chart_data["period"])
        chart_columns = st.columns(2)
        chart_columns[0].caption(f"Revenue by {data['granularity']}")
        revenue_chart = px.line(
            chart_data,
            x="period",
            y="revenue_pkr",
            markers=True,
            labels={"period": "Date", "revenue_pkr": "Revenue (PKR)"},
        )
        revenue_chart.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=12, r=16, t=16, b=12),
            font=dict(color="#c5d2e1"),
            hoverlabel=dict(bgcolor="#142235", bordercolor="#45d6c5"),
            transition=dict(duration=250),
        )
        revenue_chart.update_traces(
            line=dict(color="#45d6c5", width=3),
            marker=dict(color="#45d6c5", size=7),
            fill="tozeroy",
            fillcolor="rgba(69, 214, 197, .08)",
        )
        revenue_chart.update_xaxes(
            gridcolor="rgba(147, 164, 184, .14)",
            zerolinecolor="rgba(147, 164, 184, .22)",
        )
        revenue_chart.update_yaxes(
            gridcolor="rgba(147, 164, 184, .14)",
            zerolinecolor="rgba(147, 164, 184, .22)",
        )
        chart_columns[0].plotly_chart(
            revenue_chart,
            config={"scrollZoom": False},
            width="stretch",
        )
        chart_columns[1].caption(f"Completed orders by {data['granularity']}")
        orders_chart = px.line(
            chart_data,
            x="period",
            y="order_count",
            markers=True,
            labels={"period": "Date", "order_count": "Completed orders"},
        )
        orders_chart.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=12, r=16, t=16, b=12),
            font=dict(color="#c5d2e1"),
            hoverlabel=dict(bgcolor="#142235", bordercolor="#8fa9c4"),
            transition=dict(duration=250),
        )
        orders_chart.update_traces(
            line=dict(color="#8fa9c4", width=3),
            marker=dict(color="#8fa9c4", size=7),
        )
        orders_chart.update_xaxes(
            gridcolor="rgba(147, 164, 184, .14)",
            zerolinecolor="rgba(147, 164, 184, .22)",
        )
        orders_chart.update_yaxes(
            gridcolor="rgba(147, 164, 184, .14)",
            zerolinecolor="rgba(147, 164, 184, .22)",
        )
        chart_columns[1].plotly_chart(
            orders_chart,
            config={"scrollZoom": False},
            width="stretch",
        )
    else:
        render_empty_state("No completed sales were recorded in this date range.")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="bb-section-label">Product performance</div>', unsafe_allow_html=True)
        if data["top_products"]:
            import pandas as pd
            import plotly.express as px

            product_data = pd.DataFrame(data["top_products"])
            products_chart = px.bar(
                product_data.sort_values("units_sold", ascending=True),
                x="units_sold",
                y="name",
                orientation="h",
                hover_data={"revenue_pkr": ":,", "units_sold": True},
                labels={
                    "units_sold": "Units sold",
                    "name": "Product",
                    "revenue_pkr": "Revenue (PKR)",
                },
            )
            products_chart.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=12, t=8, b=12),
                font=dict(color="#c5d2e1"),
                showlegend=False,
                transition=dict(duration=250),
                yaxis=dict(title=None, automargin=True),
                xaxis=dict(gridcolor="rgba(147, 164, 184, .14)"),
            )
            products_chart.update_traces(
                marker_color="#45d6c5",
                marker_line_width=0,
            )
            st.plotly_chart(
                products_chart,
                config={"scrollZoom": False},
                width="stretch",
            )
        else:
            render_empty_state("No product sales in this date range.")

    with right:
        st.markdown('<div class="bb-section-label">Customer insights</div>', unsafe_allow_html=True)
        if data["top_customers"]:
            st.dataframe(
                [
                    {
                        "Customer": item["name"],
                        "Orders": item["order_count"],
                        "Spending (PKR)": item["spend_pkr"],
                    }
                    for item in data["top_customers"]
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            render_empty_state("No completed customer orders in this date range.")

    st.markdown('<div class="bb-section-label">Inventory health</div>', unsafe_allow_html=True)
    inventory = data["low_stock_items"]
    if inventory:
        for start in range(0, len(inventory), 3):
            columns = st.columns(3)
            for column, item in zip(columns, inventory[start : start + 3]):
                out_of_stock = item["quantity"] == 0
                status = "Out of stock" if out_of_stock else "Low stock"
                status_class = "bb-status-danger" if out_of_stock else "bb-status-warning"
                with column:
                    st.markdown(
                        (
                            '<div class="bb-kpi">'
                            f'<div class="bb-status {status_class}">{escape(status)}</div>'
                            f'<div class="bb-kpi-label" style="margin-top:.7rem">{escape(str(item["category"]))}</div>'
                            f'<div class="bb-kpi-value" style="font-size:1.05rem">{escape(str(item["name"]))}</div>'
                            f'<div class="bb-kpi-label" style="margin-top:.55rem">'
                            f'Stock <strong>{item["quantity"]:,}</strong>'
                            f' &nbsp;·&nbsp; Reorder at <strong>{item["reorder_level"]:,}</strong>'
                            "</div></div>"
                        ),
                        unsafe_allow_html=True,
                    )
    else:
        render_empty_state("No products are currently at or below their reorder level.")
