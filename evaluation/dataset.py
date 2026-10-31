"""Predefined evaluation prompts with expectations grounded in the local database."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import analytics
from seed import COLD_BEVERAGE_NAMES, DECLINING_PRODUCT_NAME, POPULAR_PRODUCT_NAME


Behavior = Literal["answer", "answer_without_tools", "clarify", "refuse"]


@dataclass(frozen=True)
class ExpectedToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_tool_calls: tuple[ExpectedToolCall, ...]
    required_fact_groups: tuple[tuple[str, ...], ...]
    expected_numbers: tuple[int, ...] = ()
    behavior: Behavior = "answer"
    description: str = ""


def _read_only_connection(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path).expanduser().resolve()
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.row_factory = sqlite3.Row
    return connection


def _expected(
    name: str,
    **arguments: Any,
) -> ExpectedToolCall:
    return ExpectedToolCall(name=name, arguments=arguments)


def _number_group(value: int, labels: tuple[str, ...]) -> tuple[str, ...]:
    formatted = f"{value:,}"
    alternatives = list(
        alternative
        for label in labels
        for alternative in (f"{formatted} {label}", f"{value} {label}", formatted, str(value))
    )
    if "PKR" in labels:
        alternatives.extend((f"PKR {formatted}", f"PKR {value}"))
    return tuple(dict.fromkeys(alternatives))


def _summary_facts(summary: dict[str, int]) -> tuple[tuple[str, ...], ...]:
    return (
        _number_group(summary["order_count"], ("completed orders", "orders")),
        _number_group(summary["revenue_pkr"], ("PKR", "rupees")),
        _number_group(summary["units_sold"], ("units", "items")),
    )


def _product_facts(
    products: list[dict[str, Any]],
    count: int,
) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    for product in products[:count]:
        groups.append((product["name"].casefold(),))
        groups.append(
            _number_group(
                int(product["units_sold"]),
                ("units", "sold", product["name"].casefold()),
            )
        )
        groups.append(
            _number_group(
                int(product["revenue_pkr"]),
                ("PKR", "revenue"),
            )
        )
    return tuple(groups)


def _product_numbers(
    products: list[dict[str, Any]],
    count: int,
) -> tuple[int, ...]:
    return tuple(
        value
        for product in products[:count]
        for value in (int(product["units_sold"]), int(product["revenue_pkr"]))
    )


def _customer_facts(
    customers: list[dict[str, Any]],
) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    for customer in customers:
        groups.append((str(customer["name"]).casefold(),))
        groups.append(
            _number_group(
                int(customer["spend_pkr"]),
                ("PKR", "rupees", "spent"),
            )
        )
    return tuple(groups)


def _units_for(products: list[dict[str, Any]], product_names: set[str]) -> int:
    return sum(
        int(product["units_sold"])
        for product in products
        if product["name"] in product_names
    )


def load_evaluation_cases(
    database_path: str | Path = analytics.DEFAULT_DATABASE_PATH,
) -> list[EvaluationCase]:
    """Build 25 repeatable cases and derive numeric expectations from this database."""
    with closing(_read_only_connection(database_path)) as connection:
        bounds = connection.execute(
            "SELECT MIN(order_date) AS first_day, MAX(order_date) AS last_day "
            "FROM orders"
        ).fetchone()
        if not bounds["first_day"] or not bounds["last_day"]:
            raise ValueError("The evaluation database contains no dated orders.")
        first_day = date.fromisoformat(bounds["first_day"])
        last_day = date.fromisoformat(bounds["last_day"])

        def find_last_weekday(weekday: int) -> date:
            candidate = last_day
            while candidate.weekday() != weekday:
                candidate -= timedelta(days=1)
            return candidate

        friday = find_last_weekday(4)
        saturday = friday + timedelta(days=1)
        tuesday = friday - timedelta(days=3)
        wednesday = friday - timedelta(days=2)
        if saturday > last_day:
            friday -= timedelta(days=7)
            saturday = friday + timedelta(days=1)
            tuesday = friday - timedelta(days=3)
            wednesday = friday - timedelta(days=2)

        total_orders = int(
            connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        )
        cancelled_refunded = int(
            connection.execute(
                "SELECT COUNT(*) FROM orders "
                "WHERE status IN ('cancelled', 'refunded')"
            ).fetchone()[0]
        )
        weekday_counts = connection.execute(
            "SELECT CAST(strftime('%w', order_date) AS INTEGER) AS weekday, "
            "AVG(order_count) AS average_orders "
            "FROM (SELECT order_date, COUNT(*) AS order_count "
            "FROM orders GROUP BY order_date) "
            "GROUP BY weekday"
        ).fetchall()
        weekend_average = sum(
            float(row["average_orders"]) for row in weekday_counts if row["weekday"] in (5, 6)
        ) / 2
        weekday_average = sum(
            float(row["average_orders"]) for row in weekday_counts if row["weekday"] in (0, 1, 2, 3, 4)
        ) / 5

        first_name = connection.execute(
            "SELECT substr(name, 1, instr(name || ' ', ' ') - 1) AS first_name, "
            "COUNT(*) AS customer_count FROM customers "
            "GROUP BY first_name HAVING COUNT(*) > 1 "
            "ORDER BY customer_count DESC, first_name LIMIT 1"
        ).fetchone()["first_name"]

        selected_customer = connection.execute(
            "SELECT id, name FROM customers WHERE id = 1"
        ).fetchone()
        if selected_customer is None:
            raise ValueError("Evaluation requires the seeded customer with ID 1.")

    full_summary = analytics.sales_summary(
        first_day.isoformat(), last_day.isoformat(), database_path=database_path
    )
    day_summary = analytics.sales_summary(
        last_day.isoformat(), last_day.isoformat(), database_path=database_path
    )
    popular_last10_start = last_day - timedelta(days=9)
    popular_prev10_start = last_day - timedelta(days=19)
    popular_prev10_end = popular_last10_start - timedelta(days=1)
    popular_last10 = analytics.top_products(
        popular_last10_start.isoformat(),
        last_day.isoformat(),
        limit=50,
        database_path=database_path,
    )
    popular_prev10 = analytics.top_products(
        popular_prev10_start.isoformat(),
        popular_prev10_end.isoformat(),
        limit=50,
        database_path=database_path,
    )
    popular_last_units = _units_for(popular_last10, {POPULAR_PRODUCT_NAME})
    popular_prev_units = _units_for(popular_prev10, {POPULAR_PRODUCT_NAME})
    cold_first = analytics.top_products(
        first_day.isoformat(),
        (first_day + timedelta(days=29)).isoformat(),
        limit=50,
        database_path=database_path,
    )
    cold_final_start = last_day - timedelta(days=29)
    cold_final = analytics.top_products(
        cold_final_start.isoformat(),
        last_day.isoformat(),
        limit=50,
        database_path=database_path,
    )
    cold_first_units = _units_for(
        cold_first,
        set(COLD_BEVERAGE_NAMES),
    )
    cold_final_units = _units_for(
        cold_final,
        set(COLD_BEVERAGE_NAMES),
    )
    decline_first = analytics.top_products(
        first_day.isoformat(),
        (first_day + timedelta(days=29)).isoformat(),
        limit=50,
        database_path=database_path,
    )
    decline_final = analytics.top_products(
        cold_final_start.isoformat(),
        last_day.isoformat(),
        limit=50,
        database_path=database_path,
    )
    decline_first_units = _units_for(decline_first, {DECLINING_PRODUCT_NAME})
    decline_final_units = _units_for(decline_final, {DECLINING_PRODUCT_NAME})
    fri_sat_summary = analytics.sales_summary(
        friday.isoformat(), saturday.isoformat(), database_path=database_path
    )
    tue_wed_summary = analytics.sales_summary(
        tuesday.isoformat(), wednesday.isoformat(), database_path=database_path
    )
    weekly_history = analytics.customer_order_history(
        int(selected_customer["id"]), limit=50, database_path=database_path
    )
    weekly_completed = [
        row for row in weekly_history if row["status"] == "completed"
    ]
    weekly_dates = [
        date.fromisoformat(row["order_date"])
        for row in sorted(weekly_completed, key=lambda row: row["order_date"])
    ]
    weekly_intervals = [
        (later - earlier).days
        for earlier, later in zip(weekly_dates, weekly_dates[1:])
    ]
    weekly_interval_is_consistent = bool(weekly_intervals) and all(
        interval == 7 for interval in weekly_intervals
    )
    same_weekly_amount = len(
        {row["total_pkr"] for row in weekly_completed}
    ) == 1
    top_full_customers = analytics.top_customers(
        first_day.isoformat(), last_day.isoformat(), limit=3, database_path=database_path
    )
    latest_week_start = last_day - timedelta(days=6)
    latest_week_customers = analytics.top_customers(
        latest_week_start.isoformat(),
        last_day.isoformat(),
        limit=1,
        database_path=database_path,
    )
    final_month_customers = analytics.top_customers(
        cold_final_start.isoformat(),
        last_day.isoformat(),
        limit=1,
        database_path=database_path,
    )
    latest_history = analytics.customer_order_history(
        1, limit=20, database_path=database_path
    )
    low_stock = analytics.low_stock_items(database_path=database_path)
    top_products_full = analytics.top_products(
        first_day.isoformat(), last_day.isoformat(), limit=3, database_path=database_path
    )
    top_products_latest = analytics.top_products(
        (last_day - timedelta(days=9)).isoformat(),
        last_day.isoformat(),
        limit=3,
        database_path=database_path,
    )
    first_month_summary = analytics.sales_summary(
        first_day.isoformat(),
        (first_day + timedelta(days=29)).isoformat(),
        database_path=database_path,
    )
    final_month_summary = analytics.sales_summary(
        cold_final_start.isoformat(),
        last_day.isoformat(),
        database_path=database_path,
    )
    no_orders_summary = analytics.sales_summary(
        (last_day + timedelta(days=1)).isoformat(),
        (last_day + timedelta(days=10)).isoformat(),
        database_path=database_path,
    )
    customer_id = int(selected_customer["id"])
    history_summary = analytics.top_customers(
        first_day.isoformat(), last_day.isoformat(), limit=50, database_path=database_path
    )
    target_spend = next(
        (row["spend_pkr"] for row in history_summary if row["customer_id"] == customer_id),
        0,
    )

    summary_call = _expected(
        "sales_summary",
        start_date=first_day.isoformat(),
        end_date=last_day.isoformat(),
    )
    full_top_products_call = _expected(
        "top_products",
        start_date=first_day.isoformat(),
        end_date=last_day.isoformat(),
        limit=3,
    )
    cases = [
        EvaluationCase(
            "sales-full-period",
            f"Summarize completed sales from {first_day} through {last_day}.",
            (summary_call,),
            _summary_facts(full_summary),
            tuple(full_summary.values()),
            description="Inclusive full-period sales totals.",
        ),
        EvaluationCase(
            "sales-final-day-boundary",
            f"How many completed orders, PKR revenue, and units were sold on {last_day}?",
            (_expected("sales_summary", start_date=last_day.isoformat(), end_date=last_day.isoformat()),),
            _summary_facts(day_summary),
            tuple(day_summary.values()),
            description="Checks inclusive single-day date boundaries.",
        ),
        EvaluationCase(
            "top-products-full-period",
            f"Name the three top products by units sold from {first_day} through {last_day}, including each unit count.",
            (full_top_products_call,),
            _product_facts(top_products_full, 3),
            _product_numbers(top_products_full, 3),
            description="Ranks by the existing analytics-layer ordering.",
        ),
        EvaluationCase(
            "top-products-recent",
            f"What were the top three products by units sold over the last 10 dataset days, {last_day - timedelta(days=9)} through {last_day}?",
            (_expected("top_products", start_date=(last_day-timedelta(days=9)).isoformat(), end_date=last_day.isoformat(), limit=3),),
            _product_facts(top_products_latest, 3),
            _product_numbers(top_products_latest, 3),
            description="Recent product ranking.",
        ),
        EvaluationCase(
            "low-stock-current",
            "Which products are at or below their reorder levels? Include current and reorder quantities.",
            (_expected("low_stock_items"),),
            tuple(
                (f"{row['name'].casefold()}",)
                + _number_group(int(row["quantity"]), ("units",))
                + _number_group(int(row["reorder_level"]), ("reorder level",))
                for row in low_stock
            ),
            tuple(
                value
                for row in low_stock
                for value in (int(row["quantity"]), int(row["reorder_level"]))
            ),
            description="Current inventory threshold check.",
        ),
        EvaluationCase(
            "top-customer-full-period",
            f"Who spent the most on completed orders between {first_day} and {last_day}, and how much?",
            (_expected("top_customers", start_date=first_day.isoformat(), end_date=last_day.isoformat(), limit=1),),
            _customer_facts(top_full_customers[:1]),
            (int(top_full_customers[0]["spend_pkr"]),),
            description="Overall completed-order spending leader.",
        ),
        EvaluationCase(
            "customer-history-exact-id",
            f"Show the recent order history for customer ID {customer_id}, including dates and statuses.",
            (_expected("customer_order_history", customer_id=customer_id, limit=20),),
            (
                (f"{len(latest_history)} orders", str(len(latest_history))),
                *tuple((row["status"],) for row in latest_history),
            ),
            (len(latest_history),),
            description="Uses an exact numeric customer ID, not a name search.",
        ),
        EvaluationCase(
            "weekly-recurring-orders",
            f"Does customer ID {customer_id} place a recurring weekly order? Report the interval, completed-order count, and whether the order total repeats.",
            (_expected("customer_order_history", customer_id=customer_id, limit=50),),
            (
                (selected_customer["name"].casefold(),),
                (
                    ("every 7 days", "every seven days", "weekly", "7-day interval")
                    if weekly_interval_is_consistent
                    else ("not every 7 days", "not weekly", "interval varies")
                ),
                (f"{len(weekly_completed)} completed orders", f"{len(weekly_completed)} orders"),
                (
                    ("same total", "identical total", "same order total")
                    if same_weekly_amount
                    else ("different totals", "totals vary", "not the same total")
                ),
            ),
            (7, len(weekly_completed), int(weekly_completed[0]["total_pkr"])),
            description="Weekly cadence and repeated order total are verified from returned history.",
        ),
        EvaluationCase(
            "weekend-versus-weekday-volume",
            f"Compare completed-order volume for Friday {friday} and Saturday {saturday} with Tuesday {tuesday} and Wednesday {wednesday}.",
            (
                _expected("sales_summary", start_date=friday.isoformat(), end_date=saturday.isoformat()),
                _expected("sales_summary", start_date=tuesday.isoformat(), end_date=wednesday.isoformat()),
            ),
            (
                _number_group(fri_sat_summary["order_count"], ("completed orders", "orders")),
                _number_group(tue_wed_summary["order_count"], ("completed orders", "orders")),
                ("friday",),
                ("saturday",),
                ("tuesday",),
                ("wednesday",),
            ),
            (fri_sat_summary["order_count"], tue_wed_summary["order_count"]),
            description=f"Dataset's verified mean daily order volume is {weekend_average:.2f} on Friday/Saturday vs {weekday_average:.2f} on weekdays.",
        ),
        EvaluationCase(
            "cold-beverage-demand-rises",
            f"Compare total completed units for beverage products in the first 30 days ({first_day} through {first_day + timedelta(days=29)}) with the final 30 days ({cold_final_start} through {last_day}). Did beverage demand rise?",
            (
                _expected("top_products", start_date=first_day.isoformat(), end_date=(first_day+timedelta(days=29)).isoformat(), limit=50),
                _expected("top_products", start_date=cold_final_start.isoformat(), end_date=last_day.isoformat(), limit=50),
            ),
            (
                _number_group(cold_first_units, ("beverage units", "units")),
                _number_group(cold_final_units, ("beverage units", "units")),
                ("increased", "rose", "higher"),
            ),
            (cold_first_units, cold_final_units),
            description="Cold-beverage demand comparison over equal inclusive windows.",
        ),
        EvaluationCase(
            "seasonal-product-declines",
            f"Compare completed units for {DECLINING_PRODUCT_NAME} in the first 30 days and final 30 days of the dataset. Did demand decline?",
            (
                _expected("top_products", start_date=first_day.isoformat(), end_date=(first_day+timedelta(days=29)).isoformat(), limit=50),
                _expected("top_products", start_date=cold_final_start.isoformat(), end_date=last_day.isoformat(), limit=50),
            ),
            (
                (DECLINING_PRODUCT_NAME.casefold(),),
                _number_group(decline_first_units, ("units",)),
                _number_group(decline_final_units, ("units",)),
                ("declined", "decreased", "lower"),
            ),
            (decline_first_units, decline_final_units),
            description="Seasonal household product demand comparison.",
        ),
        EvaluationCase(
            "exclude-cancelled-refunded",
            f"Report completed sales for the whole dataset, {first_day} through {last_day}. Do not count cancelled or refunded orders as sales.",
            (summary_call,),
            (
                *_summary_facts(full_summary),
                ("cancelled and refunded orders are excluded", "excluding cancelled and refunded", "completed orders only"),
            ),
            tuple(full_summary.values()),
            description=f"Database has {total_orders} orders total, including {cancelled_refunded} cancelled/refunded.",
        ),
        EvaluationCase(
            "combined-sales-products-customers",
            f"For the final 30 days ({cold_final_start} through {last_day}), give the completed sales summary, top three products, and highest-spending customer.",
            (
                _expected("sales_summary", start_date=cold_final_start.isoformat(), end_date=last_day.isoformat()),
                _expected("top_products", start_date=cold_final_start.isoformat(), end_date=last_day.isoformat(), limit=3),
                _expected("top_customers", start_date=cold_final_start.isoformat(), end_date=last_day.isoformat(), limit=1),
            ),
            (
                *_summary_facts(final_month_summary),
                *_product_facts(top_products_latest, 3),
                *_customer_facts(final_month_customers[:1]),
            ),
            (
                *tuple(final_month_summary.values()),
                *_product_numbers(top_products_latest, 3),
                *(int(row["spend_pkr"]) for row in final_month_customers[:1]),
            ),
            description="Requires three distinct read-only analytics tools.",
        ),
        EvaluationCase(
            "clarify-ambiguous-first-name",
            f"Find order history for {first_name}.",
            (),
            (("which customer", "customer id", "customer ID", "please clarify", "which one"),),
            behavior="clarify",
            description="A first name is shared; no name-based history tool exists.",
        ),
        EvaluationCase(
            "clarify-invalid-calendar-date",
            "Summarize completed sales for 2026-02-30 through 2026-03-01.",
            (),
            (("valid date", "invalid date", "correct date", "please clarify", "cannot use"),),
            behavior="clarify",
            description="Rejects an impossible calendar date without calling a tool.",
        ),
        EvaluationCase(
            "clarify-reversed-date-range",
            f"Summarize sales from {last_day} through {first_day}.",
            (),
            (("start date", "date range", "before", "reverse", "please clarify"),),
            behavior="clarify",
            description="Rejects a reversed date range without calling a tool.",
        ),
        EvaluationCase(
            "empty-future-date-range",
            f"How many completed orders occurred on {last_day + timedelta(days=1)}?",
            (_expected("sales_summary", start_date=(last_day+timedelta(days=1)).isoformat(), end_date=(last_day+timedelta(days=1)).isoformat()),),
            (
                ("0 completed orders", "zero completed orders", "no completed orders"),
                ("0 PKR", "PKR 0", "no revenue"),
            ),
            (0,),
            description="A valid but empty range should return zeros.",
        ),
        EvaluationCase(
            "no-tool-general-knowledge",
            "What is the capital of France?",
            (),
            (("paris",),),
            behavior="answer_without_tools",
            description="Does not require any business data.",
        ),
        EvaluationCase(
            "unavailable-customer-age",
            "What is the average age of our customers?",
            (),
            (("age is not available", "do not record ages", "not recorded", "not in the database", "cannot determine"),),
            behavior="refuse",
            description="Customer age is not a field in the schema.",
        ),
        EvaluationCase(
            "clarify-customer-phone-lookup",
            "Find customer order history for phone number 0300-0000001.",
            (),
            (("customer id", "customer ID", "please provide", "cannot search by phone"),),
            behavior="clarify",
            description="History supports exact numeric ID only.",
        ),
        EvaluationCase(
            "missing-customer-history",
            "Show recent orders for customer ID 999999.",
            (_expected("customer_order_history", customer_id=999999, limit=20),),
            (("no orders", "no order history", "no matching orders", "empty history"),),
            behavior="answer",
            description="Valid but absent ID returns an empty result.",
        ),
        EvaluationCase(
            "top-customer-latest-week",
            f"Who spent the most on completed orders in the last seven days, {latest_week_start} through {last_day}?",
            (_expected("top_customers", start_date=latest_week_start.isoformat(), end_date=last_day.isoformat(), limit=1),),
            _customer_facts(latest_week_customers[:1]),
            (int(latest_week_customers[0]["spend_pkr"]),),
            description="A date-bounded customer ranking.",
        ),
        EvaluationCase(
            "first-month-sales",
            f"Give completed order count, revenue, and units from {first_day} through {first_day + timedelta(days=29)}.",
            (_expected("sales_summary", start_date=first_day.isoformat(), end_date=(first_day+timedelta(days=29)).isoformat()),),
            _summary_facts(first_month_summary),
            tuple(first_month_summary.values()),
            description="Checks a bounded inclusive date window.",
        ),
        EvaluationCase(
            "final-month-sales",
            f"Give completed order count, revenue, and units from {cold_final_start} through {last_day}.",
            (_expected("sales_summary", start_date=cold_final_start.isoformat(), end_date=last_day.isoformat()),),
            _summary_facts(final_month_summary),
            tuple(final_month_summary.values()),
            description="Checks the recent inclusive date window.",
        ),
        EvaluationCase(
            "exact-history-includes-all-statuses",
            f"For customer ID {customer_id}, show the order history and every status actually present.",
            (_expected("customer_order_history", customer_id=customer_id, limit=20),),
            tuple((status,) for status in sorted({row["status"] for row in latest_history})),
            tuple(int(row["total_pkr"]) for row in latest_history),
            description="Order history includes statuses excluded from sales metrics.",
        ),
        EvaluationCase(
            "ambiguous-first-name-is-not-a-customer-id",
            f"There are multiple customers named {first_name}. Can you pick the right customer and show their orders?",
            (),
            (("which customer", "customer id", "please specify", "please clarify"),),
            behavior="clarify",
            description="The agent must not guess an identity from a shared first name.",
        ),
        EvaluationCase(
            "dataset-window-empty-result",
            f"Summarize sales from {last_day + timedelta(days=1)} through {last_day + timedelta(days=10)}.",
            (_expected("sales_summary", start_date=(last_day+timedelta(days=1)).isoformat(), end_date=(last_day+timedelta(days=10)).isoformat()),),
            _summary_facts(no_orders_summary),
            tuple(no_orders_summary.values()),
            description="Checks a valid date range with no generated records.",
        ),
    ]
    if len(cases) < 20:
        raise AssertionError("The evaluation dataset must contain at least 20 cases.")
    if len({case.case_id for case in cases}) != len(cases):
        raise AssertionError("Evaluation case IDs must be unique.")
    return cases
