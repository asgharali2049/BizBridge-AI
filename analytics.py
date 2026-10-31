"""Read-only analytics queries for the BizBridge AI SQLite database.

All monetary result fields are whole PKR. Functions return dictionaries with
stable keys or lists of those dictionaries; date ranges are inclusive.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parent / "shop.db"
MAX_LIMIT = 50


def _validate_date(value: str, argument_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{argument_name} must be a YYYY-MM-DD string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{argument_name} must be a valid YYYY-MM-DD date") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{argument_name} must use the exact YYYY-MM-DD format")
    return value


def _validate_date_range(start_date: str, end_date: str) -> tuple[str, str]:
    start = _validate_date(start_date, "start_date")
    end = _validate_date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    return start, end


def _validate_customer_id(customer_id: int) -> int:
    if isinstance(customer_id, bool) or not isinstance(customer_id, int):
        raise TypeError("customer_id must be a positive integer")
    if customer_id <= 0:
        raise ValueError("customer_id must be a positive integer")
    return customer_id


def _validate_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    return min(limit, MAX_LIMIT)


def _connect_readonly(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path).expanduser().resolve()
    uri = f"{path.as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        if connection.execute("PRAGMA foreign_keys").fetchone() != (1,):
            raise RuntimeError("SQLite foreign-key enforcement could not be enabled")
        connection.row_factory = sqlite3.Row
    except Exception:
        connection.close()
        raise
    return connection


def _fetch_dicts(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query, parameters).fetchall()]


def order_date_bounds(
    *,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> tuple[str, str] | None:
    """Return the earliest and latest order dates, or None when there are no orders."""
    query = "SELECT MIN(order_date), MAX(order_date) FROM orders"
    with closing(_connect_readonly(database_path)) as connection:
        row = connection.execute(query).fetchone()
        if row[0] is None or row[1] is None:
            return None
        return row[0], row[1]


def daily_sales_summary(
    start_date: str,
    end_date: str,
    *,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Return daily completed sales only for dates with completed orders."""
    start, end = _validate_date_range(start_date, end_date)
    query = """
        WITH daily_orders AS (
            SELECT
                order_date,
                COUNT(*) AS order_count,
                SUM(total) AS revenue_pkr
            FROM orders
            WHERE status = 'completed'
              AND order_date BETWEEN ? AND ?
            GROUP BY order_date
        ),
        daily_units AS (
            SELECT
                o.order_date,
                SUM(i.quantity) AS units_sold
            FROM orders AS o
            JOIN order_items AS i ON i.order_id = o.id
            WHERE o.status = 'completed'
              AND o.order_date BETWEEN ? AND ?
            GROUP BY o.order_date
        )
        SELECT
            daily_orders.order_date,
            daily_orders.order_count,
            daily_orders.revenue_pkr,
            COALESCE(daily_units.units_sold, 0) AS units_sold
        FROM daily_orders
        LEFT JOIN daily_units ON daily_units.order_date = daily_orders.order_date
        ORDER BY daily_orders.order_date
    """
    with closing(_connect_readonly(database_path)) as connection:
        return _fetch_dicts(connection, query, (start, end, start, end))


def sales_summary(
    start_date: str,
    end_date: str,
    *,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, int]:
    """Return completed order count, revenue_pkr, and units_sold for inclusive dates."""
    start, end = _validate_date_range(start_date, end_date)
    query = """
        SELECT
            COUNT(*) AS order_count,
            COALESCE(SUM(o.total), 0) AS revenue_pkr,
            COALESCE((
                SELECT SUM(i.quantity)
                FROM order_items AS i
                JOIN orders AS item_order ON item_order.id = i.order_id
                WHERE item_order.status = 'completed'
                  AND item_order.order_date BETWEEN ? AND ?
            ), 0) AS units_sold
        FROM orders AS o
        WHERE o.status = 'completed'
          AND o.order_date BETWEEN ? AND ?
    """
    with closing(_connect_readonly(database_path)) as connection:
        row = connection.execute(query, (start, end, start, end)).fetchone()
        return {
            "order_count": row["order_count"],
            "revenue_pkr": row["revenue_pkr"],
            "units_sold": row["units_sold"],
        }


def top_products(
    start_date: str,
    end_date: str,
    limit: int = 10,
    *,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Rank sold products by units then revenue; rows include product metadata."""
    start, end = _validate_date_range(start_date, end_date)
    bounded_limit = _validate_limit(limit)
    query = """
        SELECT
            p.id AS product_id,
            p.sku,
            p.name,
            p.category,
            SUM(i.quantity) AS units_sold,
            SUM(i.quantity * i.unit_price_pkr) AS revenue_pkr
        FROM products AS p
        JOIN order_items AS i ON i.product_id = p.id
        JOIN orders AS o ON o.id = i.order_id
        WHERE o.status = 'completed'
          AND o.order_date BETWEEN ? AND ?
        GROUP BY p.id, p.sku, p.name, p.category
        ORDER BY units_sold DESC, revenue_pkr DESC, p.name ASC, p.id ASC
        LIMIT ?
    """
    with closing(_connect_readonly(database_path)) as connection:
        return _fetch_dicts(connection, query, (start, end, bounded_limit))


def low_stock_items(
    *,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Return stocked products whose quantity is at or below reorder_level."""
    query = """
        SELECT
            p.id AS product_id,
            p.sku,
            p.name,
            p.category,
            s.quantity,
            s.reorder_level
        FROM products AS p
        JOIN stock AS s ON s.product_id = p.id
        WHERE s.quantity <= s.reorder_level
        ORDER BY s.quantity ASC, p.name ASC, p.id ASC
    """
    with closing(_connect_readonly(database_path)) as connection:
        return _fetch_dicts(connection, query)


def top_customers(
    start_date: str,
    end_date: str,
    limit: int = 10,
    *,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Rank customers by completed-order spending for inclusive dates."""
    start, end = _validate_date_range(start_date, end_date)
    bounded_limit = _validate_limit(limit)
    query = """
        SELECT
            c.id AS customer_id,
            c.name,
            c.city,
            COUNT(o.id) AS order_count,
            SUM(o.total) AS spend_pkr
        FROM customers AS c
        JOIN orders AS o ON o.customer_id = c.id
        WHERE o.status = 'completed'
          AND o.order_date BETWEEN ? AND ?
        GROUP BY c.id, c.name, c.city
        ORDER BY spend_pkr DESC, c.name ASC, c.id ASC
        LIMIT ?
    """
    with closing(_connect_readonly(database_path)) as connection:
        return _fetch_dicts(connection, query, (start, end, bounded_limit))


def customer_order_history(
    customer_id: int,
    limit: int = 20,
    *,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Return recent orders for an exact numeric customer ID, including all statuses."""
    validated_id = _validate_customer_id(customer_id)
    bounded_limit = _validate_limit(limit)
    query = """
        SELECT
            id AS order_id,
            order_date,
            status,
            total AS total_pkr
        FROM orders
        WHERE customer_id = ?
        ORDER BY order_date DESC, id DESC
        LIMIT ?
    """
    with closing(_connect_readonly(database_path)) as connection:
        return _fetch_dicts(connection, query, (validated_id, bounded_limit))
