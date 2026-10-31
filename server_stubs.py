"""BizBridge AI's read-only MCP analytics server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

import analytics


DATABASE_PATH = Path(__file__).resolve().with_name("shop.db")


def create_server(database_path: str | Path = DATABASE_PATH) -> FastMCP:
    """Create the analytics MCP server, optionally targeting an isolated database."""
    database = Path(database_path).expanduser().resolve()
    server = FastMCP(
        name="BizBridge AI Analytics",
        instructions=(
            "Read-only business analytics for a fictional general retail shop. "
            "All sales and spending metrics include completed orders only; "
            "order history includes every status."
        ),
    )

    @server.tool(
        name="sales_summary",
        description=(
            "Summarize completed sales for an inclusive date range. "
            "Provide start_date and end_date as valid ISO YYYY-MM-DD strings "
            "with start_date on or before end_date. Returns completed order "
            "count, revenue in PKR, and order-item units; cancelled and refunded "
            "orders are excluded."
        ),
    )
    def sales_summary(start_date: str, end_date: str) -> dict[str, int]:
        return analytics.sales_summary(
            start_date,
            end_date,
            database_path=database,
        )

    @server.tool(
        name="top_products",
        description=(
            "Rank products by units sold and then revenue for an inclusive "
            "YYYY-MM-DD date range. Only completed orders count. limit defaults "
            "to 10 and values above 50 are capped at 50."
        ),
    )
    def top_products(
        start_date: str,
        end_date: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return analytics.top_products(
            start_date,
            end_date,
            limit=limit,
            database_path=database,
        )

    @server.tool(
        name="low_stock_items",
        description=(
            "List products whose current stock quantity is at or below their "
            "reorder level, including product details and both quantities."
        ),
    )
    def low_stock_items() -> list[dict[str, Any]]:
        return analytics.low_stock_items(database_path=database)

    @server.tool(
        name="top_customers",
        description=(
            "Rank customers by spending on completed orders during an inclusive "
            "YYYY-MM-DD date range. limit defaults to 10 and values above 50 "
            "are capped at 50."
        ),
    )
    def top_customers(
        start_date: str,
        end_date: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return analytics.top_customers(
            start_date,
            end_date,
            limit=limit,
            database_path=database,
        )

    @server.tool(
        name="customer_order_history",
        description=(
            "Show recent orders for one exact positive numeric customer_id. "
            "Returns order date, status, and total in PKR; unlike sales "
            "rankings, history includes completed, cancelled, and refunded "
            "orders. limit defaults to 20 and values above 50 are capped at 50."
        ),
    )
    def customer_order_history(
        customer_id: int,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return analytics.customer_order_history(
            customer_id,
            limit=limit,
            database_path=database,
        )

    return server


def main() -> None:
    """Run the local MCP server over stdio."""
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
