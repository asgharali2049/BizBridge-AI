from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any

from mcp.shared.memory import create_connected_server_and_client_session

import analytics
import server_stubs


class MCPServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_directory.name) / "mcp-fixture.db"
        connection = sqlite3.connect(self.database_path)
        try:
            connection.executescript(
                (Path(__file__).resolve().parents[1] / "schema.sql").read_text(
                    encoding="utf-8"
                )
            )
            connection.executemany(
                "INSERT INTO customers (id, name, phone, city, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (1, "Amina Ahmed", "0300-0000001", "Sukkur", "2024-01-01 09:00:00"),
                    (2, "Bilal Ali", "0300-0000002", "Rohri", "2024-01-01 09:00:00"),
                ],
            )
            connection.executemany(
                "INSERT INTO products "
                "(id, sku, name, category, selling_price_pkr, purchase_cost_pkr) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (1, "GS-0001", "Rice", "Groceries", 150, 100),
                    (2, "GS-0002", "Tea", "Groceries", 200, 125),
                ],
            )
            connection.executemany(
                "INSERT INTO stock (product_id, quantity, reorder_level) VALUES (?, ?, ?)",
                [(1, 2, 2), (2, 8, 3)],
            )
            connection.executemany(
                "INSERT INTO orders (id, customer_id, order_date, status, total) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (1, 1, "2025-01-01", "completed", 300),
                    (2, 1, "2025-01-02", "cancelled", 999),
                    (3, 2, "2025-01-02", "refunded", 200),
                ],
            )
            connection.executemany(
                "INSERT INTO order_items "
                "(order_id, product_id, quantity, unit_price_pkr) VALUES (?, ?, ?, ?)",
                [(1, 1, 2, 100), (1, 2, 1, 100), (2, 1, 9, 111), (3, 2, 1, 200)],
            )
            connection.commit()
        finally:
            connection.close()
        self.server = server_stubs.create_server(self.database_path)

    async def asyncTearDown(self) -> None:
        self.temp_directory.cleanup()

    async def _call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        async with create_connected_server_and_client_session(self.server) as client:
            return await client.call_tool(name, arguments)

    @staticmethod
    def _structured_content(result: Any) -> Any:
        if result.structuredContent is not None:
            structured = result.structuredContent
            return structured.get("result", structured)
        text = result.content[0].text
        return json.loads(text)

    async def test_server_exposes_exactly_five_typed_tools(self) -> None:
        async with create_connected_server_and_client_session(self.server) as client:
            response = await client.list_tools()

        tools = {tool.name: tool for tool in response.tools}
        self.assertEqual(
            set(tools),
            {
                "sales_summary",
                "top_products",
                "low_stock_items",
                "top_customers",
                "customer_order_history",
            },
        )
        self.assertTrue(all(tool.description for tool in tools.values()))
        schemas = {name: tool.inputSchema for name, tool in tools.items()}
        self.assertEqual(schemas["sales_summary"]["required"], ["start_date", "end_date"])
        self.assertEqual(
            schemas["sales_summary"]["properties"]["start_date"]["type"], "string"
        )
        self.assertEqual(schemas["top_products"]["properties"]["limit"]["type"], "integer")
        self.assertEqual(schemas["top_customers"]["properties"]["limit"]["type"], "integer")
        self.assertEqual(
            schemas["customer_order_history"]["properties"]["customer_id"]["type"],
            "integer",
        )
        self.assertEqual(schemas["low_stock_items"]["properties"], {})

    async def test_valid_tool_calls_return_analytics_results(self) -> None:
        summary = await self._call_tool(
            "sales_summary",
            {"start_date": "2025-01-01", "end_date": "2025-01-02"},
        )
        self.assertFalse(summary.isError)
        self.assertEqual(
            self._structured_content(summary),
            {"order_count": 1, "revenue_pkr": 300, "units_sold": 3},
        )

        products = await self._call_tool(
            "top_products",
            {"start_date": "2025-01-01", "end_date": "2025-01-02", "limit": 1},
        )
        self.assertFalse(products.isError)
        self.assertEqual(self._structured_content(products)[0]["name"], "Rice")

        low_stock = await self._call_tool("low_stock_items")
        self.assertFalse(low_stock.isError)
        self.assertEqual(
            [row["name"] for row in self._structured_content(low_stock)],
            ["Rice"],
        )

        customers = await self._call_tool(
            "top_customers",
            {"start_date": "2025-01-01", "end_date": "2025-01-02"},
        )
        self.assertFalse(customers.isError)
        self.assertEqual(self._structured_content(customers)[0]["customer_id"], 1)

        history = await self._call_tool(
            "customer_order_history",
            {"customer_id": 1},
        )
        self.assertFalse(history.isError)
        self.assertEqual(
            [row["status"] for row in self._structured_content(history)],
            ["cancelled", "completed"],
        )

    async def test_invalid_inputs_return_safe_useful_mcp_errors(self) -> None:
        result = await self._call_tool(
            "sales_summary",
            {"start_date": "not-a-date", "end_date": "2025-01-02"},
        )
        self.assertTrue(result.isError)
        message = "\n".join(item.text for item in result.content if item.type == "text")
        self.assertIn("start_date", message)
        self.assertIn("YYYY-MM-DD", message)
        self.assertNotIn(str(self.database_path), message)

        result = await self._call_tool(
            "customer_order_history",
            {"customer_id": -1},
        )
        self.assertTrue(result.isError)
        message = "\n".join(item.text for item in result.content if item.type == "text")
        self.assertIn("positive integer", message)
        self.assertNotIn(str(self.database_path), message)

    async def test_server_import_and_calls_do_not_write_database(self) -> None:
        before = self.database_path.read_bytes()
        await self._call_tool(
            "sales_summary",
            {"start_date": "2025-01-01", "end_date": "2025-01-02"},
        )
        self.assertEqual(self.database_path.read_bytes(), before)

        with self.assertRaises(sqlite3.OperationalError):
            connection = analytics._connect_readonly(self.database_path)
            try:
                connection.execute("DELETE FROM customers WHERE id = 1")
            finally:
                connection.close()
        self.assertEqual(self.database_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
