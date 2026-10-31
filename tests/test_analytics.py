from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import analytics


class AnalyticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.database_path = Path(cls.temp_directory.name) / "analytics fixture.db"
        connection = sqlite3.connect(cls.database_path)
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
                    (2, "Amina Khan", "0300-0000002", "Rohri", "2024-01-01 09:00:00"),
                    (3, "Bilal Ali", "0300-0000003", "Khairpur", "2024-01-01 09:00:00"),
                ],
            )
            connection.executemany(
                "INSERT INTO products "
                "(id, sku, name, category, selling_price_pkr, purchase_cost_pkr) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (1, "GS-0001", "Rice", "Groceries", 150, 100),
                    (2, "GS-0002", "Tea", "Groceries", 200, 125),
                    (3, "GS-0003", "Soap", "Personal care", 300, 180),
                ],
            )
            connection.executemany(
                "INSERT INTO stock (product_id, quantity, reorder_level) VALUES (?, ?, ?)",
                [(1, 3, 3), (2, 0, 5), (3, 8, 2)],
            )
            connection.executemany(
                "INSERT INTO orders (id, customer_id, order_date, status, total) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (1, 1, "2025-01-01", "completed", 300),
                    (2, 2, "2025-01-31", "completed", 500),
                    (3, 1, "2025-01-31", "cancelled", 1000),
                    (4, 3, "2025-02-01", "refunded", 800),
                    (5, 2, "2025-02-01", "completed", 300),
                ],
            )
            connection.executemany(
                "INSERT INTO order_items "
                "(order_id, product_id, quantity, unit_price_pkr) "
                "VALUES (?, ?, ?, ?)",
                [
                    (1, 1, 2, 100),
                    (1, 2, 1, 100),
                    (2, 1, 1, 150),
                    (2, 3, 2, 175),
                    (3, 1, 10, 100),
                    (4, 2, 4, 200),
                    (5, 3, 1, 300),
                ],
            )
            connection.commit()
        finally:
            connection.close()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_directory.cleanup()

    def test_sales_summary_counts_completed_orders_and_inclusive_boundaries(self) -> None:
        self.assertEqual(
            analytics.sales_summary(
                "2025-01-01", "2025-01-31", database_path=self.database_path
            ),
            {"order_count": 2, "revenue_pkr": 800, "units_sold": 6},
        )

    def test_sales_summary_empty_range_returns_zero_metrics(self) -> None:
        self.assertEqual(
            analytics.sales_summary(
                "2025-03-01", "2025-03-31", database_path=self.database_path
            ),
            {"order_count": 0, "revenue_pkr": 0, "units_sold": 0},
        )

    def test_order_date_bounds_and_daily_completed_sales(self) -> None:
        self.assertEqual(
            analytics.order_date_bounds(database_path=self.database_path),
            ("2025-01-01", "2025-02-01"),
        )
        daily_result = analytics.daily_sales_summary(
            "2025-01-01", "2025-01-31", database_path=self.database_path
        )
        self.assertEqual(
            daily_result,
            [
                {
                    "order_date": "2025-01-01",
                    "order_count": 1,
                    "revenue_pkr": 300,
                    "units_sold": 3,
                },
                {
                    "order_date": "2025-01-31",
                    "order_count": 1,
                    "revenue_pkr": 500,
                    "units_sold": 3,
                },
            ],
        )
        self.assertEqual(
            sum(row["order_count"] for row in daily_result),
            analytics.sales_summary(
                "2025-01-01", "2025-01-31", database_path=self.database_path
            )["order_count"],
        )
        self.assertEqual(
            sum(row["revenue_pkr"] for row in daily_result),
            analytics.sales_summary(
                "2025-01-01", "2025-01-31", database_path=self.database_path
            )["revenue_pkr"],
        )
        self.assertEqual(
            sum(row["units_sold"] for row in daily_result),
            analytics.sales_summary(
                "2025-01-01", "2025-01-31", database_path=self.database_path
            )["units_sold"],
        )
        self.assertEqual(
            analytics.daily_sales_summary(
                "2025-03-01", "2025-03-31", database_path=self.database_path
            ),
            [],
        )

    def test_top_products_rank_by_completed_units_then_revenue(self) -> None:
        result = analytics.top_products(
            "2025-01-01", "2025-01-31", database_path=self.database_path
        )
        self.assertEqual(
            result,
            [
                {
                    "product_id": 1,
                    "sku": "GS-0001",
                    "name": "Rice",
                    "category": "Groceries",
                    "units_sold": 3,
                    "revenue_pkr": 350,
                },
                {
                    "product_id": 3,
                    "sku": "GS-0003",
                    "name": "Soap",
                    "category": "Personal care",
                    "units_sold": 2,
                    "revenue_pkr": 350,
                },
                {
                    "product_id": 2,
                    "sku": "GS-0002",
                    "name": "Tea",
                    "category": "Groceries",
                    "units_sold": 1,
                    "revenue_pkr": 100,
                },
            ],
        )
        self.assertEqual(
            analytics.top_products(
                "2025-01-01", "2025-01-31", limit=1, database_path=self.database_path
            ),
            result[:1],
        )
        self.assertEqual(
            analytics.top_products(
                "2025-03-01", "2025-03-31", database_path=self.database_path
            ),
            [],
        )

    def test_low_stock_items_include_boundary_quantity(self) -> None:
        self.assertEqual(
            analytics.low_stock_items(database_path=self.database_path),
            [
                {
                    "product_id": 2,
                    "sku": "GS-0002",
                    "name": "Tea",
                    "category": "Groceries",
                    "quantity": 0,
                    "reorder_level": 5,
                },
                {
                    "product_id": 1,
                    "sku": "GS-0001",
                    "name": "Rice",
                    "category": "Groceries",
                    "quantity": 3,
                    "reorder_level": 3,
                },
            ],
        )

    def test_top_customers_rank_completed_order_spending(self) -> None:
        self.assertEqual(
            analytics.top_customers(
                "2025-01-01", "2025-01-31", database_path=self.database_path
            ),
            [
                {
                    "customer_id": 2,
                    "name": "Amina Khan",
                    "city": "Rohri",
                    "order_count": 1,
                    "spend_pkr": 500,
                },
                {
                    "customer_id": 1,
                    "name": "Amina Ahmed",
                    "city": "Sukkur",
                    "order_count": 1,
                    "spend_pkr": 300,
                },
            ],
        )
        self.assertEqual(
            analytics.top_customers(
                "2025-03-01", "2025-03-31", database_path=self.database_path
            ),
            [],
        )

    def test_customer_history_uses_exact_id_and_includes_all_statuses(self) -> None:
        self.assertEqual(
            analytics.customer_order_history(
                1, database_path=self.database_path
            ),
            [
                {
                    "order_id": 3,
                    "order_date": "2025-01-31",
                    "status": "cancelled",
                    "total_pkr": 1000,
                },
                {
                    "order_id": 1,
                    "order_date": "2025-01-01",
                    "status": "completed",
                    "total_pkr": 300,
                },
            ],
        )
        self.assertEqual(
            analytics.customer_order_history(
                999, database_path=self.database_path
            ),
            [],
        )
        self.assertEqual(
            analytics.customer_order_history(
                1, limit=1, database_path=self.database_path
            ),
            [
                {
                    "order_id": 3,
                    "order_date": "2025-01-31",
                    "status": "cancelled",
                    "total_pkr": 1000,
                }
            ],
        )

    def test_invalid_dates_and_reversed_ranges_are_rejected(self) -> None:
        for invalid_date in ("2025-1-01", "2025-02-30", "2025-01-01' OR 1=1 --"):
            with self.subTest(invalid_date=invalid_date):
                with self.assertRaises(ValueError):
                    analytics.sales_summary(
                        invalid_date, "2025-01-31", database_path=self.database_path
                    )
        with self.assertRaises(TypeError):
            analytics.sales_summary(
                20250101, "2025-01-31", database_path=self.database_path  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            analytics.top_products(
                "2025-02-01", "2025-01-01", database_path=self.database_path
            )

    def test_customer_ids_and_limits_are_validated_and_limits_are_capped(self) -> None:
        for invalid_id in (0, -1):
            with self.subTest(invalid_id=invalid_id):
                with self.assertRaises(ValueError):
                    analytics.customer_order_history(
                        invalid_id, database_path=self.database_path
                    )
        for invalid_id in ("1 OR 1=1", True):
            with self.subTest(invalid_id=invalid_id):
                with self.assertRaises(TypeError):
                    analytics.customer_order_history(
                        invalid_id, database_path=self.database_path  # type: ignore[arg-type]
                    )

        for invalid_limit in (0, -1):
            with self.subTest(invalid_limit=invalid_limit):
                with self.assertRaises(ValueError):
                    analytics.top_products(
                        "2025-01-01",
                        "2025-01-31",
                        limit=invalid_limit,
                        database_path=self.database_path,
                    )
        for invalid_limit in (1.5, True):
            with self.subTest(invalid_limit=invalid_limit):
                with self.assertRaises(TypeError):
                    analytics.top_customers(
                        "2025-01-01",
                        "2025-01-31",
                        limit=invalid_limit,  # type: ignore[arg-type]
                        database_path=self.database_path,
                    )
        capped_result = analytics.top_products(
            "2025-01-01",
            "2025-01-31",
            limit=100,
            database_path=self.database_path,
        )
        self.assertLessEqual(len(capped_result), analytics.MAX_LIMIT)

    def test_read_only_connections_enable_foreign_keys_and_reject_writes(self) -> None:
        before = self.database_path.read_bytes()
        connection = analytics._connect_readonly(self.database_path)
        try:
            self.assertEqual(
                connection.execute("PRAGMA foreign_keys").fetchone()[0],
                1,
            )
            with self.assertRaisesRegex(sqlite3.OperationalError, "readonly"):
                connection.execute("DELETE FROM customers WHERE id = 1")
        finally:
            connection.close()

        analytics.sales_summary(
            "2025-01-01", "2025-01-31", database_path=self.database_path
        )
        analytics.top_products(
            "2025-01-01", "2025-01-31", database_path=self.database_path
        )
        analytics.low_stock_items(database_path=self.database_path)
        analytics.top_customers(
            "2025-01-01", "2025-01-31", database_path=self.database_path
        )
        analytics.customer_order_history(1, database_path=self.database_path)
        self.assertEqual(self.database_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
