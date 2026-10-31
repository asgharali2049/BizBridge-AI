from __future__ import annotations

import sqlite3
import tempfile
import unittest
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import seed


REFERENCE_DATE = date(2025, 12, 31)
TABLE_NAMES = (
    "customers",
    "products",
    "stock",
    "orders",
    "order_items",
    "restock_drafts",
    "restock_draft_items",
)


class SeedDatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.database_path = Path(cls.temp_directory.name) / "test-shop.db"
        cls.record_counts = seed.generate_database(
            cls.database_path,
            seed=418,
            reference_date=REFERENCE_DATE,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_directory.cleanup()

    def setUp(self) -> None:
        self.connection = seed.connect_database(self.database_path)
        self.connection.row_factory = sqlite3.Row

    def tearDown(self) -> None:
        self.connection.close()

    def test_required_tables_and_expected_record_counts(self) -> None:
        table_names = {
            row["name"]
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        self.assertTrue(set(TABLE_NAMES).issubset(table_names))
        self.assertGreaterEqual(self.record_counts["customers"], 100)
        self.assertLessEqual(self.record_counts["customers"], 120)
        self.assertGreaterEqual(self.record_counts["products"], 30)
        self.assertLessEqual(self.record_counts["products"], 40)
        self.assertGreaterEqual(self.record_counts["orders"], 400)
        self.assertEqual(
            self.record_counts["stock"],
            self.record_counts["products"],
        )
        self.assertGreater(self.record_counts["order_items"], self.record_counts["orders"])
        for table in ("customers", "products", "stock", "orders", "order_items"):
            actual = self.connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            expected_key = "order_items" if table == "order_items" else table
            self.assertEqual(actual, self.record_counts[expected_key])

    def test_relationships_and_foreign_keys(self) -> None:
        self.assertEqual(
            self.connection.execute("PRAGMA foreign_keys").fetchone()[0],
            1,
        )
        self.assertEqual(
            self.connection.execute("PRAGMA foreign_key_check").fetchall(),
            [],
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM products p "
                "LEFT JOIN stock s ON s.product_id = p.id "
                "WHERE s.product_id IS NULL"
            ).fetchone()[0],
            0,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM orders o "
                "LEFT JOIN order_items i ON i.order_id = o.id "
                "WHERE i.order_id IS NULL"
            ).fetchone()[0],
            0,
        )

    def test_values_statuses_dates_and_order_totals(self) -> None:
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM products "
                "WHERE selling_price_pkr < 0 OR purchase_cost_pkr < 0"
            ).fetchone()[0],
            0,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM stock "
                "WHERE quantity < 0 OR reorder_level < 0"
            ).fetchone()[0],
            0,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM order_items "
                "WHERE quantity <= 0 OR unit_price_pkr < 0"
            ).fetchone()[0],
            0,
        )
        products_with_price_variation = self.connection.execute(
            "SELECT COUNT(*) FROM ("
            "SELECT product_id FROM order_items "
            "GROUP BY product_id HAVING COUNT(DISTINCT unit_price_pkr) > 1)"
        ).fetchone()[0]
        self.assertGreater(products_with_price_variation, 0)
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM orders "
                "WHERE status NOT IN ('completed', 'cancelled', 'refunded')"
            ).fetchone()[0],
            0,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM orders "
                "WHERE order_date < ? OR order_date > ? OR date(order_date) IS NULL",
                (
                    (REFERENCE_DATE - timedelta(days=89)).isoformat(),
                    REFERENCE_DATE.isoformat(),
                ),
            ).fetchone()[0],
            0,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM orders o "
                "WHERE o.total != ("
                "SELECT COALESCE(SUM(i.quantity * i.unit_price_pkr), 0) "
                "FROM order_items i WHERE i.order_id = o.id)"
            ).fetchone()[0],
            0,
        )
        statuses = {
            row["status"]: row["count"]
            for row in self.connection.execute(
                "SELECT status, COUNT(*) AS count FROM orders GROUP BY status"
            )
        }
        self.assertGreater(statuses.get("cancelled", 0), 0)
        self.assertGreater(statuses.get("refunded", 0), 0)

    def test_schema_rejects_invalid_prices_quantities_and_statuses(self) -> None:
        self.connection.execute("SAVEPOINT constraint_checks")
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                self.connection.execute(
                    "INSERT INTO products "
                    "(id, sku, name, category, selling_price_pkr, purchase_cost_pkr) "
                    "VALUES (999, 'INVALID-PRICE', 'Test item', 'Groceries', -1, 1)"
                )
            with self.assertRaises(sqlite3.IntegrityError):
                self.connection.execute(
                    "INSERT INTO stock (product_id, quantity, reorder_level) "
                    "VALUES (1, -1, 5)"
                )
            with self.assertRaises(sqlite3.IntegrityError):
                self.connection.execute(
                    "INSERT INTO stock (product_id, quantity, reorder_level) "
                    "VALUES (999, 1, 5)"
                )
            with self.assertRaises(sqlite3.IntegrityError):
                self.connection.execute(
                    "INSERT INTO orders (customer_id, order_date, status, total) "
                    "VALUES (1, '2025-12-31', 'pending', 0)"
                )
            with self.assertRaises(sqlite3.IntegrityError):
                self.connection.execute(
                    "INSERT INTO orders (customer_id, order_date, status, total) "
                    "VALUES (1, 'not-a-date', 'completed', 0)"
                )

            unused_pair = self.connection.execute(
                "SELECT o.id, p.id FROM orders o CROSS JOIN products p "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM order_items i "
                "WHERE i.order_id = o.id AND i.product_id = p.id) LIMIT 1"
            ).fetchone()
            with self.assertRaises(sqlite3.IntegrityError):
                self.connection.execute(
                    "INSERT INTO order_items "
                    "(order_id, product_id, quantity, unit_price_pkr) "
                    "VALUES (?, ?, 0, 1)",
                    tuple(unused_pair),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                self.connection.execute(
                    "INSERT INTO restock_drafts (created_at, status) "
                    "VALUES ('2025-12-31 10:00:00', 'pending')"
                )

            draft_id = self.connection.execute(
                "INSERT INTO restock_drafts (created_at, status) "
                "VALUES ('2025-12-31 10:00:00', 'draft')"
            ).lastrowid
            with self.assertRaises(sqlite3.IntegrityError):
                self.connection.execute(
                    "INSERT INTO restock_draft_items (draft_id, product_id, quantity) "
                    "VALUES (?, 1, 0)",
                    (draft_id,),
                )
        finally:
            self.connection.execute("ROLLBACK TO constraint_checks")
            self.connection.execute("RELEASE constraint_checks")

    def test_popular_product_spikes_and_finishes_out_of_stock(self) -> None:
        product_id = self._product_id(seed.POPULAR_PRODUCT_NAME)
        start_date = REFERENCE_DATE - timedelta(days=89)
        spike_start = REFERENCE_DATE - timedelta(days=9)
        prior_start = spike_start - timedelta(days=10)
        prior_end = spike_start - timedelta(days=1)
        spike_units = self._completed_units(product_id, spike_start, REFERENCE_DATE)
        prior_units = self._completed_units(product_id, prior_start, prior_end)
        self.assertGreater(spike_units, prior_units * 2)
        self.assertGreaterEqual(spike_start, start_date)
        stock = self.connection.execute(
            "SELECT quantity FROM stock WHERE product_id = ?", (product_id,)
        ).fetchone()["quantity"]
        self.assertEqual(stock, 0)

    def test_cold_beverage_demand_rises_and_seasonal_product_declines(self) -> None:
        first_period_start = REFERENCE_DATE - timedelta(days=89)
        first_period_end = first_period_start + timedelta(days=29)
        final_period_start = REFERENCE_DATE - timedelta(days=29)
        cold_names = ",".join("?" for _ in seed.COLD_BEVERAGE_NAMES)
        cold_first = self.connection.execute(
            f"SELECT COALESCE(SUM(i.quantity), 0) FROM order_items i "
            f"JOIN orders o ON o.id = i.order_id "
            f"JOIN products p ON p.id = i.product_id "
            f"WHERE o.status = 'completed' AND p.name IN ({cold_names}) "
            "AND o.order_date BETWEEN ? AND ?",
            (*seed.COLD_BEVERAGE_NAMES, first_period_start.isoformat(), first_period_end.isoformat()),
        ).fetchone()[0]
        cold_final = self.connection.execute(
            f"SELECT COALESCE(SUM(i.quantity), 0) FROM order_items i "
            f"JOIN orders o ON o.id = i.order_id "
            f"JOIN products p ON p.id = i.product_id "
            f"WHERE o.status = 'completed' AND p.name IN ({cold_names}) "
            "AND o.order_date BETWEEN ? AND ?",
            (*seed.COLD_BEVERAGE_NAMES, final_period_start.isoformat(), REFERENCE_DATE.isoformat()),
        ).fetchone()[0]
        self.assertGreater(cold_final, cold_first)

        declining_id = self._product_id(seed.DECLINING_PRODUCT_NAME)
        declining_first = self._completed_units(
            declining_id, first_period_start, first_period_end
        )
        declining_final = self._completed_units(
            declining_id, final_period_start, REFERENCE_DATE
        )
        self.assertGreater(declining_first, declining_final)

    def test_wholesale_customer_leads_overall_but_not_every_date_range(self) -> None:
        top_customer = self.connection.execute(
            "SELECT c.name, SUM(o.total) AS spend FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "WHERE o.status = 'completed' GROUP BY c.id "
            "ORDER BY spend DESC LIMIT 1"
        ).fetchone()
        self.assertEqual(top_customer["name"], "Sukkur Wholesale Traders")

        opening_day = (REFERENCE_DATE - timedelta(days=89)).isoformat()
        opening_day_top = self.connection.execute(
            "SELECT c.name, SUM(o.total) AS spend FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "WHERE o.status = 'completed' AND o.order_date = ? "
            "GROUP BY c.id ORDER BY spend DESC LIMIT 1",
            (opening_day,),
        ).fetchone()
        self.assertEqual(opening_day_top["name"], "Amina Ahmed")
        self.assertNotEqual(opening_day_top["name"], top_customer["name"])

    def test_customer_repeats_the_same_basket_every_seven_days(self) -> None:
        customer_id = self.connection.execute(
            "SELECT id FROM customers WHERE name = 'Amina Ahmed'"
        ).fetchone()["id"]
        orders = self.connection.execute(
            "SELECT id, order_date FROM orders "
            "WHERE customer_id = ? AND status = 'completed' ORDER BY order_date",
            (customer_id,),
        ).fetchall()
        self.assertEqual(len(orders), 13)
        order_dates = [date.fromisoformat(row["order_date"]) for row in orders]
        self.assertTrue(
            all((later - earlier).days == 7 for earlier, later in zip(order_dates, order_dates[1:]))
        )

        baskets = []
        for order in orders:
            baskets.append(
                tuple(
                    self.connection.execute(
                        "SELECT p.name, i.quantity FROM order_items i "
                        "JOIN products p ON p.id = i.product_id "
                        "WHERE i.order_id = ? ORDER BY p.name",
                        (order["id"],),
                    ).fetchall()
                )
            )
        normalized_baskets = [
            tuple((row["name"], row["quantity"]) for row in basket)
            for basket in baskets
        ]
        self.assertTrue(all(basket == normalized_baskets[0] for basket in normalized_baskets))
        self.assertEqual(
            normalized_baskets[0],
            tuple(sorted(seed.WEEKLY_BASKET)),
        )

    def test_fridays_and_saturdays_have_higher_order_volume(self) -> None:
        daily_counts = self.connection.execute(
            "SELECT order_date, COUNT(*) AS count FROM orders "
            "GROUP BY order_date ORDER BY order_date"
        ).fetchall()
        weekend_counts = []
        weekday_counts = []
        for row in daily_counts:
            if date.fromisoformat(row["order_date"]).weekday() in (4, 5):
                weekend_counts.append(row["count"])
            else:
                weekday_counts.append(row["count"])
        self.assertEqual(len(daily_counts), seed.DAY_COUNT)
        self.assertGreater(sum(weekend_counts) / len(weekend_counts),
                           sum(weekday_counts) / len(weekday_counts))

    def test_cancelled_and_refunded_orders_are_excluded_from_completed_sales(self) -> None:
        completed_total = self.connection.execute(
            "SELECT SUM(total) FROM orders WHERE status = 'completed'"
        ).fetchone()[0]
        all_orders_total = self.connection.execute(
            "SELECT SUM(total) FROM orders"
        ).fetchone()[0]
        self.assertLess(completed_total, all_orders_total)
        cancelled_or_refunded_total = self.connection.execute(
            "SELECT SUM(total) FROM orders "
            "WHERE status IN ('cancelled', 'refunded')"
        ).fetchone()[0]
        self.assertGreater(cancelled_or_refunded_total, 0)

    def test_multiple_customers_share_first_names(self) -> None:
        first_names = Counter(
            row["name"].split()[0]
            for row in self.connection.execute("SELECT name FROM customers")
        )
        self.assertGreaterEqual(sum(count > 1 for count in first_names.values()), 5)

    def test_generator_refuses_to_overwrite_existing_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            existing_path = Path(directory) / "existing.db"
            original_contents = b"preserve this existing file"
            existing_path.write_bytes(original_contents)
            with self.assertRaisesRegex(FileExistsError, "will not be overwritten"):
                seed.generate_database(
                    existing_path,
                    seed=418,
                    reference_date=REFERENCE_DATE,
                )
            self.assertEqual(existing_path.read_bytes(), original_contents)

    def test_fixed_seed_and_reference_date_produce_equivalent_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            second_path = Path(directory) / "second.db"
            seed.generate_database(second_path, seed=418, reference_date=REFERENCE_DATE)
            second_connection = seed.connect_database(second_path)
            try:
                for table in TABLE_NAMES:
                    columns = [
                        row["name"]
                        for row in self.connection.execute(f"PRAGMA table_info({table})")
                    ]
                    order_by = ", ".join(columns)
                    first_rows = self.connection.execute(
                        f"SELECT * FROM {table} ORDER BY {order_by}"
                    ).fetchall()
                    second_rows = second_connection.execute(
                        f"SELECT * FROM {table} ORDER BY {order_by}"
                    ).fetchall()
                    self.assertEqual(
                        [tuple(row) for row in first_rows],
                        [tuple(row) for row in second_rows],
                        table,
                    )
            finally:
                second_connection.close()

    def _product_id(self, product_name: str) -> int:
        row = self.connection.execute(
            "SELECT id FROM products WHERE name = ?", (product_name,)
        ).fetchone()
        self.assertIsNotNone(row)
        return row["id"]

    def _completed_units(self, product_id: int, start: date, end: date) -> int:
        return self.connection.execute(
            "SELECT COALESCE(SUM(i.quantity), 0) FROM order_items i "
            "JOIN orders o ON o.id = i.order_id "
            "WHERE i.product_id = ? AND o.status = 'completed' "
            "AND o.order_date BETWEEN ? AND ?",
            (product_id, start.isoformat(), end.isoformat()),
        ).fetchone()[0]


if __name__ == "__main__":
    unittest.main()
