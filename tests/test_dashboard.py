from __future__ import annotations

import unittest
from unittest.mock import patch

from ui.dashboard import (
    _aggregate_trend,
    _trend_granularity,
    prepare_dashboard_data,
    validate_date_range,
)


AVAILABLE_RANGE = ("2025-01-01", "2025-03-31")


class DashboardDateRangeTests(unittest.TestCase):
    def test_validates_iso_dates_order_and_available_bounds(self) -> None:
        self.assertEqual(
            validate_date_range("2025-01-02", "2025-03-30", AVAILABLE_RANGE),
            ("2025-01-02", "2025-03-30"),
        )
        for start_date, end_date in (
            ("2025-1-02", "2025-03-30"),
            ("2025-02-30", "2025-03-30"),
            ("2025-02-01", "2025-01-01"),
            ("2024-12-31", "2025-01-01"),
            ("2025-01-01", "2025-04-01"),
        ):
            with self.subTest(start_date=start_date, end_date=end_date):
                with self.assertRaises(ValueError):
                    validate_date_range(start_date, end_date, AVAILABLE_RANGE)
        for invalid_range in (
            ("2025-02-30", "2025-03-31"),
            ("2025-03-31", "2025-01-01"),
        ):
            with self.subTest(invalid_range=invalid_range):
                with self.assertRaisesRegex(ValueError, "Available"):
                    validate_date_range("2025-01-02", "2025-03-30", invalid_range)


class DashboardDataPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = patch(
            "ui.dashboard.analytics.order_date_bounds",
            return_value=AVAILABLE_RANGE,
        )
        self.summary = patch(
            "ui.dashboard.analytics.sales_summary",
            return_value={
                "order_count": 2,
                "revenue_pkr": 900,
                "units_sold": 5,
            },
        )
        self.daily = patch(
            "ui.dashboard.analytics.daily_sales_summary",
            return_value=[
                {
                    "order_date": "2025-01-01",
                    "order_count": 1,
                    "revenue_pkr": 400,
                    "units_sold": 2,
                },
                {
                    "order_date": "2025-01-03",
                    "order_count": 1,
                    "revenue_pkr": 500,
                    "units_sold": 3,
                },
            ],
        )
        self.products = patch(
            "ui.dashboard.analytics.top_products",
            return_value=[
                {"name": "Rice", "units_sold": 3, "revenue_pkr": 450}
            ],
        )
        self.inventory = patch(
            "ui.dashboard.analytics.low_stock_items",
            return_value=[
                {
                    "name": "Tea",
                    "category": "Groceries",
                    "quantity": 0,
                    "reorder_level": 4,
                }
            ],
        )
        self.customers = patch(
            "ui.dashboard.analytics.top_customers",
            return_value=[
                {"name": "Amina", "order_count": 1, "spend_pkr": 500}
            ],
        )
        self.mocks = [
            self.bounds,
            self.summary,
            self.daily,
            self.products,
            self.inventory,
            self.customers,
        ]
        self.entered = [mock.start() for mock in self.mocks]
        self.addCleanup(self._stop_mocks)

    def _stop_mocks(self) -> None:
        for mock in reversed(self.mocks):
            mock.stop()

    def test_uses_analytics_outputs_for_kpis_trends_rankings_and_inventory(self) -> None:
        data = prepare_dashboard_data("2025-01-01", "2025-02-01", top_limit=5)

        self.assertEqual(data["summary"]["order_count"], 2)
        self.assertEqual(data["average_order_value_pkr"], 450)
        self.assertEqual(data["granularity"], "week")
        self.assertEqual(len(data["trend"]), 1)
        self.assertEqual(data["trend"][0]["revenue_pkr"], 900)
        self.assertEqual(data["top_products"][0]["name"], "Rice")
        self.assertEqual(data["low_stock_items"][0]["quantity"], 0)
        self.assertEqual(data["top_customers"][0]["spend_pkr"], 500)
        self.entered[1].assert_called_once_with("2025-01-01", "2025-02-01")
        self.entered[2].assert_called_once_with("2025-01-01", "2025-02-01")
        self.entered[3].assert_called_once_with(
            "2025-01-01", "2025-02-01", limit=5
        )
        self.entered[4].assert_called_once_with()
        self.entered[5].assert_called_once_with(
            "2025-01-01", "2025-02-01", limit=5
        )

    def test_empty_database_returns_empty_dashboard_without_other_queries(self) -> None:
        self.entered[0].return_value = None

        data = prepare_dashboard_data("2025-01-01", "2025-03-31")

        self.assertIsNone(data["available_range"])
        self.assertIsNone(data["average_order_value_pkr"])
        self.assertEqual(data["summary"]["order_count"], 0)
        self.assertEqual(data["trend"], [])
        for mock in self.entered[1:]:
            mock.assert_not_called()

    def test_no_sales_has_no_average_order_value(self) -> None:
        self.entered[1].return_value = {
            "order_count": 0,
            "revenue_pkr": 0,
            "units_sold": 0,
        }
        self.entered[2].return_value = []
        self.entered[3].return_value = []
        self.entered[5].return_value = []

        data = prepare_dashboard_data("2025-01-01", "2025-01-02")

        self.assertIsNone(data["average_order_value_pkr"])
        self.assertEqual(data["trend"], [])
        self.assertEqual(data["top_products"], [])
        self.assertEqual(data["top_customers"], [])

    def test_rejects_invalid_date_range_before_loading_metrics(self) -> None:
        with self.assertRaises(ValueError):
            prepare_dashboard_data("2025-02-01", "2025-01-01")
        self.entered[1].assert_not_called()

    def test_rejects_missing_or_invalid_analytics_values(self) -> None:
        self.entered[1].return_value = {"order_count": 2, "revenue_pkr": 900}
        with self.assertRaisesRegex(ValueError, "incomplete"):
            prepare_dashboard_data("2025-01-01", "2025-01-31")

        self.entered[1].return_value = {
            "order_count": -1,
            "revenue_pkr": 900,
            "units_sold": 5,
        }
        with self.assertRaisesRegex(ValueError, "negative"):
            prepare_dashboard_data("2025-01-01", "2025-01-31")

    def test_rejects_invalid_product_rows(self) -> None:
        self.entered[3].return_value = [{"name": "Rice", "units_sold": "many"}]
        with self.assertRaisesRegex(ValueError, "incomplete|valid"):
            prepare_dashboard_data("2025-01-01", "2025-01-31")


class DashboardTrendTests(unittest.TestCase):
    def test_selects_granularity_from_range_length(self) -> None:
        self.assertEqual(_trend_granularity("2025-01-01", "2025-01-31"), "day")
        self.assertEqual(_trend_granularity("2025-01-01", "2025-03-31"), "week")
        self.assertEqual(_trend_granularity("2024-01-01", "2025-01-01"), "month")

    def test_aggregates_observed_dates_without_filling_gaps(self) -> None:
        rows = [
            {
                "order_date": "2025-01-06",
                "order_count": 2,
                "revenue_pkr": 300,
                "units_sold": 3,
            },
            {
                "order_date": "2025-01-08",
                "order_count": 1,
                "revenue_pkr": 100,
                "units_sold": 1,
            },
        ]

        trend = _aggregate_trend(rows, "day")

        self.assertEqual([row["period"] for row in trend], ["2025-01-06", "2025-01-08"])
        self.assertEqual(_aggregate_trend(rows, "week"), [
            {
                "period": "2025-01-06",
                "order_count": 3,
                "revenue_pkr": 400,
                "units_sold": 4,
            }
        ])

    def test_rejects_incomplete_or_invalid_trend_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete"):
            _aggregate_trend([{"order_date": "2025-01-01"}], "day")
        with self.assertRaisesRegex(ValueError, "invalid date"):
            _aggregate_trend(
                [{
                    "order_date": "2025-02-30",
                    "order_count": 1,
                    "revenue_pkr": 100,
                    "units_sold": 1,
                }],
                "day",
            )


if __name__ == "__main__":
    unittest.main()
