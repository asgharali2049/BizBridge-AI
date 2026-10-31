"""Create a deterministic synthetic general-store SQLite database."""

from __future__ import annotations

import argparse
import random
import sqlite3
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Sequence


PROJECT_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = PROJECT_DIR / "schema.sql"
DEFAULT_DATABASE_PATH = PROJECT_DIR / "shop.db"
DEFAULT_SEED = 20261025
DAY_COUNT = 90
CUSTOMER_COUNT = 110

FIRST_NAMES = (
    "Ayesha", "Bilal", "Danish", "Farah", "Hassan", "Iqra", "Kamran",
    "Maryam", "Naveed", "Rabia", "Saad", "Sana", "Umar", "Zainab",
    "Adnan", "Hina", "Imran", "Mehwish", "Omar", "Sadia", "Tariq",
    "Yasmin",
)
LAST_NAMES = (
    "Ahmed", "Ali", "Ansari", "Baloch", "Chandio", "Hussain", "Jatoi",
    "Khan", "Memon", "Qureshi", "Rind", "Shaikh",
)
CITIES = (
    "Sukkur", "Rohri", "Khairpur", "Shikarpur", "Ghotki", "Larkana",
    "Jacobabad", "Pano Aqil", "Kandhkot",
)

# Names and prices are fictional and all prices represent whole PKR.
PRODUCTS = (
    ("Groceries", "Basmati Rice 5kg", 1450, 1190),
    ("Groceries", "Wheat Flour 5kg", 780, 620),
    ("Groceries", "Cooking Oil 1L", 590, 475),
    ("Groceries", "White Sugar 1kg", 175, 145),
    ("Groceries", "Red Lentils 1kg", 430, 350),
    ("Groceries", "Black Tea 250g", 860, 710),
    ("Groceries", "Table Salt 800g", 95, 68),
    ("Groceries", "Milk Powder 400g", 690, 570),
    ("Beverages", "Cola Bottle 1.5L", 190, 145),
    ("Beverages", "Orange Soda Bottle 1.5L", 185, 140),
    ("Beverages", "Chilled Mango Drink 1L", 220, 168),
    ("Beverages", "Bottled Water 1.5L", 90, 55),
    ("Beverages", "Lemon Drink 1L", 170, 128),
    ("Beverages", "Instant Coffee 100g", 740, 590),
    ("Snacks", "Family Biscuit Multipack", 260, 190),
    ("Snacks", "Salted Potato Chips", 110, 72),
    ("Snacks", "Roasted Peanuts 200g", 210, 155),
    ("Snacks", "Chocolate Wafer Pack", 130, 85),
    ("Snacks", "Spicy Snack Mix 250g", 180, 125),
    ("Snacks", "Plain Crackers", 150, 105),
    ("Snacks", "Fruit Candy Bag", 95, 60),
    ("Personal care", "Bath Soap 3 Pack", 330, 250),
    ("Personal care", "Shampoo 350ml", 580, 440),
    ("Personal care", "Toothpaste 150g", 290, 215),
    ("Personal care", "Toothbrush 2 Pack", 180, 125),
    ("Personal care", "Laundry Detergent 1kg", 420, 325),
    ("Personal care", "Hand Wash 250ml", 260, 190),
    ("Household supplies", "Mosquito Coil Pack", 160, 105),
    ("Household supplies", "Dishwashing Liquid 500ml", 310, 230),
    ("Household supplies", "Kitchen Tissue Roll 2 Pack", 240, 175),
    ("Household supplies", "Garbage Bags 20 Pack", 190, 135),
    ("Household supplies", "Floor Cleaner 1L", 390, 295),
    ("Household supplies", "LED Bulb 9W", 280, 195),
    ("Household supplies", "Plastic Food Container", 350, 245),
    ("Stationery", "A4 Notebook", 140, 90),
    ("Stationery", "Ballpoint Pens 5 Pack", 160, 105),
    ("Stationery", "Pencil Set 10 Pack", 120, 75),
    ("Stationery", "Permanent Marker", 95, 58),
    ("Stationery", "A4 Paper 100 Sheets", 260, 185),
    ("Stationery", "Glue Stick", 85, 50),
)

POPULAR_PRODUCT_NAME = "Family Biscuit Multipack"
DECLINING_PRODUCT_NAME = "Mosquito Coil Pack"
COLD_BEVERAGE_NAMES = (
    "Cola Bottle 1.5L",
    "Orange Soda Bottle 1.5L",
    "Chilled Mango Drink 1L",
    "Lemon Drink 1L",
)
WEEKLY_BASKET = (
    ("Basmati Rice 5kg", 1),
    ("Wheat Flour 5kg", 1),
    ("Cooking Oil 1L", 2),
)
WHOLESALE_BASKET = (
    ("Basmati Rice 5kg", 20),
    ("Cooking Oil 1L", 15),
    ("Red Lentils 1kg", 25),
    ("Black Tea 250g", 12),
    ("White Sugar 1kg", 18),
    ("Laundry Detergent 1kg", 20),
)


def connect_database(database_path: str | Path) -> sqlite3.Connection:
    """Open a database connection with foreign-key enforcement enabled."""
    connection = sqlite3.connect(str(database_path))
    connection.execute("PRAGMA foreign_keys = ON")
    if connection.execute("PRAGMA foreign_keys").fetchone() != (1,):
        connection.close()
        raise RuntimeError("SQLite foreign-key enforcement could not be enabled")
    return connection


def _customer_rows(rng: random.Random, start_date: date) -> list[tuple[object, ...]]:
    names = [("Amina", "Ahmed"), ("Sukkur Wholesale Traders", "")]
    for first_name in FIRST_NAMES:
        for last_name in LAST_NAMES:
            candidate = (first_name, last_name)
            if candidate not in names:
                names.append(candidate)
            if len(names) == CUSTOMER_COUNT:
                break
        if len(names) == CUSTOMER_COUNT:
            break

    rows = []
    for customer_id, (first_name, last_name) in enumerate(names, start=1):
        name = f"{first_name} {last_name}".strip()
        created_date = start_date - timedelta(days=rng.randint(0, 90))
        created_at = datetime.combine(
            created_date, time(hour=rng.randint(8, 19), minute=rng.choice((0, 15, 30, 45)))
        ).isoformat(sep=" ")
        rows.append(
            (
                customer_id,
                name,
                f"0300-{customer_id:07d}",
                rng.choice(CITIES),
                created_at,
            )
        )
    return rows


def _product_rows() -> list[tuple[object, ...]]:
    return [
        (
            product_id,
            f"GS-{product_id:04d}",
            name,
            category,
            selling_price,
            purchase_cost,
        )
        for product_id, (category, name, selling_price, purchase_cost) in enumerate(
            PRODUCTS, start=1
        )
    ]


def _insert_order(
    order_rows: list[tuple[object, ...]],
    item_rows: list[tuple[object, ...]],
    completed_units: Counter[int],
    customer_id: int,
    order_date: date,
    status: str,
    product_quantities: Sequence[tuple[int, int]],
    product_prices: dict[int, int],
) -> None:
    order_id = len(order_rows) + 1
    line_rows = []
    total = 0
    merged_quantities: dict[int, int] = {}
    for product_id, quantity in product_quantities:
        merged_quantities[product_id] = merged_quantities.get(product_id, 0) + quantity
    for product_id, quantity in merged_quantities.items():
        unit_price = product_prices[product_id]
        total += unit_price * quantity
        line_rows.append((product_id, quantity, unit_price))
        if status == "completed":
            completed_units[product_id] += quantity

    order_rows.append((order_id, customer_id, order_date.isoformat(), status, total))
    item_rows.extend(
        (order_id, product_id, quantity, unit_price)
        for product_id, quantity, unit_price in line_rows
    )


def _random_order_lines(
    rng: random.Random,
    product_ids: Sequence[int],
) -> list[tuple[int, int]]:
    selected = rng.sample(list(product_ids), k=rng.randint(2, 4))
    return [(product_id, rng.randint(1, 3)) for product_id in selected]


def generate_database(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    seed: int = DEFAULT_SEED,
    reference_date: date | str | None = None,
) -> dict[str, int]:
    """Create a new database without replacing an existing path."""
    path = Path(database_path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(
            f"Database already exists and will not be overwritten: {path}"
        )
    if reference_date is None:
        reference = date.today()
    elif isinstance(reference_date, str):
        try:
            reference = date.fromisoformat(reference_date)
        except ValueError as error:
            raise ValueError("reference_date must be a valid YYYY-MM-DD date") from error
    else:
        reference = reference_date
    if not isinstance(reference, date):
        raise TypeError("reference_date must be a date, YYYY-MM-DD string, or None")
    start_date = reference - timedelta(days=DAY_COUNT - 1)
    dates = [start_date + timedelta(days=offset) for offset in range(DAY_COUNT)]

    rng = random.Random(seed)
    customers = _customer_rows(rng, start_date)
    products = _product_rows()
    product_ids_by_name = {str(row[2]): int(row[0]) for row in products}
    product_prices = {int(row[0]): int(row[4]) for row in products}
    all_product_ids = tuple(product_prices)
    regular_customer_ids = tuple(range(3, CUSTOMER_COUNT + 1))
    weekly_customer_id = 1
    wholesale_customer_id = 2

    order_rows: list[tuple[object, ...]] = []
    item_rows: list[tuple[object, ...]] = []
    completed_units: Counter[int] = Counter()
    status_weights = (("completed", 0.88), ("cancelled", 0.08), ("refunded", 0.04))
    weekly_offsets = set(range(0, DAY_COUNT, 7))
    wholesale_offsets = {12, 24, 36, 48, 60, 72, 84}
    popular_id = product_ids_by_name[POPULAR_PRODUCT_NAME]
    declining_id = product_ids_by_name[DECLINING_PRODUCT_NAME]
    cold_beverage_ids = tuple(product_ids_by_name[name] for name in COLD_BEVERAGE_NAMES)

    for offset, order_date in enumerate(dates):
        if offset != 0:
            daily_order_count = (
                rng.randint(7, 10)
                if order_date.weekday() in (4, 5)
                else rng.randint(4, 6)
            )
            campaign_day = offset < 30 or offset >= 60
            for order_number in range(daily_order_count):
                if campaign_day and order_number == 0:
                    status = "completed"
                else:
                    status = rng.choices(
                        [entry[0] for entry in status_weights],
                        weights=[entry[1] for entry in status_weights],
                        k=1,
                    )[0]
                lines = _random_order_lines(rng, all_product_ids)
                if status == "completed":
                    if offset < 30 and order_number == 0:
                        lines.append((declining_id, rng.randint(2, 4)))
                    if offset >= 60 and order_number == 0:
                        lines.append((cold_beverage_ids[0], rng.randint(5, 7)))
                        lines.append((cold_beverage_ids[2], rng.randint(2, 4)))
                    if offset >= DAY_COUNT - 10 and order_number == 0:
                        lines.append((popular_id, rng.randint(7, 9)))
                order_prices = {
                    product_id: max(
                        1,
                        int(product_prices[product_id] * rng.uniform(0.94, 1.0)),
                    )
                    for product_id, _ in lines
                }
                _insert_order(
                    order_rows,
                    item_rows,
                    completed_units,
                    rng.choice(regular_customer_ids),
                    order_date,
                    status,
                    lines,
                    order_prices,
                )

        if offset in weekly_offsets:
            weekly_lines = [
                (product_ids_by_name[name], quantity)
                for name, quantity in WEEKLY_BASKET
            ]
            _insert_order(
                order_rows,
                item_rows,
                completed_units,
                weekly_customer_id,
                order_date,
                "completed",
                weekly_lines,
                product_prices,
            )

        if offset in wholesale_offsets:
            wholesale_lines = [
                (product_ids_by_name[name], quantity)
                for name, quantity in WHOLESALE_BASKET
            ]
            _insert_order(
                order_rows,
                item_rows,
                completed_units,
                wholesale_customer_id,
                order_date,
                "completed",
                wholesale_lines,
                product_prices,
            )

    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    reserved_path = False
    connection: sqlite3.Connection | None = None
    try:
        with path.open("xb"):
            reserved_path = True
        connection = connect_database(path)
        connection.executescript("BEGIN IMMEDIATE;\n" + schema)
        connection.executemany(
            "INSERT INTO customers (id, name, phone, city, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            customers,
        )
        connection.executemany(
            "INSERT INTO products "
            "(id, sku, name, category, selling_price_pkr, purchase_cost_pkr) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            products,
        )
        connection.executemany(
            "INSERT INTO orders (id, customer_id, order_date, status, total) "
            "VALUES (?, ?, ?, ?, ?)",
            order_rows,
        )
        connection.executemany(
            "INSERT INTO order_items (order_id, product_id, quantity, unit_price_pkr) "
            "VALUES (?, ?, ?, ?)",
            item_rows,
        )
        stock_rows = []
        for product_id in all_product_ids:
            quantity = 0 if product_id == popular_id else rng.randint(8, 80)
            stock_rows.append((product_id, quantity, rng.randint(5, 20)))
        connection.executemany(
            "INSERT INTO stock (product_id, quantity, reorder_level) VALUES (?, ?, ?)",
            stock_rows,
        )

        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise sqlite3.IntegrityError(f"Foreign-key violations found: {violations}")
        connection.commit()
    except Exception:
        if connection is not None:
            connection.rollback()
            connection.close()
            connection = None
        if reserved_path:
            path.unlink(missing_ok=True)
        raise
    finally:
        if connection is not None:
            connection.close()

    return {
        "customers": len(customers),
        "products": len(products),
        "stock": len(stock_rows),
        "orders": len(order_rows),
        "order_items": len(item_rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help=f"output SQLite database (default: {DEFAULT_DATABASE_PATH})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--reference-date",
        default=None,
        help="final date in the 90-day dataset, in YYYY-MM-DD format",
    )
    args = parser.parse_args(argv)
    try:
        counts = generate_database(
            database_path=args.database,
            seed=args.seed,
            reference_date=args.reference_date,
        )
    except (FileExistsError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(f"Created {args.database.resolve()}")
    print("Generated records: " + ", ".join(f"{table}={count}" for table, count in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
