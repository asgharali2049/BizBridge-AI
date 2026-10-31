CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    phone TEXT NOT NULL UNIQUE,
    city TEXT NOT NULL CHECK (length(trim(city)) > 0),
    created_at TEXT NOT NULL
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    category TEXT NOT NULL CHECK (length(trim(category)) > 0),
    selling_price_pkr INTEGER NOT NULL CHECK (selling_price_pkr >= 0),
    purchase_cost_pkr INTEGER NOT NULL CHECK (purchase_cost_pkr >= 0)
);

CREATE TABLE stock (
    product_id INTEGER PRIMARY KEY,
    quantity INTEGER NOT NULL CHECK (quantity >= 0),
    reorder_level INTEGER NOT NULL CHECK (reorder_level >= 0),
    FOREIGN KEY (product_id) REFERENCES products (id) ON DELETE CASCADE
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TEXT NOT NULL CHECK (
        length(order_date) = 10
        AND order_date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
        AND date(order_date) = order_date
    ),
    status TEXT NOT NULL CHECK (status IN ('completed', 'cancelled', 'refunded')),
    total INTEGER NOT NULL CHECK (total >= 0),
    FOREIGN KEY (customer_id) REFERENCES customers (id)
);

CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price_pkr INTEGER NOT NULL CHECK (unit_price_pkr >= 0),
    FOREIGN KEY (order_id) REFERENCES orders (id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products (id),
    UNIQUE (order_id, product_id)
);

CREATE TABLE restock_drafts (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('draft', 'confirmed', 'discarded')),
    note TEXT NOT NULL DEFAULT ''
);

CREATE TABLE restock_draft_items (
    draft_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    PRIMARY KEY (draft_id, product_id),
    FOREIGN KEY (draft_id) REFERENCES restock_drafts (id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products (id)
);

CREATE INDEX idx_orders_customer_date ON orders (customer_id, order_date);
CREATE INDEX idx_orders_date_status ON orders (order_date, status);
CREATE INDEX idx_order_items_product ON order_items (product_id);
CREATE INDEX idx_restock_draft_items_product ON restock_draft_items (product_id);
