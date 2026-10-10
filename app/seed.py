from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import hash_password
from app.db import Customer, Order


DEMO_CUSTOMERS = (
    ("CUST-789", "avery", "gold-pass", "Avery Chen", "gold"),
    ("CUST-456", "jordan", "silver-pass", "Jordan Lee", "silver"),
    ("CUST-123", "sam", "bronze-pass", "Sam Patel", "bronze"),
)

REVIEWER = ("STAFF-001", "riley", "review-pass", "Riley Morgan")


def seed(db: Session, today: date | None = None) -> None:
    existing = db.scalar(select(Customer.id).limit(1))
    if not existing:
        today = today or date.today()
        db.add_all(_customers())
        _load_orders(db, today)
    _backfill_logins(db)
    _backfill_orders(db, today or date.today())


def _customers() -> list[Customer]:
    return [
        Customer(
            id=customer_id,
            username=username,
            password_hash=hash_password(password),
            name=name,
            tier=tier,
            role="customer",
        )
        for customer_id, username, password, name, tier in DEMO_CUSTOMERS
    ]


def _backfill_logins(db: Session) -> None:
    for customer_id, username, password, name, tier in DEMO_CUSTOMERS:
        customer = db.get(Customer, customer_id)
        if customer is None:
            continue
        if not customer.username:
            customer.username = username
        if not customer.password_hash:
            customer.password_hash = hash_password(password)
        if not customer.name:
            customer.name = name
        if not customer.tier:
            customer.tier = tier
        if not customer.role:
            customer.role = "customer"
    reviewer_id, username, password, name = REVIEWER
    reviewer = db.get(Customer, reviewer_id)
    if reviewer is None:
        db.add(
            Customer(
                id=reviewer_id,
                username=username,
                password_hash=hash_password(password),
                name=name,
                tier="reviewer",
                role="reviewer",
            )
        )


def _backfill_orders(db: Session, today: date) -> None:
    if db.get(Customer, "CUST-789") is None or db.get(Order, "ORD-10005") is not None:
        return
    db.add(
        _order(
            "ORD-10005",
            "CUST-789",
            "delivered",
            "Tablet",
            "Electronics",
            "649.99",
            today - timedelta(days=4),
            "avery.chen@example.com",
            "123 Main St, Seattle, WA 98101",
        )
    )


def _load_orders(db: Session, today: date) -> None:
    db.add_all(
        [
            _order(
                "ORD-10001",
                "CUST-789",
                "delivered",
                "Laptop",
                "Electronics",
                "1999.99",
                today - timedelta(days=10),
                "avery.chen@example.com",
                "123 Main St, Seattle, WA 98101",
            ),
            _order(
                "ORD-10002",
                "CUST-789",
                "delivered",
                "Phone",
                "Electronics",
                "1199.99",
                today - timedelta(days=50),
                "avery.chen@example.com",
                "123 Main St, Seattle, WA 98101",
            ),
            _order(
                "ORD-10004",
                "CUST-789",
                "cancelled",
                "Charger",
                "Electronics",
                "29.99",
                None,
                "avery.chen@example.com",
                "123 Main St, Seattle, WA 98101",
            ),
            _order(
                "ORD-10005",
                "CUST-789",
                "delivered",
                "Tablet",
                "Electronics",
                "649.99",
                today - timedelta(days=4),
                "avery.chen@example.com",
                "123 Main St, Seattle, WA 98101",
            ),
            _order(
                "ORD-20001",
                "CUST-456",
                "delivered",
                "Headphones",
                "Electronics",
                "149.99",
                today - timedelta(days=8),
                "jordan.lee@example.com",
                "45 Pine Ave, Portland, OR 97201",
            ),
            _order(
                "ORD-30001",
                "CUST-123",
                "delivered",
                "Mug",
                "Home",
                "24.99",
                today - timedelta(days=3),
                "sam.patel@example.com",
                "9 Lake St, Austin, TX 78701",
            ),
        ]
    )


def _order(
    order_id: str,
    customer_id: str,
    status: str,
    product_name: str,
    category: str,
    amount: str,
    delivered_at: date | None,
    email: str,
    address: str,
) -> Order:
    return Order(
        order_id=order_id,
        customer_id=customer_id,
        status=status,
        product_name=product_name,
        category=category,
        total_amount=Decimal(amount),
        delivered_at=delivered_at,
        contact_email=email,
        shipping_address=address,
    )
