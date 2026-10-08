from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import Customer, Order


def seed(db: Session, today: date | None = None) -> None:
    existing = db.scalar(select(Customer.id).limit(1))
    if existing:
        return

    today = today or date.today()
    db.add_all(
        [
            Customer(id="CUST-789", name="Avery Chen", tier="gold"),
            Customer(id="CUST-456", name="Jordan Lee", tier="silver"),
            Customer(id="CUST-123", name="Sam Patel", tier="bronze"),
        ]
    )
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
