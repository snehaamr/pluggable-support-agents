from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Date, DateTime, ForeignKey, Numeric, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    tier: Mapped[str] = mapped_column(String, nullable=False)


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String, primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    product_name: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    delivered_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    contact_email: Mapped[str] = mapped_column(String, nullable=False)
    shipping_address: Mapped[str] = mapped_column(String, nullable=False)


class Refund(Base):
    __tablename__ = "refunds"

    refund_id: Mapped[str] = mapped_column(String, primary_key=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.order_id"), unique=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    reason: Mapped[str] = mapped_column(String, nullable=False)
    eta: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ChatMessage(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String, index=True)
    customer_id: Mapped[str] = mapped_column(String, index=True)
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MemoryFact(Base):
    __tablename__ = "memory_facts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(String, index=True)
    fact: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def make_engine(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)


def make_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)
