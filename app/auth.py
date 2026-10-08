import hashlib
import hmac
import secrets

from sqlalchemy.orm import Session

from app.db import Customer


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 20_000)
    return f"pbkdf2${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = stored.split("$", 2)
    except ValueError:
        return False
    if scheme != "pbkdf2":
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 20_000)
    return hmac.compare_digest(digest.hex(), digest_hex)


def find_customer(db: Session, username: str, password: str) -> Customer | None:
    customer = db.query(Customer).filter(Customer.username == username).one_or_none()
    if customer is None or not verify_password(password, customer.password_hash):
        return None
    return customer
