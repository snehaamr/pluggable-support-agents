from app.config import Settings
from app.services import create_refund_app

app = create_refund_app(Settings())
