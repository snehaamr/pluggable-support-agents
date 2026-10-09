from app.config import Settings
from app.services import create_order_app

app = create_order_app(Settings())
