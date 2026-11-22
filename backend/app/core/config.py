import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATA_FILE = DATA_DIR / "online_retail.csv"
FALLBACK_EXCEL = Path("D:/MIT/Sem 7/Gen AI/online+retail/Online Retail.xlsx")

PROJECT_NAME = "RetailIQ API"
VERSION = "1.0.0"
API_V1_PREFIX = "/api"

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
