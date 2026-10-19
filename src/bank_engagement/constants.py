from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
DB = ROOT / "data/processed/bank_marketing.sqlite"
ARTIFACTS = ROOT / "artifacts"
DEMO = ROOT / "demo"
FEATURES = ["age", "job", "marital", "education", "default", "housing", "loan",
            "previous", "pdays", "poutcome"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "poutcome"]
NUMERIC = ["age", "previous", "pdays"]
EXCLUDED = ["duration", "contact", "month", "day_of_week", "campaign", "emp.var.rate",
            "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
RAW_COLUMNS = ["age", "job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "duration", "campaign", "pdays",
               "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
               "euribor3m", "nr.employed", "y"]
SOURCE_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
SPLITS = {"train": (1, 28831), "validation": (28832, 35009), "test": (35010, 41188)}
SEED = 42
