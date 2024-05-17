from pathlib import Path

import pandas as pd

app_dir = Path(__file__).parent
data = pd.read_csv(app_dir / "tips.csv")
data = pd.read_csv(app_dir / "chdage.txt", sep = " ")