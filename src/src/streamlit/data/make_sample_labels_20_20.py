from pathlib import Path
import random
import pandas as pd

covid_dir = Path("data_bin/test/covid/images")
non_dir = Path("data_bin/test/noncovid/images")
out_csv = Path("src/streamlit/data/sample_labels.csv")

N = 20
SEED = 42
exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def collect(folder):
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


random.seed(SEED)
cov = collect(covid_dir)
non = collect(non_dir)

if len(cov) < N or len(non) < N:
    raise SystemExit(
        f"Not enough images: covid={len(cov)} noncovid={len(non)} need={N} each"
    )

random.shuffle(cov)
random.shuffle(non)

rows = [{"image_path": p.as_posix(), "label": 1} for p in cov[:N]]
rows += [{"image_path": p.as_posix(), "label": 0} for p in non[:N]]

df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print(df["label"].value_counts().to_dict())
print(df.head(10).to_string(index=False))
