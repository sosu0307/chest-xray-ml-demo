from pathlib import Path
import pandas as pd

# -------------------------
# Fixed paths (as requested)
# -------------------------
STREAMLIT_DIR = Path(__file__).resolve().parent  # src/streamlit
PROJECT_ROOT = STREAMLIT_DIR.parent.parent  # repo root

CSV_IN = PROJECT_ROOT / "models" / "results" / "resnet50_probs_all.csv"
CSV_OUT = STREAMLIT_DIR / "data" / "demo4_borderline.csv"
CSV_OUT.parent.mkdir(parents=True, exist_ok=True)

COVID_IMG_DIR = PROJECT_ROOT / "data_bin" / "test" / "covid" / "images"
NONCOVID_IMG_DIR = PROJECT_ROOT / "data_bin" / "test" / "noncovid" / "images"

CENTER = 0.40
N_PER_CLASS = 2  # exactly 2 + 2 = 4


def pick_first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def main():
    if not CSV_IN.exists():
        raise SystemExit(f"Input CSV not found: {CSV_IN}")

    if not COVID_IMG_DIR.exists() or not NONCOVID_IMG_DIR.exists():
        raise SystemExit("Image directories not found in data_bin/test/.../images")

    df = pd.read_csv(CSV_IN)
    if df.empty:
        raise SystemExit("Input CSV is empty")

    prob_col = pick_first_existing(
        df.columns, ["prob_covid", "p_covid", "covid_prob", "prob", "probability", "p1"]
    )
    path_col = pick_first_existing(
        df.columns, ["image_path", "img_path", "path", "filepath", "file_path", "image"]
    )

    if prob_col is None or path_col is None:
        raise SystemExit(f"Missing columns. Need prob+path. Found: {list(df.columns)}")

    w = df.copy()
    w["prob_covid"] = pd.to_numeric(w[prob_col], errors="coerce")
    w = w[w["prob_covid"].notna()].copy()

    # normalize filename from CSV path
    w["filename"] = w[path_col].astype(str).apply(lambda s: Path(s).name)

    # keep only files that really exist in fixed image dirs
    covid_files = {p.name for p in COVID_IMG_DIR.glob("*") if p.is_file()}
    noncovid_files = {p.name for p in NONCOVID_IMG_DIR.glob("*") if p.is_file()}

    # build candidates for each class via filename membership
    covid_df = w[w["filename"].isin(covid_files)].copy()
    covid_df["true_class"] = "covid"
    covid_df["abs_path"] = covid_df["filename"].apply(lambda n: str(COVID_IMG_DIR / n))

    non_df = w[w["filename"].isin(noncovid_files)].copy()
    non_df["true_class"] = "noncovid"
    non_df["abs_path"] = non_df["filename"].apply(lambda n: str(NONCOVID_IMG_DIR / n))

    # distance to threshold center
    covid_df["distance"] = (covid_df["prob_covid"] - CENTER).abs()
    non_df["distance"] = (non_df["prob_covid"] - CENTER).abs()

    # take top 2 each
    covid_pick = (
        covid_df.sort_values("distance", ascending=True).head(N_PER_CLASS).copy()
    )
    non_pick = non_df.sort_values("distance", ascending=True).head(N_PER_CLASS).copy()

    out = pd.concat([covid_pick, non_pick], axis=0, ignore_index=True)
    out = out[["true_class", "filename", "abs_path", "prob_covid", "distance"]].copy()

    if out.empty:
        raise SystemExit(
            "No matching images found between CSV and fixed image folders."
        )

    out.to_csv(CSV_OUT, index=False)

    print("Done.")
    print(f"Saved: {CSV_OUT}")
    print(f"Rows: {len(out)}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
