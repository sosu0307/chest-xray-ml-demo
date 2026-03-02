# src/features/build_features.py

from src.features.build_dataset import build_features_dataset, save_features_dataset

# ============================================================
# Paths - EDIT THESE AS NEEDED
# ============================================================
RAW_ROOT = r"D:\DS_ML\COVID-19_Radiography_Dataset\raw"
OUT_CSV = r"D:\DS_ML\COVID_Projekt\data\processed\sonja_features_all.csv"
OUT_CSV_REPO = r"D:\DS_ML\Repository\nov25_bds_int_covid1\data\processed\sonja_features_all.csv"

def main():
    df = build_features_dataset(
        data_root=RAW_ROOT,
        use_roi=True,
        only_both=True
    )

    if df.empty:
        raise ValueError(
            "No rows created.\n"
            f"Check that *_both images exist under:\n{RAW_ROOT}\\<class>\\images\n"
            f"and that masks exist under:\n{RAW_ROOT}\\<class>\\maske"
        )

    save_features_dataset(df, OUT_CSV)
    save_features_dataset(df, OUT_CSV_REPO)

if __name__ == "__main__":
    main()
