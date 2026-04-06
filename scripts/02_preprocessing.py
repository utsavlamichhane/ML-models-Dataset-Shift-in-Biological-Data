import pandas as pd
from pathlib import Path


FIXED_COLUMNS = [
    "SampleID",
    "Farm_Code",
    "Weight",
    "ADG",
    "Crude_Protein",
    "Calcium",
    "Phosphorous",
    "Magnesium",
    "TDN",
]

LEVELS = ["level5", "level6", "level7"]


def preprocess_level(file_path: Path, output_dir: Path) -> None:
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()

    if "Profit" not in df.columns:
        print(f"'Profit' column not found in {file_path.name} — skipping.")
        return

    missing = [c for c in FIXED_COLUMNS if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns in {file_path.name}: {missing}")

    available_fixed = [c for c in FIXED_COLUMNS if c in df.columns]
    profit_index = df.columns.get_loc("Profit")
    microbiome_columns = list(df.columns[profit_index + 1:])

    selected = available_fixed + microbiome_columns
    processed_df = df[selected]

    output_file = output_dir / f"preprocessed_{file_path.name}"
    processed_df.to_excel(output_file, index=False)
    print(
        f"Saved: {output_file.name}  "
        f"({len(processed_df)} rows × {processed_df.shape[1]} columns)"
    )


def main(data_dir: str = ".", output_dir: str = ".") -> None:
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_files = {f.name.lower(): f for f in data_path.glob("*.xlsx")}

    for level in LEVELS:
        fname = f"data_{level}.xlsx"
        if fname not in all_files:
            print(f"File not found: {fname} — skipping.")
            continue
        preprocess_level(all_files[fname], out_path)


if __name__ == "__main__":
    main(data_dir=".", output_dir=".")
