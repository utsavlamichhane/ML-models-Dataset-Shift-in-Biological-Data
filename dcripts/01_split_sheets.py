import pandas as pd
from pathlib import Path


def split_excel_sheets(input_file: str = "data.xlsx", output_dir: str = ".") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(input_file)

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        safe_name = sheet_name.replace(" ", "_")
        output_file = output_path / f"data_{safe_name}.xlsx"
        df.to_excel(output_file, index=False)
        print(f"Saved sheet '{sheet_name}' → '{output_file}'")


if __name__ == "__main__":
    split_excel_sheets(input_file="data.xlsx", output_dir=".")
