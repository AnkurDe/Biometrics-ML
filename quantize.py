import pandas as pd
from pathlib import Path

def load_with_dtype(file_path, float_dtype):
    """
    Load CSV with specified float dtype for all columns except 'name'.
    """
    columns = pd.read_csv(file_path, nrows=0).columns

    dtype_map = {
        col: float_dtype for col in columns if col != "name"
    }
    dtype_map["name"] = "string"

    df = pd.read_csv(
        file_path,
        dtype=dtype_map,
        low_memory=True,
        memory_map=True
    )
    return df

def saving(input_file_name:str, bit: int):
    path = Path(__file__).parent
    file_path = path / "TabulatedData" / (input_file_name+".csv")

    pth = path / "TabulatedData" / f"{input_file_name}_{bit}bit.parquet"
    print(f"Loading data as float{bit}...")
    df_r = load_with_dtype(file_path, f"float{bit}")
    df_r.info()

    print(f"Saving float{bit} parquet...")
    df_r.to_parquet(
        pth,
        engine="fastparquet"
    )


if __name__ == "__main__":
    saving(input_file_name="values_v6", bit=16)
    saving(input_file_name="values_v6", bit=32)
