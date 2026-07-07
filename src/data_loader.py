# Read CSV files
import pandas as pd
from pathlib import Path

def load_all_csvs(data_dir= "data"):

    project_root= Path(__file__).resolve().parent.parent
    data_path= project_root/ data_dir

    files = list(data_path.glob("*.csv"))

    dfs=[]

    for file in files:

        print(f"reading :{file.name}")
        df = pd.read_csv(file)
        print(df.columns)
        df.columns = df.columns.str.strip()
        print(df["Label"].value_counts())
        dfs.append(df)

    data= pd.concat(dfs, ignore_index=True)
 
    print("data shape:", data.shape)
    print("data head", data.head)

    return data

if __name__ == "__main__":

    data = load_all_csvs()
 

