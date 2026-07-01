# Read CSV files

import pandas as pd
import glob
# from sklearn.preprocessing import StandardScaler

files = glob.glob("../data/*.csv")
dfs=[]

for file in files:
   
    df = pd.read_csv(file)
    print(df.columns)
    df.columns = df.columns.str.strip()
    print(df["Label"].value_counts())
    dfs.append(df)

data= pd.concat(dfs, ignore_index=True)



