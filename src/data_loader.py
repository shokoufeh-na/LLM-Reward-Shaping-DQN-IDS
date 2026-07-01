import pandas as pd
import glob
# from sklearn.preprocessing import StandardScaler

files = glob.glob("../data/*.csv")
dfs=[]

for file in files:
    # print("=" * 60)
    # print(file)

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    dfs.append(df)

data= pd.concat(dfs, ignore_index=True)

