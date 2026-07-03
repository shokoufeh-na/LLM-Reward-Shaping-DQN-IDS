# Build Network Flow Features & State Vector
from pyexpat import features

import numpy as np
from sklearn.preprocessing import StandardScaler
from data_loader import load_all_csvs

df= load_all_csvs() 

def build_network_flow_features(df):
   df= df.copy()
   df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names


   labels = df["Label"].copy()
   features = df.drop(columns=["Label"])

     

   scaler = StandardScaler()
   states = scaler.fit_transform(features)

   return scaler ,states

if __name__ == "__main__":
         
    data = build_network_flow_features(df)
    print(data.shape)      