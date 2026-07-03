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

   features = features.replace((np.inf, -np.inf),np.nan)
   features = features.dropna()

   labels= labels.loc[features.index]
   binary_labels = labels.apply(lambda x: 0 if x == "BENIGN" else 1)
  
 
   scaler = StandardScaler()
   states = scaler.fit_transform(features)

   return scaler ,states, binary_labels

if __name__ == "__main__":
         
    data = build_network_flow_features(df)
    print(data.shape)      