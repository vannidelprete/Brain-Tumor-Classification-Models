#============= IMPORT ==========================
import pandas as pd
import numpy as np

brain_tumor_data = pd.read_csv(r"Data/bt_dataset_t3.csv")

#============= FIX MISSING DATA =================
del brain_tumor_data["Image"]
brain_tumor_data = brain_tumor_data.replace(np.inf, 999)
bt_rep = brain_tumor_data.mean()
brain_tumor_data = brain_tumor_data.fillna(bt_rep)

brain_tumor_data.to_csv(r"Data/bt_dataset_t3_fixed.csv")