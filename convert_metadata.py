import pandas as pd
import pickle
import numpy as np

df = pd.read_csv("sen12ms_metadata.csv")

# remove invalid class
df = df[df.label != 17]

train_list = []
label_dict = {}

for _, row in df.iterrows():

    fname = f"ROIs1158_spring_s2_{row.roi}_p{row.patch}.tif"

    train_list.append(fname)

    vec = np.zeros(17, dtype=np.float32)

    if row.label < 17:
        vec[int(row.label)] = 1.0

    label_dict[fname] = vec

print(len(train_list))
print(len(label_dict))

with open(
    "datasets/SEN12MS/metadata/train_list_spring.pkl",
    "wb"
) as f:
    pickle.dump(train_list, f)

with open(
    "datasets/SEN12MS/metadata/IGBP_probability_labels_spring.pkl",
    "wb"
) as f:
    pickle.dump(label_dict, f)

print("done")
