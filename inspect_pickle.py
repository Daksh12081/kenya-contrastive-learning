import pickle

print("=== TRAIN LIST ===")
with open("datasets/SEN12MS/metadata/train_list.pkl", "rb") as f:
    train = pickle.load(f)

print(type(train))
print("Number of samples:", len(train))
print("First 5 samples:")
print(train[:5])

print("\n=== LABELS ===")
with open("datasets/SEN12MS/metadata/IGBP_probability_labels.pkl", "rb") as f:
    labels = pickle.load(f)

print(type(labels))
print("Number of labels:", len(labels))

first_key = list(labels.keys())[0]
print("\nFirst key:", first_key)
print("First label vector:")
print(labels[first_key])