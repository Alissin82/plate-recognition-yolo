import os
import pandas as pd

base_dir = "../dataset/ocr/images/"
data = []

all_labels = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
label_to_index = {label: idx for idx, label in enumerate(sorted(all_labels))}

for label in all_labels:
    label_path = os.path.join(base_dir, label)
    for f_name in os.listdir(label_path):
        fpath = os.path.join(label_path, f_name)
        data.append({
            "image_path": fpath,
            "label": label,
            "encoded": label_to_index[label]
        })

df = pd.DataFrame(data)
df.to_csv("../dataset/ocr/csv/iranis_labels_encoded.csv", index=False)

print("Saved CSV with", len(df), "samples and", len(all_labels), "labels.")
