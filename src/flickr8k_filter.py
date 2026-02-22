import os
import pandas as pd

# Comment translated to English and cleaned.
base_dir = os.getenv("FLICKR8K_TEXT_DIR", os.path.join("data", "flickr8k"))
caption_file = os.path.join(base_dir, "Flickr8k.token.txt")
expert_file = os.path.join(base_dir, "ExpertAnnotations.txt")
crowd_file = os.path.join(base_dir, "CrowdFlowerAnnotations.txt")
output_dir = os.getenv("OUTPUT_DIR", os.path.join(os.getenv("PROJECT_ROOT", "."), "output"))
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "filtered_captions_top3.csv")


# Comment translated to English and cleaned.
print("  captions")
caps = []
with open(caption_file, 'r', encoding='utf-8') as f:
    for line in f:
        if "\t" in line:
            img_id, text = line.strip().split("\t", 1)
            caps.append((img_id.strip(), text.strip()))
df = pd.DataFrame(caps, columns=["full_image_id", "caption"])
df["full_image_id"] = df["full_image_id"].str.lower()
df["base_image_id"] = df["full_image_id"].str.split("#").str[0]
print(f"  caption : {len(df)}")
print(" caption IDs:", df["full_image_id"].unique()[:5])


# Comment translated to English and cleaned.
print(" ")
e = pd.read_csv(expert_file, sep="\t", header=None,
                names=["image_file","caption_full_id","e1","e2","e3"])
e["expert_score"] = e[["e1","e2","e3"]].mean(axis=1)
expert_scores = dict(zip(e["caption_full_id"], e["expert_score"]))
df["expert_score"] = df["full_image_id"].map(expert_scores)
print(f" expert_score {df['expert_score'].notnull().sum()}")

before = len(df)
df = df[df["expert_score"] >= 1.5]
print(f"{len(df)} {before-len(df)}")

# Comment translated to English and cleaned.
print(" ")
c = {}
with open(crowd_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts)>=3:
            capid = parts[0]+"#"+parts[1].split("#")[-1]
            try:
                c[capid] = float(parts[2])
            except:
                pass
df["crowd_score"] = df["full_image_id"].map(c)
print(f" crowd_score {df['crowd_score'].notnull().sum()}")

before = len(df)
df = df[df["crowd_score"] >= 0.7]
print(f"{len(df)} {before-len(df)}")

# Comment translated to English and cleaned.
print("  top3 crowd_score")
df = df.sort_values(["base_image_id","crowd_score"], ascending=False)
df_top3 = df.groupby("base_image_id").head(3).reset_index(drop=True)
print(f"{len(df_top3)}{df_top3['base_image_id'].nunique()}")

# Comment translated to English and cleaned.
df_top3.to_csv(output_file,index=False)
print(f"  {output_file}")
