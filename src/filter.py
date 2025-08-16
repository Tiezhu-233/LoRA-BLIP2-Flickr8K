import os
import pandas as pd

# 配置路径
base_dir = "/root/autodl-tmp/data/Flickr8k/captions.txt"
caption_file = os.path.join(base_dir, "Flickr8k.token.txt")
expert_file = os.path.join(base_dir, "ExpertAnnotations.txt")
crowd_file = os.path.join(base_dir, "CrowdFlowerAnnotations.txt")
output_file = os.path.join(base_dir, "filtered_captions_top3.csv")


# === 加载 captions ===
print("📄 加载 captions")
caps = []
with open(caption_file, 'r', encoding='utf-8') as f:
    for line in f:
        if "\t" in line:
            img_id, text = line.strip().split("\t", 1)
            caps.append((img_id.strip(), text.strip()))
df = pd.DataFrame(caps, columns=["full_image_id", "caption"])
df["full_image_id"] = df["full_image_id"].str.lower()
df["base_image_id"] = df["full_image_id"].str.split("#").str[0]
print(f"✅ 总 caption 数: {len(df)}")
print("示例 caption IDs:", df["full_image_id"].unique()[:5])


# 加载 expert score
print("🔍 读取专家评分")
e = pd.read_csv(expert_file, sep="\t", header=None,
                names=["image_file","caption_full_id","e1","e2","e3"])
e["expert_score"] = e[["e1","e2","e3"]].mean(axis=1)
expert_scores = dict(zip(e["caption_full_id"], e["expert_score"]))
df["expert_score"] = df["full_image_id"].map(expert_scores)
print(f"非空 expert_score 条数：{df['expert_score'].notnull().sum()}")

before = len(df)
df = df[df["expert_score"] >= 1.5]
print(f"专家过滤后条数：{len(df)}，过滤掉 {before-len(df)}")

# 加载 crowd score
print("🧑‍🤝‍🧑 读取群体验证评分")
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
print(f"非空 crowd_score 条数：{df['crowd_score'].notnull().sum()}")

before = len(df)
df = df[df["crowd_score"] >= 0.7]
print(f"群体验证过滤后条数：{len(df)}，过滤掉 {before-len(df)}")

# 保留每张图 top3
print("📊 保留每张图 top3 crowd_score")
df = df.sort_values(["base_image_id","crowd_score"], ascending=False)
df_top3 = df.groupby("base_image_id").head(3).reset_index(drop=True)
print(f"最终条数：{len(df_top3)}，图像数量：{df_top3['base_image_id'].nunique()}")

# 保存结果
df_top3.to_csv(output_file,index=False)
print(f"✅ 保存至 {output_file}")
