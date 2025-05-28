import pandas as pd
import json

# 读取 JSON 文件
with open("chat_output2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 转为 DataFrame
df = pd.DataFrame(data)

# 保存为 Parquet 文件（需要 pyarrow 或 fastparquet）
df.to_parquet("val_1k.parquet", engine="pyarrow", index=False)