import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet("/root/data/align_anything_t2t/val_1k.parquet", engine="pyarrow")  # 或 engine="fastparquet"

# 转换为 JSON 列表格式并写入文件
df.to_json(
    "output.json",
    orient="records",   # 转换为列表，每行是一个字典
    lines=False,        # 不使用 JSON Lines 格式
    force_ascii=False   # 保留非 ASCII 字符，如中文
)
