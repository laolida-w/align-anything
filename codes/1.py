import json

# 逐行读取 JSONL 文件，构造列表
with open("dpo_scored.jsonl", "r", encoding="utf-8") as f:
    data1 = [json.loads(line) for line in f]

with open("orig_scored.jsonl", "r", encoding="utf-8") as f:
    data2 = [json.loads(line) for line in f]

# 检查长度一致
assert len(data1) == len(data2), "两个文件长度不一致"

# 合并内容
new_data = []
for idx, (item1, item2) in enumerate(zip(data1, data2)):
    if item1["question"] != item2["question"]:
        print(f"第 {idx} 行问题不一致")
        break
   
    new_data.append({
        "question": item1["question"],
        "response_1": item1["prediction"].split("\nassistant\n", 1)[-1].strip(),   # DPO模型输出
        "response_2": item2["prediction"],    # 原始模型输出
        "overall_response": 1                # 固定标记优劣（可以改为真实对比结果）
    })

# 保存为 JSON 文件
with open("newvalsetgbr.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)
