import json
import torch
from itertools import islice
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# —————— 1. 配置 ——————
device = "npu:0"                   # 或 "npu:0"
reward_model_name = "/root/data/align-anything/outputs/qwen_2_5_rm/slice_end"
tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name).to(device)
reward_model.eval()

# 文件路径
#file_in = "./origin_answers.jsonl"    # 微调前模型输出
file_in  = "./dpo_answers.jsonl"     # 微调后模型输出
#file_out  = "./orig_scored.jsonl"
file_out   = "./dpo_scored.jsonl"
batch_size = 32                       # 根据显存和模型大小调整

# ———— 批量打分函数 ————
@torch.no_grad()
def score_responses_in_batches(fin_path, fout_path, bs):
    with open(fin_path,  encoding="utf-8") as fin, \
         open(fout_path, "w", encoding="utf-8") as fout:

        # iter(fin) 对象上不断 islice，就能按 bs 取块
        it = iter(fin)
        for chunk in tqdm(iter(lambda: list(islice(it, bs)), []), desc="Scoring batches"):
            samples = [json.loads(line) for line in chunk]

            # 构造文本列表
            texts = []
            for obj in samples:
                prompt   = obj["question"]
                response = obj.get("prediction", obj.get("response"))
                texts.append(f"<s>用户：{prompt}</s><s>助手：{response}</s>")

            # batch tokenize + to(device)
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length
            ).to(device)

            # 一次 forward
            logits = reward_model(**inputs).logits  # shape (bs, 2) 或 (bs,1)
            if logits.shape[-1] == 2:
                probs = torch.softmax(logits, dim=-1)[:, 1].tolist()
            else:
                probs = logits[:, 0].tolist()

            # 写回每条记录
            for obj, score in zip(samples, probs):
                obj["reward_score"] = score
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

# 执行
score_responses_in_batches(file_in, file_out, batch_size)
print(f"打分完成，结果保存在 {file_out}")