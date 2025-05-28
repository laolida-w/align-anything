from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from itertools import islice

model_name = "/root/data/Qwen2.5-0.5B-Instruct"
dpo_model_name = "/root/align-anything/outputs/qwen_2_5_dpo/slice_end"
device = "npu:0"

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dpo_model = AutoModelForCausalLM.from_pretrained(dpo_model_name).to(device)
dpo_tokenizer = AutoTokenizer.from_pretrained(dpo_model_name)

def batch_chat(prompts, model, tokenizer, max_new_tokens=512):
    # 1. 构造所有 messages → texts 列表
    texts = []
    for prompt in prompts:
        msgs = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user",   "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

    # 2. 一次性 batch 编码
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    # 3. 一次性生成
    with torch.no_grad():
        generated = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            # 你可根据显存加上 num_beams, do_sample 等选项
        )

    # 4. 拆分出每条的回复
    responses = []
    input_ids = enc["input_ids"]
    for input_id, output_id in zip(input_ids, generated):
        # 删掉 prompt 部分
        resp_ids = output_id[len(input_id):]
        resp = tokenizer.decode(resp_ids, skip_special_tokens=True)
        responses.append(resp)
    return responses

# 读取所有问题
with open("valset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

prompts = [item["question"] for item in data]
batch_size = 50  # 根据显存调整

new_data = []
for i in range(0, len(prompts), batch_size):
    batch_prompts = prompts[i : i + batch_size]

    # 分别 batch 生成 DPO 和 Base 回复
    dpo_resps  = batch_chat(batch_prompts, dpo_model, dpo_tokenizer)
    base_resps = batch_chat(batch_prompts, model,   tokenizer)

    # 拼回结果
    for prompt, r1, r2 in zip(batch_prompts, dpo_resps, base_resps):
        new_data.append({
            "question": prompt,
            "response_1": r1,
            "response_2": r2,
            "overall_response": 1
        })
    print(f"Processed {i + len(batch_prompts)} / {len(prompts)}")

# 保存
with open("chat_output2.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)
