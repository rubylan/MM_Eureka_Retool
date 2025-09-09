import os
import re, regex
from datetime import datetime

import torch
# from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify
import nltk
from typing import Optional, Tuple


LOG_PATH = os.environ.get("REWARD_LOG_PATH", "reward.log")

choices = ["a", "b", "c", "d"]
problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
response_prefix = r"<\|im_start\|>assistant\n"


def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return ""
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


def get_query_from_query(q: str):
    try:
        matches = re.findall(problem_pattern, q, re.DOTALL)
        return matches[0]
    except:
        return q


def extract_answer_with_tags(text):
    match = re.search(r"(<answer>.*?</answer>)", text)
    if match:
        return match.group(1)
    return None


def preprocess_numeric_text(text: str) -> str:
    """预处理可能包含数值的文本，移除约数词、千位分隔符等"""
    # 移除约数词
    text = re.sub(r'\b(about|around|approximately|approx\.?|~)\s*', '', text.lower())
    
    # 移除所有千位分隔符
    text = re.sub(r'(\d),(?=\d)', r'\1', text)
    
    return text.strip()

def is_numeric(text: str) -> bool:
    """判断文本是否为数值（包括百分比）"""
    # 预处理文本
    text = preprocess_numeric_text(text)
    # 检查是否为数值或百分比
    return bool(re.match(r'^-?\d+(\.\d+)?%?$', text))

def to_float(text: str) -> Optional[float]:
    """将文本转换为浮点数，支持百分比转换"""
    try:
        text = preprocess_numeric_text(text)
        if text.endswith("%"):
            # 将百分比转换为浮点数
            return float(text.rstrip("%")) / 100.0  # TODO: 不确定合理性
        else:
            return float(text)
    except ValueError:
        return None

def relaxed_accuracy(target: str, prediction: str, max_relative_change: float = 0.05) -> float:
    """计算放宽的数值精度比较结果"""
    prediction_float = to_float(prediction)
    target_float = to_float(target)
    
    # # 如果任一值无法转换为浮点数，返回0
    if prediction_float is None or target_float is None:
        return 0.0

    # 计算相对误差
    relative_change = abs(prediction_float - target_float) / (abs(target_float) if target_float != 0 else 1.0)
    # 将相对误差转换为得分：误差为0得1分，误差为max_relative_change得0分，线性插值
    score = max(0.0, 1.0 - (relative_change / max_relative_change))
    return min(score, 1.0)  # 确保得分不超过1.0


def accuracy_reward_func(completion, answer, max_relative_change: float = 0.05):
    # 提取<answer>标签中的内容
    completion_match = re.findall(r'<answer>(.*?)</answer>', completion, re.DOTALL)
    extracted_completion = completion_match[-1].strip() if completion_match else ""
    
    clean_answer = answer.strip()
    clean_completion = extracted_completion.strip()

    # 判断答案是否为数值
    if is_numeric(clean_answer) and is_numeric(clean_completion):
        # 使用放宽的数值精度比较
        reward = relaxed_accuracy(clean_answer, clean_completion, max_relative_change)
    else:
        # 移除约数词和千位分隔符
        clean_answer = preprocess_numeric_text(clean_answer)
        clean_completion = preprocess_numeric_text(clean_completion)
        # 使用编辑距离
        edit_dist = nltk.edit_distance(clean_completion.lower(), clean_answer.lower()) / max(len(clean_completion), len(clean_answer), 1)
        reward = 1.0 - edit_dist

    return reward, extracted_completion


def format_reward_func(completion, **kwargs):
    pattern = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?=(?:.*<answer>){1})(?=(?:.*<\/answer>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"(?!.*<answer>.*<answer>)"
        r"(?!.*<\/answer>.*<\/answer>)"
        r".*<think>(.+?)</think>\s*<answer>.+?</answer>.*$"
    )
    matches = re.search(pattern, completion, re.DOTALL)
    return 0.5 if matches else 0.0


def reward_func(queries, prompts, labels):
    # queries is prompts + responses

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    accuracy_rewards = []
    format_rewards = []
    with open(LOG_PATH, "a") as f:
        f.write(f"----------------------------- {current_time} -----------------------------\n")
        for query, prompt, answer in zip(queries, prompts, labels):
            try:
                response = get_response_from_query(query)
                if response == "":
                    f.write("Error: " + query + "\n")
                    rewards.append(0.0)
                    accuracy_rewards.append(0.0)
                    format_rewards.append(0.0)

                else:
                    query1 = get_query_from_query(query)
                    query1_tmp = query1.split("<|vision_end|>")[1]

                    accuracy_reward, answer_parsed = accuracy_reward_func(response, answer)
                    format_reward = format_reward_func(response)

                    rewards.append(accuracy_reward + format_reward)
                    accuracy_rewards.append(accuracy_reward)
                    format_rewards.append(format_reward)
                    f.write(f"===============================================================\n")
                    f.write("Query: " + query1_tmp + "\n")
                    f.write("Response: " + response + "\n")
                    f.write("Answer: " + answer + "\n")
                    f.write(f"Accuracy Reward: {accuracy_reward}\tFormat Reward: {format_reward}\n\n\n\n")
                    f.write(f"===============================================================\n")
            except:
                f.write("Error: " + query + "\n")
                rewards.append(0.0)
                accuracy_rewards.append(0.0)
                format_rewards.append(0.0)

    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "accuracy_rewards": torch.tensor(accuracy_rewards, dtype=torch.float32),
        "format_rewards": torch.tensor(format_rewards, dtype=torch.float32),
    }
