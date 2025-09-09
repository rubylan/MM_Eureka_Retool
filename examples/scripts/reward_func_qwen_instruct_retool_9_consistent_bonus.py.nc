import os
import re
from datetime import datetime

import torch
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify

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


def accuracy_reward_func(completion, answer):
    reward = 0.0
    response = extract_answer_with_tags(completion)
    if response != None:
        response = response
    else:
        try:
            response = completion.split("<answer>")[-1]
        except:
            response = completion.split("\n")[-1]

    content, sol = response, answer
    answer_parsed = content
    sol = f"${str(sol)}$"
    gold_parsed = parse(sol)
    if len(gold_parsed) != 0:
        answer_parsed = parse(
            content,
            extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
        )
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception:
            pass

        if reward == 0.0:
            try:
                content_match = re.search(r"<answer>(.*?)</answer>", completion)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = student_answer.replace("</answer>", "").replace("<answer>", "").strip()
                for answer in gold_parsed:
                    if str(answer).lower() in choices:
                        if str(answer).lower() in student_answer.lower():
                            choices_other = [choice for choice in choices if choice != str(answer).lower()]
                            if all(choice not in student_answer.lower() for choice in choices_other):
                                reward = 1.0
            except Exception:
                pass
    else:
        reward = 1.0
        print("Failed to parse gold solution: ", sol)

    return reward, answer_parsed


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

def format_code_func(completion):
    pattern_format =r"(.*?<code>.*?</code>\s*?<interpreter>(?:(?!<code>).*?)*?</interpreter>.*?)+"
    matches_format = re.search(pattern_format, completion, re.DOTALL)

    reward_score = 0.0
    if matches_format is None:
        return reward_score, 0

    # 正则表达式模式
    pattern_interpreter = r"<interpreter>(.*?)</interpreter>"
    # 使用 re.findall 查找所有匹配的内容
    matches_res = re.findall(pattern_interpreter, completion, re.DOTALL)

    # 检查每个解释中是否有"error"，不区分大小写
    # error_exist = False
    # for i, match_tmp in enumerate(matches_res, 1):
    #     # 使用正则表达式进行不区分大小写的查找
    #     if re.search(r"error", match_tmp, re.IGNORECASE):
    #         error_exist = True
    #         break
    reward_score = 0.5
    reward_run_acc = 0.2*len(matches_res)
    original_reward_run_acc = reward_run_acc

    for i, match_tmp in enumerate(matches_res, 0):
        # 使用正则表达式进行不区分大小写的查找
        if re.search(r"error", match_tmp, re.IGNORECASE):
            reward_run_acc -= 0.2
    # reward_score = 0.0 if error_exist else 0.5

    return original_reward_run_acc / 0.2, reward_run_acc / 0.2


def reward_func(queries, prompts, labels, global_step):
    # queries is prompts + responses

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    accuracy_rewards = []
    format_rewards = []
    run_code_times_totals = []
    run_code_times_successes = []
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
                    run_code_times_totals.append(0.0)
                    run_code_times_successes.append(0.0)
                else:
                    query1 = get_query_from_query(query)

                    accuracy_reward, answer_parsed = accuracy_reward_func(response, answer)
                    format_reward = format_reward_func(response)

                    run_code_times_total, run_code_times_success = format_code_func(response)
                    early_bonus = 0
                    if global_step < 300 and run_code_times_success > 0:
                        early_bonus = 0.5
                    reward_tmp = format_reward + accuracy_reward + early_bonus
                    rewards.append(reward_tmp)
                    accuracy_rewards.append(accuracy_reward)
                    format_rewards.append(format_reward)
                    run_code_times_totals.append(run_code_times_total)
                    run_code_times_successes.append(run_code_times_success)
                    f.write(f"===============================================================\n")
                    f.write("Query: " + query1 + "\n")
                    f.write("Response: " + response + "\n")
                    f.write("GT Answer: " + answer + "\n")
                    f.write(f"Global step: {global_step} \n Accuracy Reward: {accuracy_reward}\tFormat Reward: {format_reward} \t run_code_times_total: {run_code_times_total} \t run_code_times_success: {run_code_times_success} \t early_bonus: {early_bonus} \n\n\n\n")
                    f.write(f"===============================================================\n")
            except:
                f.write("Error: " + query + "\n")
                rewards.append(0.0)
                accuracy_rewards.append(0.0)
                format_rewards.append(0.0)
                run_code_times_totals.append(0.0)
                run_code_times_successes.append(0.0)

    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "accuracy_rewards": torch.tensor(accuracy_rewards, dtype=torch.float32),
        "format_rewards": torch.tensor(format_rewards, dtype=torch.float32),
        "run_code_times_totals": torch.tensor(run_code_times_totals, dtype=torch.float32),
        "run_code_times_successes": torch.tensor(run_code_times_successes, dtype = torch.float32)
    }
