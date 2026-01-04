# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

This version has been modified to run sequentially without using the Ray library.
"""

from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm

# 假设这些自定义模块存在且路径正确
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local


def process_item(reward_fn, data_source, response_lst, reward_data):
    """
    为单个数据项计算奖励分数。
    
    Args:
        reward_fn: 用于计算分数的奖励函数。
        data_source: 数据的来源标识。
        response_lst: 模型生成的响应列表。
        reward_data: 包含 "ground_truth" 的字典。

    Returns:
        一个元组 (data_source, score)，其中 score 是 response_lst 的平均分。
    """
    ground_truth = reward_data["ground_truth"]
    # 为每个响应计算分数
    score_lst = [reward_fn(data_source, r, ground_truth)['acc'] for r in response_lst]
    # 返回数据源和平均分
    return data_source, np.mean(score_lst)


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    # 1. 加载数据
    local_path = copy_to_local(config.data.path, use_shm=config.data.get('use_shm', False))
    dataset = pd.read_parquet(local_path)
    
    # 从DataFrame中提取所需的列
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    responses = responses.reset_index(drop=True)
    data_sources = data_sources.reset_index(drop=True)
    reward_model_data = reward_model_data.reset_index(drop=True)
    
    total = len(dataset)

    # 2. 初始化
    # 用于存储每个数据源的分数列表
    data_source_reward = defaultdict(list)
    # 根据配置获取奖励计算函数
    compute_score = get_custom_reward_fn(config)

    # 3. 串行评估循环 (替换了原来的 Ray 并行处理)
    # 使用 tqdm 显示进度条
    print(f"Starting evaluation for {total} items...")
    for i in tqdm(range(total), desc="Evaluating items"):
        # 直接调用 process_item 函数处理当前行的数据
        data_source, score = process_item(
            compute_score, 
            data_sources[i], 
            responses[i], 
            reward_model_data[i]
        )
        # 将计算出的分数添加到对应数据源的列表中
        data_source_reward[data_source].append(score)

    # 4. 计算和汇总指标
    metric_dict = {}
    # 为每个数据源计算平均分
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"test_score/{data_source}"] = np.mean(rewards)

    # 计算所有数据源的总体平均分
    # 注意：原始代码的平均分计算方式是 "所有源的平均分相加再除以源的数量"
    # 这假设了每个源的数据量是均衡的。如果数据量不均衡，直接对所有分数求平均可能更合理。
    # 这里我们保留原始的计算逻辑。
    avg_score = 0
    if data_source_reward: # 避免在字典为空时除以零
        total_avg_score = sum(metric_dict.values())
        metric_dict["test_score/avg"] = total_avg_score / len(metric_dict)
    else:
        metric_dict["test_score/avg"] = 0

    # 5. 打印结果
    print("\nEvaluation finished. Results:")
    for key, value in metric_dict.items():
        print(f"{key}: {value:.4f}")

    # 6. 保存结果到jsonl文件
    import json
    results = [{"data_source": ds, "score": np.mean(scores)} for ds, scores in metric_dict.items()]
    with open(config.data.path.replace(".parquet", "_results.jsonl"), "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()