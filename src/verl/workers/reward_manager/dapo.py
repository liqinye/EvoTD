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

from collections import defaultdict

import torch, json

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("dapo")
class DAPORewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        **reward_kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.reward_kwargs = reward_kwargs

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        prompt_len = prompt_ids.shape[-1]

        valid_prompt_lengths = attention_mask[:, :prompt_len].sum(dim=-1)
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        eos_token = self.tokenizer.eos_token
        prompt_strs: list[str] = []
        response_strs: list[str] = []
        for i in range(len(data)):
            prompt_length = int(valid_prompt_lengths[i].item())
            response_length = int(valid_response_lengths[i].item())
            valid_prompt_ids = prompt_ids[i][-prompt_length:] if prompt_length > 0 else prompt_ids[i][:0]
            valid_response_ids = response_ids[i][:response_length]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            if eos_token and response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            prompt_strs.append(prompt_str)
            response_strs.append(response_str)

        ground_truths = [item.non_tensor_batch.get("ground_truth", None) for item in data]
        task_types = [item.non_tensor_batch.get("task_type", None) for item in data]
        codes = [item.non_tensor_batch.get("metadata", {}).get("code", None) for item in data]
        data_sources = [item.non_tensor_batch[self.reward_fn_key] for item in data]

        scores = self.compute_score(
            task_types=task_types,
            codes=codes,
            predicts=response_strs,
            ground_truths=ground_truths,
            **self.reward_kwargs,
        )

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            valid_response_length = int(valid_response_lengths[i].item())
            result = scores[i]

            if isinstance(result, dict):
                reward = result.get("overall", result.get("score"))
                for key, value in result.items():
                    reward_extra_info[key].append(value)
                acc_value = result.get("acc", result.get("accuracy", reward))
                reward_extra_info["acc"].append(acc_value)
            else:
                reward = result
                reward_extra_info["acc"].append(reward)

            # Log reward before any overlong penalty is applied.
            reward_extra_info["reward_raw"].append(reward)

            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = reward

            data_source = data_sources[i]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_strs[i])
                print("[response]", response_strs[i])
                print("[ground_truth]", ground_truths[i])
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", result)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
