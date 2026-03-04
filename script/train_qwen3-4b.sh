set -x

cd src

SRC_DIR=$(pwd)

project_name='evo-td'
exp_name='qwen3-4b-iter=0'

NNODES=${NNODES:-1}

TRAIN_FILE=${TRAIN_FILE:-"<train_file_path>"}
TEST_FILE=${TEST_FILE:-"<test_file_path>"}

MODEL_PATH=${MODEL_PATH:-"<model_path>"}

max_prompt_length=6144
max_response_length=8096

train_bsz=64
n_resp_per_prompt=16
train_mini_bsz=8

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28
clip_ratio_c=10.0

lr=1e-6
lr_warmup_steps=-1
weight_decay=0.01

# DAPO
# Reward
reward_manager=dapo
enable_overlong_buffer=True
overlong_buffer_len=1024
overlong_penalty_factor=1.0

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

gen_batch_size=88

loss_agg_mode="token-mean"

# Algorithm
temperature=1.0
top_p=0.99
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Dataset class overrides (use SkillTD-specific RLHF dataset)
DATASET_MODULE=${DATASET_MODULE:-"${SRC_DIR}/verl/utils/dataset/rl_dataset.py"}
DATASET_CLASS=${DATASET_CLASS:-"RLHFDataset_SkillTD"}

# Customized reward function (defaults to src/reward_function.py::compute_score)
CUSTOM_REWARD_FN_PATH=${CUSTOM_REWARD_FN_PATH:-"${SRC_DIR}/reward_function.py"}
CUSTOM_REWARD_FN_NAME=${CUSTOM_REWARD_FN_NAME:-"compute_score"}

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
offload=True
gen_tp=4

# Trainer
ngpus_per_node=4
val_before_train=False
test_freq=-1
save_freq=100
total_epochs=12
total_training_steps=null
CKPTS_DIR=${CKPTS_DIR:-"<ckpts_dir>"}
resume_mode=auto
device=cuda

python -m verl.trainer.main_dapo \
    data.train_files="'${TRAIN_FILE}'" \
    data.val_files="'${TEST_FILE}'" \
    data.custom_cls.path="${DATASET_MODULE}" \
    data.custom_cls.name="${DATASET_CLASS}" \
    custom_reward_function.path="${CUSTOM_REWARD_FN_PATH}" \
    custom_reward_function.name="${CUSTOM_REWARD_FN_NAME}" \
    data.shuffle=True \
    data.seed=42 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.gen_batch_size=${gen_batch_size} \
    data.train_batch_size=${train_bsz} \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    +data.apply_chat_template_kwargs.BASE=False \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.path="'${MODEL_PATH}'" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    +actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.log_entropy=true \
    reward_model.reward_manager=${reward_manager} \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    reward_model.overlong_buffer.log=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name="'${project_name}'" \
    trainer.experiment_name="'${exp_name}'" \
    trainer.n_gpus_per_node=${ngpus_per_node} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="'${CKPTS_DIR}'" \
    trainer.resume_mode=${resume_mode} \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    trainer.device=${device}
