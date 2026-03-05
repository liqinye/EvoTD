gpus="0,1,2,3"

dataset="usaco"
proposer="o4-mini"
proposer_batch_size=100
solver_batch_size=10000000 # PLACEHOLDER: just for whole batch size
iteration=0

solver="qwen3-4b"

lower_bound=0.0
upper_bound=1.0

task_type="code_out"

CUDA_VISIBLE_DEVICES=$gpus python src/main.py \
    --dataset $dataset \
    --proposer $proposer \
    --proposer_batch_size $proposer_batch_size \
    --solver_batch_size $solver_batch_size \
    --task_type $task_type \
    --solver $solver \
    --iteration $iteration \
    --lower_bound $lower_bound \
    --upper_bound $upper_bound


# task_type="code_in"

# CUDA_VISIBLE_DEVICES=$gpus python src/main.py \
#     --dataset $dataset \
#     --proposer $proposer \
#     --proposer_batch_size $proposer_batch_size \
#     --solver_batch_size $solver_batch_size \
#     --task_type $task_type \
#     --solver $solver \
#     --iteration $iteration \
#     --lower_bound $lower_bound \
#     --upper_bound $upper_bound

# task_type="code_func"

# CUDA_VISIBLE_DEVICES=$gpus python src/main.py \
#     --dataset $dataset \
#     --proposer $proposer \
#     --proposer_batch_size $proposer_batch_size \
#     --solver_batch_size $solver_batch_size \
#     --task_type $task_type \
#     --solver $solver \
#     --iteration $iteration \
#     --lower_bound $lower_bound \
#     --upper_bound $upper_bound
