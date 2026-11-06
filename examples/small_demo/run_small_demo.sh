#!/usr/bin/env bash
# AgriManager small demo: train and score a policy on a small WOFOST maize task.
#
#   bash examples/small_demo/run_small_demo.sh --gpu      # one 24 GB GPU
#   bash examples/small_demo/run_small_demo.sh --cpu-nn   # CPU only
#
# --gpu     builds the demo dataset, scores three scripted references (no action,
#           a random valid action, the scripted agronomic reference), scores the
#           untrained language model (greedy and sampled), trains it with GRPO,
#           scores the trained model, and prints a table and a training curve.
# --cpu-nn  builds the same dataset, scores the same references, trains the
#           numeric PPO baseline for a few minutes on the CPU and scores it on the
#           same scenarios.
#
# Options (defaults are the measured demo settings, see README.md):
#   --out DIR       output directory (default: examples/small_demo/runs/<mode>)
#   --model ID      language model (default: Qwen/Qwen3-0.6B)
#   --steps N       GRPO steps (default: 120)
#   --think         train and score with a reasoning block (Qwen3 thinking on,
#                   512-token answers); default is no-think (64-token answers)
#   --lr X          actor learning rate (default: 3e-6)
#   --seed N        training seed (default: 1)
#   --nn-epochs N   passes over the 128 training scenarios for PPO (default: 10)
#   --rollout-mem X fraction of GPU memory vLLM may hold during GRPO rollouts
#                   (default: 0.35, i.e. about 8.4 GB of a 24 GB card)
#   --lora-rank N   train LoRA adapters of rank N instead of all weights (needed
#                   for models above about 1B on 24 GB); 0 = full fine-tuning (default)
#   --lora-alpha X  LoRA alpha (default: 2 x rank)
#   --gen-batch N   scenarios per GRPO step (default: 8)
#   --rollout-n N   sampled episodes per scenario, the GRPO group size (default: 4)
#   --save-freq N   save a resumable checkpoint every N GRPO steps (default: 20;
#                   lower it when a time-limited job covers fewer steps)
#   --fresh         redo every phase; by default a rerun with the same --out keeps
#                   the dataset and the scores that already exist and redoes the rest
#
# Arguments after `--` are passed to entrypoints/train/train.sh as extra Hydra
# overrides. They come last, so they also override the script's own settings.
#
# Everything here is a demo on 64 held-out scenarios. It is not one of the
# paper's results; those use the frozen test suites (docs/guides/evaluation.md).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE=""
OUT=""
MODEL="Qwen/Qwen3-0.6B"
STEPS=120
THINK=0
SAVE_FREQ=20
LR=3e-6
SEED=1
NN_EPOCHS=10
ROLLOUT_MEM=0.35
GEN_BATCH=8
ROLLOUT_N=4
LORA_RANK=0
LORA_ALPHA=""
EXTRA_TRAIN_ARGS=()
FRESH=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) MODE="gpu"; shift ;;
        --cpu-nn) MODE="cpu-nn"; shift ;;
        --out) OUT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --think) THINK=1; shift ;;
        --save-freq) SAVE_FREQ="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --nn-epochs) NN_EPOCHS="$2"; shift 2 ;;
        --rollout-mem) ROLLOUT_MEM="$2"; shift 2 ;;
        --gen-batch) GEN_BATCH="$2"; shift 2 ;;
        --rollout-n) ROLLOUT_N="$2"; shift 2 ;;
        --lora-rank) LORA_RANK="$2"; shift 2 ;;
        --lora-alpha) LORA_ALPHA="$2"; shift 2 ;;
        --) shift; EXTRA_TRAIN_ARGS=("$@"); break ;;
        --fresh) FRESH=1; shift ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done
[[ -n "$MODE" ]] || { echo "choose --gpu or --cpu-nn (see --help)" >&2; exit 2; }

if [[ "$MODE" == "gpu" && "$THINK" == "1" ]]; then
    OUT="${OUT:-$SCRIPT_DIR/runs/gpu-think}"
fi
OUT="${OUT:-$SCRIPT_DIR/runs/$MODE}"
mkdir -p "$OUT/logs"
OUT="$(cd "$OUT" && pwd)"
if [[ "$FRESH" == "1" ]]; then
    rm -rf "$OUT/data" "$OUT/eval" "$OUT/grpo" "$OUT/nn_train" "$OUT/logs"/* "$OUT/timings.tsv"
fi
DATA_DIR="$OUT/data/small_demo_maize"
TIMINGS="$OUT/timings.tsv"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

phase_start() { PHASE="$1"; PHASE_T0=$(date +%s); echo; echo "=== [$(date +%H:%M:%S)] $PHASE ==="; }
phase_end() {
    local dt=$(( $(date +%s) - PHASE_T0 ))
    printf '%s\t%s\n' "$PHASE" "$dt" >> "$TIMINGS"
    echo "--- $PHASE done in ${dt}s"
}

# GPU memory sampler: logs memory.used of the whole GPU every 2 s while a GPU
# phase runs; summarize.py reports the peak. Run on an otherwise idle GPU, or the
# peak includes other processes.
GPU_SAMPLER_PID=""
start_gpu_sampler() {
    command -v nvidia-smi >/dev/null || return 0
    local idx="${CUDA_VISIBLE_DEVICES:-0}"; idx="${idx%%,*}"
    : > "$OUT/logs/gpu_memory_${1}.txt"
    ( while true; do
          nvidia-smi --id="$idx" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
              | head -1 >> "$OUT/logs/gpu_memory_${1}.txt"
          sleep 2
      done ) &
    GPU_SAMPLER_PID=$!
}
stop_gpu_sampler() {
    [[ -n "$GPU_SAMPLER_PID" ]] && kill "$GPU_SAMPLER_PID" 2>/dev/null || true
    GPU_SAMPLER_PID=""
}
trap stop_gpu_sampler EXIT

[[ "$FRESH" == "1" ]] && : > "$TIMINGS"
DEMO_T0=$(date +%s)
cat > "$OUT/settings.json" <<EOF
{"mode": "$MODE", "model": "$MODEL", "think": $THINK, "grpo_steps": $STEPS, "lr": "$LR", "seed": $SEED,
 "gen_batch": $GEN_BATCH, "rollout_n": $ROLLOUT_N, "rollout_mem": "$ROLLOUT_MEM",
 "extra_train_args": "${EXTRA_TRAIN_ARGS[*]}",
 "nn_epochs": $NN_EPOCHS, "started": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF

# ---------------------------------------------------------------------------
# 1. Dataset: 128 train, 16 val (training curve), 64 test (final scores)
# ---------------------------------------------------------------------------
if [[ ! -f "$DATA_DIR/test.parquet" ]]; then
    phase_start build_dataset
    bash entrypoints/dataset/build.sh \
        --config "$SCRIPT_DIR/config/small_demo_maize.yaml" \
        --num-workers 4 \
        --output-dir "$DATA_DIR" 2>&1 | tee "$OUT/logs/build_dataset.log" | tail -5
    phase_end
fi

# ---------------------------------------------------------------------------
# 2. Scripted references on test (scores) and val (lines on the curve):
#    no action, a uniformly random valid action, and the scripted agronomic
#    reference (ag_heuristic)
# ---------------------------------------------------------------------------
run_scripted_baselines() {
    phase_start scripted_baselines
    for SPLIT in test val; do
        for POLICY in no_action random ag_heuristic; do
            [[ -f "$OUT/eval/$SPLIT/$POLICY/metrics.json" ]] && continue
            bash entrypoints/eval/baseline_eval.sh \
                --config-name "$POLICY" \
                data.inference_file="$DATA_DIR/$SPLIT.parquet" \
                output.dir="$OUT/eval/$SPLIT/$POLICY" \
                wandb.enabled=false > "$OUT/logs/baseline_${SPLIT}_${POLICY}.log" 2>&1
        done
    done
    phase_end
}
BASELINES_PID=""
if [[ "$MODE" == "gpu" ]]; then
    # CPU-only work: run it beside the GPU phases.
    run_scripted_baselines > "$OUT/logs/scripted_baselines.log" 2>&1 &
    BASELINES_PID=$!
    echo "scripted baselines running in the background (log: $OUT/logs/scripted_baselines.log)"
else
    run_scripted_baselines
fi

if [[ "$MODE" == "cpu-nn" ]]; then
    # -----------------------------------------------------------------------
    # 3. Numeric PPO baseline (Stable-Baselines3, CPU)
    # -----------------------------------------------------------------------
    phase_start nn_ppo_train
    NN_DIR="$OUT/nn_train"
    rm -rf "$NN_DIR"
    bash entrypoints/train/nn_train.sh \
        "data.train_files=$DATA_DIR/train.parquet" \
        "data.val_files=$DATA_DIR/val.parquet" \
        "output.save_folder=$NN_DIR" \
        "agent.exp_name=small_demo_nn_ppo" \
        "agent.track=false" \
        "agent.seed=$SEED" \
        "agent.num_envs=8" \
        "agent.num_steps=72" \
        "agent.num_minibatches=4" \
        "agent.n_epochs=1" \
        "agent.checkpoint_mode=epoch" \
        "runtime.device=cpu" \
        "runtime.train_epochs=$NN_EPOCHS" \
        "runtime.validation.enabled=true" \
        "runtime.validation.val_before_train=true" \
        "runtime.validation.evals_per_epoch=2" \
        "runtime.validation.deterministic=true" \
        "runtime.validation.log_to_wandb=false" \
        > "$OUT/logs/nn_train.log" 2>&1
    phase_end

    phase_start nn_ppo_eval
    AGENT="$(find "$NN_DIR" -name agent.zip -not -path '*/checkpoints/*' | head -1)"
    [[ -n "$AGENT" ]] || AGENT="$(find "$NN_DIR" -name agent.zip | sort | tail -1)"
    # Sampled actions are the evaluator's default and what the paper reports for
    # the neural baselines; the most likely action is shown as well.
    bash entrypoints/eval/nn_eval.sh \
        "data.inference_file=$DATA_DIR/test.parquet" \
        "agent.path=$AGENT" \
        "runtime.device=cpu" \
        "output.dir=$OUT/eval/test/nn_ppo" > "$OUT/logs/nn_eval.log" 2>&1
    bash entrypoints/eval/nn_eval.sh \
        "data.inference_file=$DATA_DIR/test.parquet" \
        "agent.path=$AGENT" \
        "runtime.device=cpu" \
        "runtime.deterministic=true" \
        "output.dir=$OUT/eval/test/nn_ppo_argmax" > "$OUT/logs/nn_eval_argmax.log" 2>&1
    phase_end
else
    # -----------------------------------------------------------------------
    # 3. Language model: untrained score, GRPO training, trained score
    # -----------------------------------------------------------------------
    python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("--gpu needs a CUDA GPU; use --cpu-nn on a CPU-only machine.")
free, total = torch.cuda.mem_get_info()
print(f"GPU: {torch.cuda.get_device_name(0)}, {free/2**30:.1f} of {total/2**30:.1f} GiB free")
if free < 20 * 2**30:
    print("warning: less than 20 GiB free; the demo is sized for an otherwise idle 24 GB GPU")
PY
    if [[ "$THINK" == "1" ]]; then
        # Qwen3's own thinking block, required by the environment as
        # <think>...</think><answer>...</answer>; 512 tokens as in the paper's runs.
        EVAL_CFG="$SCRIPT_DIR/config/vllm_think.yaml"
        RESPONSE_LEN=512
        ENABLE_THINKING=True
        FORMAT_OVERRIDES=(
            "+data.env_config_overrides.require_think=True"
            "+data.env_config_overrides.thinking_mode=think"
            "+data.env_config_overrides.think_tag=think"
        )
        EVAL_FORMAT_ARGS=("${FORMAT_OVERRIDES[@]}" "+data.assert_env_config.require_think=True")
    else
        EVAL_CFG="$SCRIPT_DIR/config/vllm_no_think.yaml"
        RESPONSE_LEN=64
        ENABLE_THINKING=False
        FORMAT_OVERRIDES=()
        EVAL_FORMAT_ARGS=()
    fi

    llm_eval() {  # name model_path temperature
        [[ -f "$OUT/eval/test/$1/metrics.json" ]] && return 0
        bash entrypoints/eval/eval.sh \
            data.inference_file="$DATA_DIR/test.parquet" \
            model.config="$EVAL_CFG" \
            model.path="$2" \
            runtime.temperature="$3" \
            runtime.max_tokens=$RESPONSE_LEN \
            "${EVAL_FORMAT_ARGS[@]}" \
            output.dir="$OUT/eval/test/$1" > "$OUT/logs/eval_$1.log" 2>&1
    }

    # Greedy decoding (temperature 0) is the paper's evaluation protocol. The
    # demo also scores sampled decoding (temperature 1, as in GRPO rollouts),
    # because a short run moves the sampled policy before it moves the greedy one.
    if [[ ! -f "$OUT/eval/test/untrained_sampled/metrics.json" ]]; then
        phase_start eval_untrained
        start_gpu_sampler eval_untrained
        llm_eval untrained "$MODEL" 0
        llm_eval untrained_sampled "$MODEL" 1.0
        stop_gpu_sampler
        phase_end
    fi

    TRAIN_DIR="$OUT/grpo"
    LORA_ALPHA="${LORA_ALPHA:-$(( 2 * LORA_RANK ))}"
    # VeRL sizes vLLM's context from max_position_embeddings (40,960 for Qwen3),
    # which leaves no room for the KV cache of a 1.7B model in 8 GB; the demo's
    # prompt plus answer never exceeds 1,536 + RESPONSE_LEN tokens.
    if [[ "$LORA_RANK" -gt 0 ]]; then
        # Frozen base weights in bf16 and offloaded between phases; vLLM loads
        # the base from safetensors and receives only the adapters.
        WEIGHT_ARGS=(
            "actor_rollout_ref.model.lora_rank=$LORA_RANK"
            "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA"
            "actor_rollout_ref.model.target_modules=all-linear"
            "actor_rollout_ref.rollout.load_format=safetensors"
            "actor_rollout_ref.rollout.layered_summon=True"
            "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16"
            "actor_rollout_ref.actor.fsdp_config.param_offload=True"
            "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]"
        )
    else
        WEIGHT_ARGS=("actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]")
    fi
    RUN_KEY="$MODEL lr=$LR seed=$SEED steps=$STEPS gen=$GEN_BATCH n=$ROLLOUT_N think=$THINK lora=$LORA_RANK/$LORA_ALPHA ${EXTRA_TRAIN_ARGS[*]}"
    SAME_RUN=0
    [[ -f "$TRAIN_DIR/run_key.txt" && "$(cat "$TRAIN_DIR/run_key.txt")" == "$RUN_KEY" ]] && SAME_RUN=1
    if [[ "$SAME_RUN" != "1" || ! -f "$TRAIN_DIR/finished" ]]; then
    phase_start grpo_train
    # A run with the same settings resumes from its last checkpoint (saved every
    # $SAVE_FREQ steps), so an interrupted or time-limited job can be rerun.
    if [[ "$SAME_RUN" == "1" && -f "$TRAIN_DIR/checkpoints/latest_checkpointed_iteration.txt" ]]; then
        echo "resuming GRPO from step $(cat "$TRAIN_DIR/checkpoints/latest_checkpointed_iteration.txt")"
    else
        rm -rf "$TRAIN_DIR" "$OUT/eval/test/trained" "$OUT/eval/test/trained_sampled"
        mkdir -p "$TRAIN_DIR"
        printf '%s' "$RUN_KEY" > "$TRAIN_DIR/run_key.txt"
    fi
    LAUNCH=$(find "$TRAIN_DIR" -maxdepth 1 -name 'metrics_*.jsonl' | wc -l)
    export VERL_FILE_LOGGER_PATH="$TRAIN_DIR/metrics_$(( LAUNCH + 1 )).jsonl"
    ray stop --force >/dev/null 2>&1 || true
    start_gpu_sampler grpo_train
    echo "GRPO: $STEPS steps; follow with: tail -f $OUT/logs/grpo_train.log"
    if ! bash entrypoints/train/train.sh \
        --log-file "$OUT/logs/grpo_train.log" \
        --config-name agri_grpo \
        "data.train_files=$DATA_DIR/train.parquet" \
        "data.val_files=$DATA_DIR/val.parquet" \
        "data.seed=$SEED" \
        "data.gen_batch_size=$GEN_BATCH" \
        "data.max_prompt_length=1536" \
        "data.max_response_length=$RESPONSE_LEN" \
        "data.dataloader_num_workers=0" \
        "+data.apply_chat_template_kwargs.enable_thinking=$ENABLE_THINKING" \
        "${FORMAT_OVERRIDES[@]}" \
        "actor_rollout_ref.model.path=$MODEL" \
        "actor_rollout_ref.model.enable_gradient_checkpointing=True" \
        "actor_rollout_ref.actor.optim.lr=$LR" \
        "actor_rollout_ref.actor.data_loader_seed=$SEED" \
        "actor_rollout_ref.actor.ppo_mini_batch_size=64" \
        "actor_rollout_ref.actor.use_dynamic_bsz=True" \
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048" \
        "actor_rollout_ref.actor.use_torch_compile=False" \
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" \
        "${WEIGHT_ARGS[@]}" \
        "actor_rollout_ref.rollout.n=$ROLLOUT_N" \
        "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_MEM" \
        "actor_rollout_ref.rollout.max_num_seqs=256" \
        "+actor_rollout_ref.model.override_config.max_position_embeddings=$(( 1536 + RESPONSE_LEN ))" \
        "actor_rollout_ref.rollout.agent.num_workers=4" \
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True" \
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096" \
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True" \
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096" \
        "trainer.total_epochs=1000" \
        "trainer.total_training_steps=$STEPS" \
        "trainer.val_before_train=True" \
        "trainer.test_freq=5" \
        "trainer.save_freq=$SAVE_FREQ" \
        "trainer.resume_mode=auto" \
        "trainer.max_actor_ckpt_to_keep=1" \
        "trainer.n_gpus_per_node=1" \
        'trainer.logger=["console","file"]' \
        "trainer.project_name=agrimanager_small_demo" \
        "trainer.experiment_name=small_demo_grpo" \
        "trainer.default_local_dir=$TRAIN_DIR/checkpoints" \
        "ray_kwargs.ray_init.num_cpus=8" \
        "+ray_kwargs.ray_init.include_dashboard=False" \
        "${EXTRA_TRAIN_ARGS[@]}" \
        > /dev/null 2> "$OUT/logs/grpo_train_launcher.log"; then
        echo "GRPO training failed; the end of $OUT/logs/grpo_train.log:" >&2
        tail -n 40 "$OUT/logs/grpo_train.log" >&2
        exit 1
    fi
    stop_gpu_sampler
    ray stop --force >/dev/null 2>&1 || true
    touch "$TRAIN_DIR/finished"
    phase_end
    # Keep only the last policy in Transformers layout; the resumable state
    # (sharded model and optimizer, about 7 GB per checkpoint) is not needed now.
    LAST_STEP="$(cat "$TRAIN_DIR/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null || true)"
    if [[ -n "$LAST_STEP" ]]; then
        rm -f "$TRAIN_DIR/checkpoints/global_step_$LAST_STEP/actor/"*.pt
        find "$TRAIN_DIR/checkpoints" -maxdepth 1 -name 'global_step_*' \
            ! -name "global_step_$LAST_STEP" -exec rm -rf {} +
    fi
    fi

    phase_start eval_trained
    # The rollout filter can end a run early (five steps in a row with no
    # informative group); score the last checkpoint that was written.
    ACTOR_DIR="$(ls -d "$TRAIN_DIR"/checkpoints/global_step_*/actor 2>/dev/null | sort -V | tail -1 || true)"
    if [[ -n "$ACTOR_DIR" && -f "$ACTOR_DIR/lora_adapter/adapter_config.json" && ! -f "$ACTOR_DIR/huggingface/model.safetensors.index.json" && ! -f "$ACTOR_DIR/huggingface/model.safetensors" ]]; then
        echo "merging the LoRA adapter into $MODEL"
        python "$SCRIPT_DIR/merge_lora.py" --base "$MODEL" --adapter "$ACTOR_DIR/lora_adapter" \
            --out "$ACTOR_DIR/huggingface" > "$OUT/logs/merge_lora.log" 2>&1
    fi
    HF_DIR="${ACTOR_DIR:+$ACTOR_DIR/huggingface}"
    [[ -n "$HF_DIR" && -f "$HF_DIR/config.json" ]] || { echo "no trained checkpoint under $TRAIN_DIR/checkpoints" >&2; exit 1; }
    start_gpu_sampler eval_trained
    llm_eval trained "$HF_DIR" 0
    llm_eval trained_sampled "$HF_DIR" 1.0
    stop_gpu_sampler
    phase_end
fi

# ---------------------------------------------------------------------------
# 4. Table and training curve
# ---------------------------------------------------------------------------
if [[ -n "$BASELINES_PID" ]]; then
    wait "$BASELINES_PID" || { echo "scripted baselines failed; see $OUT/logs/" >&2; exit 1; }
fi
printf 'total\t%s\n' "$(( $(date +%s) - DEMO_T0 ))" >> "$TIMINGS"
python "$SCRIPT_DIR/summarize.py" --out "$OUT"
