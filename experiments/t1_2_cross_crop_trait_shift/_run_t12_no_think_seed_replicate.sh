#!/usr/bin/env bash
# Internal launcher for the fixed T1.2 No-think seed-replication scripts.
#
# Do not invoke this helper as an experiment definition. The public run_*.sh
# wrappers hard-code coverage, replicate, and seed. This helper only keeps the
# audited R1 settings identical across those fixed wrappers.
#
# ---------------------------------------------------------------------------
# R1 protocol, read from the paper reference runs themselves
# ---------------------------------------------------------------------------
# The three No-think cells come from these W&B runs in the cross-crop
# generalization project:
#
#   4ID   acb2tf2j   finished after   1 validation row
#   8ID   9vdya9nj   finished after  12 validation rows
#   16ID  il1um9hi   finished after   2 validation rows
#
# The configuration matches the Think arm exactly:
#
#   total_epochs             2
#   total_training_steps     unset, so the target came out of the epoch count
#   gen_batch_size           16
#   train_batch_size         1024
#   max_response_length      1024
#   rollout.n                4
#   n_gpus_per_node          2
#   save_freq                100
#   test_freq                10
#   data.seed                unset
#
# **The history shape does not.** The Think references each logged 21 validation
# rows at steps 0, 10 ... 200. These three stopped at 1, 12 and 2 rows, so none
# of them reached step 200 and one barely left step 0.
#
# That is the finding, not a fault. Without the reasoning format the policy
# collapses into the no-action basin, `rollout_filter` sees every rollout
# filtered for `max_consecutive_all_filtered_steps` in a row, and training
# terminates. A replicate that runs to step 200 would mean the collapse did not
# reproduce, which is itself the result worth reporting.
#
# Two consequences for how this arm is read:
#
#   - Do not expect, or enforce, 21 validation rows. The termination step is a
#     measured quantity here, and it differs per run.
#   - `final_k_metrics()` averages the last K validation rows whatever they are,
#     so a run that stopped at row 1 yields a step-0 number. Report the
#     termination step alongside any score from this arm, or the reader cannot
#     tell a trained policy from an untrained one.
#
# The training budget is still 1600 scenarios over 2 epochs, identical to the
# Think arm, because the two are reported as a paired difference and the pair
# must differ only in the reasoning format. The row check below enforces it: the
# splits at these paths held 3200 rows and a single mixed val file until they
# were rebuilt on 2026-08-02, and that shape is not what the paper ran.
#
# ---------------------------------------------------------------------------
# Deliberate differences from R1
# ---------------------------------------------------------------------------
#   data.seed and actor.data_loader_seed
#       The point of the replicate. R1 left both unset. R2 = 81, R3 = 82. Both
#       knobs take the same number so one integer identifies the replicate.
#   trainer.total_training_steps=200
#       Made explicit. R1 inferred it from the epoch count; pinning it means an
#       interrupted replicate is reported against a declared target, and a
#       rebuilt dataset cannot silently change the length of the experiment.
#   trainer.max_actor_ckpt_to_keep=2 (and critic)
#       save_freq 100 over 200 steps writes checkpoints at 100 and 200. The
#       config default keeps 1, which deletes step 100 before writing step 200.
#       On this arm a run that collapses before step 100 leaves no checkpoint at
#       all. That is expected; the evidence for an early-terminating run is its
#       W&B history and its termination step, not a checkpoint.
#   actor.checkpoint.save_contents=[model,extra]
#       Drops optimizer state: 17.6 GB per checkpoint instead of 42.4 GB. The
#       protocol never resumes mid-run, so the optimizer state is not needed.
#   Results under $AGRI_WORK_DIR, outside the repository
#       Retained checkpoints are 17.6 GB each; put AGRI_WORK_DIR on a filesystem
#       with room for them (on Lustre, AGRI_LUSTRE_POOL selects a pool).

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 <4|8|16> <r2|r3|r4> <seed>" >&2
    exit 2
fi

COVERAGE="$1"
REPLICATE="$2"
TRAIN_SEED="$3"

case "${REPLICATE}:${TRAIN_SEED}" in
    r2:81|r3:82|r4:83) ;;
    *)
        echo "Unexpected T1.2 replicate/seed pair: ${REPLICATE}:${TRAIN_SEED}" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
RUN_NAME="schema-t12-crosscrop-${COVERAGE}id-no-think-${REPLICATE}"
DATASET_DIR="$SCRIPT_DIR/data/cross_crop_${COVERAGE}id_llm_traits_v1_23d_no_think"
TRAIN_FILE="$DATASET_DIR/train.parquet"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
RESULT_DIR="${AGRI_WORK_DIR:-$HOME/agrimanager_work}/agri_t12_results/t12_llm_train/${RUN_NAME}"

# Uniform across the three coverage settings, because R1 was uniform.
TOTAL_STEPS=200
TRAIN_BATCH_SIZE=1024
MAX_RESPONSE_LENGTH=1024
ROLLOUT_N=4
NUM_GPUS=2
RAY_CPUS=32
EXPECTED_TRAIN_ROWS=1600

# All three coverage settings validate on two named sets, id_crops and
# heldout_crops, exactly as R1 did. The held-out set is where the coverage
# scaling result lives, so collapsing the two into a single mixed val file would
# not produce the number the paper reports.
VAL_ID_FILE="$DATASET_DIR/val_id_crops.parquet"
VAL_HELDOUT_FILE="$DATASET_DIR/val_heldout_crops.parquet"

case "$COVERAGE" in
    4)
        # Rebuilt 2026-08-02 from the committed config with the weather-pool path
        # pointed back at the private repository the paper read, the same chain the
        # Think splits went through.
        EXPECTED_TRAIN_SHA256="b0cc9d13c0b24dd83adeb3d80545df176076df6859df210343abc0fb48f3c5ef"
        EXPECTED_VAL_ID_SHA256="b0edbd2c4668e6abb3ff18937c5fcaf711ece9612aefe771bb016215da48151d"
        EXPECTED_VAL_HELDOUT_SHA256="d27fd36e2ce2c7461ebdc2801689dda20239ed218eee3d30187a276ba14b459e"
        ;;
    8)
        # Rebuilt 2026-08-02, same provenance as 4ID above.
        EXPECTED_TRAIN_SHA256="a20c0d1c85aa04a4bf6161803b5204c0b2655ed6f8170f768cd4677f2b1de8b3"
        EXPECTED_VAL_ID_SHA256="ec396b786e6de0e3228843821fc0b7477bf7ed2e8e9d86b498ffa1ff0c555f2e"
        EXPECTED_VAL_HELDOUT_SHA256="dd348d9628b692d4109f5f86589030c1a0f27fedecada3c90b685302bd9a6810"
        ;;
    16)
        # Rebuilt 2026-08-02, same provenance as 4ID above.
        EXPECTED_TRAIN_SHA256="fb53a2d4bb1bc60a869aef2e411fec4eb7e03bf3b576c193c07e5f99a879be33"
        EXPECTED_VAL_ID_SHA256="8eeeb4a7fdfa563f46fae2976ea9f9c95667dccd6d03d9bee85ce3ad559601d7"
        EXPECTED_VAL_HELDOUT_SHA256="81dbb4c26ea2b01adb8f1ea67a9c8c428f09960cfa7452919c19ac21a2f10e78"
        ;;
    *)
        echo "Unsupported T1.2 coverage: $COVERAGE" >&2
        exit 2
        ;;
esac

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Missing frozen R1 dataset file: $1" >&2
        exit 1
    fi
}

verify_sha256() {
    local path="$1" expected="$2" actual
    require_file "$path"
    if [[ -z "$expected" ]]; then
        return 0
    fi
    actual="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$actual" != "$expected" ]]; then
        echo "Dataset checksum mismatch: $path" >&2
        echo "Expected: $expected" >&2
        echo "Actual:   $actual" >&2
        echo "Refusing to turn a seed replicate into a different-data experiment." >&2
        exit 1
    fi
}

verify_train_rows() {
    local path="$1" actual
    require_file "$path"
    actual="$(python -c "import pyarrow.parquet as pq,sys;print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)" "$path")"
    if [[ "$actual" != "$EXPECTED_TRAIN_ROWS" ]]; then
        echo "Train split has $actual rows; the R1 reference trained on $EXPECTED_TRAIN_ROWS." >&2
        echo "  $path" >&2
        echo "At gen_batch_size 16 over 2 epochs, $actual rows give $((actual * 2 / 16)) steps," >&2
        echo "not the 200 steps the paper reports for this cell, and the extra" >&2
        echo "scenarios are not the ones R1 saw. Rebuild the split to the" >&2
        echo "1600-scenario budget declared in the dataset config, then pin its" >&2
        echo "checksum in this script, before running the replicate." >&2
        exit 1
    fi
}

verify_train_rows "$TRAIN_FILE"
verify_sha256 "$TRAIN_FILE" "$EXPECTED_TRAIN_SHA256"
verify_sha256 "$VAL_ID_FILE" "$EXPECTED_VAL_ID_SHA256"
verify_sha256 "$VAL_HELDOUT_FILE" "$EXPECTED_VAL_HELDOUT_SHA256"

VAL_OVERRIDES=(
    "data.val_files=null"
    "+data.val_sets.id_crops=${VAL_ID_FILE}"
    "+data.val_sets.heldout_crops=${VAL_HELDOUT_FILE}"
)

mkdir -p "$LOG_DIR" "$RESULT_DIR"

# On a Lustre filesystem, set AGRI_LUSTRE_POOL to stripe the checkpoint directory
# onto a pool with room for multi-GB files (see ensure_hdd_pool).
#
# That is not hypothetical. On 2026-08-01 it truncated a step-100 checkpoint
# to 422 MB of the expected 8.8 GB per shard after three and a half hours of
# training, and killed three sibling jobs inside a minute, surfacing as
# `datasets` raising "I/O operation on closed file" from its cache write.
#
# Checkpoints go under $AGRI_WORK_DIR (default ~/agrimanager_work), outside the
# repository: each retained checkpoint is 17.6 GB.
ensure_hdd_pool() {
    # Optional: on Lustre, stripe the checkpoint directory onto the pool named by
    # AGRI_LUSTRE_POOL. Does nothing when the variable is unset or lfs is missing.
    local dir="$1" pool="${AGRI_LUSTRE_POOL:-}"
    [[ -n "$pool" ]] || return 0
    command -v lfs >/dev/null 2>&1 || return 0
    lfs getstripe -d "$dir" 2>/dev/null | grep -q "pool:[[:space:]]*$pool" && return 0
    if lfs setstripe --pool "$pool" -c 4 "$dir" 2>/dev/null; then
        echo "striped $dir onto the $pool pool"
    else
        echo "WARNING: could not stripe $dir onto $pool" >&2
    fi
}
ensure_hdd_pool "$RESULT_DIR"

cd "$PROJECT_ROOT"

# Keep all three replicates in the same project as the paper R1 runs.
# To log to your own W&B account, export WANDB_ENTITY=<your-entity>.

exec bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name agri_grpo \
    "+data.env_config_overrides.action_parser=legacy" \
    "data.train_files=${TRAIN_FILE}" \
    "${VAL_OVERRIDES[@]}" \
    "data.seed=${TRAIN_SEED}" \
    "data.validation_axis=crop" \
    "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
    "data.gen_batch_size=16" \
    "data.max_prompt_length=2048" \
    "data.max_response_length=${MAX_RESPONSE_LENGTH}" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.actor.data_loader_seed=${TRAIN_SEED}" \
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]" \
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64" \
    "trainer.logger=[\"console\", \"wandb\"]" \
    "trainer.project_name=agrimanager_wofost_cross_crop_generalization" \
    "trainer.default_local_dir=${RESULT_DIR}" \
    "trainer.experiment_name=${RUN_NAME}" \
    "trainer.total_epochs=2" \
    "trainer.total_training_steps=${TOTAL_STEPS}" \
    "trainer.test_freq=10" \
    "trainer.n_gpus_per_node=${NUM_GPUS}" \
    "trainer.save_freq=100" \
    "trainer.max_actor_ckpt_to_keep=2" \
    "trainer.max_critic_ckpt_to_keep=2" \
    "ray_kwargs.ray_init.num_cpus=${RAY_CPUS}" \
    "+ray_kwargs.ray_init.include_dashboard=False" \
    "+ray_kwargs.ray_init._temp_dir=/tmp/ray_${SLURM_JOB_ID:-${SLURM_JOBID:-$$}}"
