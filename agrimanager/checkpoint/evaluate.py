"""E5: evaluate one published checkpoint against the frozen E4 suites.

Given a checkpoint repository and a set of suite groups, this fetches both,
evaluates every split, and publishes one private result bundle carrying the
per-scenario outputs, the metrics, the failures, and a manifest that ties each
number to the checkpoint and suite revision that produced it.

Two properties matter more than convenience here.

**The reasoning format is read from the checkpoint, not supplied by hand.** The
frozen suites are policy-neutral, so a Think-trained policy evaluated as-is is
silently scored in No-think mode. The checkpoint's own ``run_manifest.json``
records what it trained under; that value is applied as an override and then
asserted, so a mismatch fails the run instead of producing a plausible number.

**A failed split is recorded, not skipped.** Evaluations are long and partial
failures are normal; dropping them quietly would leave a bundle that looks
complete. Every split ends up in the manifest with its status.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .publish import (
    PublishError,
    collect_forbid_terms,
    find_identity_leaks,
    guard_folder,
    _guard_text,
    _sha256,
    _versions,
)

# Which suite groups belong to which experiment. Passing them explicitly is
# still allowed; this is the default so the common case cannot be mismatched.
SUITE_GROUPS: dict[str, tuple[str, ...]] = {
    "t11": ("e4_t11_potato_weather_regime",),
    # Chickpea is a separate tier rather than a second group under t11: the two
    # crops are separate cells with separate checkpoints, and scoring one
    # checkpoint against both crops' suites would be meaningless.
    "t11_chickpea": ("e4_t11_chickpea_weather_regime",),
    "t12": ("e4_t12_cross_crop",),
    "t13": ("e4_t13_price_regime",),
    "t21": ("e4_t21_observation_schema",),
    "t22": ("e4_t22_action_menu",),
    "t23": ("e4_t23_reward_formulation",),
    "t31": (
        "e4_t31_cross_sim_wofost",
        "e4_t31_cross_sim_dssat",
        "e4_t31_cross_sim_cycles",
    ),
    "t32": (
        "e4_t32_wofost_seen",
        "e4_t32_wofost_heldout",
        "e4_t32_dssat_yield",
        "e4_t32_dssat_profit",
        "e4_t32_cycles_yield",
        "e4_t32_cycles_profit",
    ),
}

# Each value is the axis the matching suite manifest declares. They are read
# from those manifests, not chosen here, so a mismatch means the suite and the
# evaluator disagree about what the experiment varies.
VALIDATION_AXIS: dict[str, str] = {
    "t11": "weather_regime",
    "t11_chickpea": "weather_regime",
    "t12": "crop",
    "t13": "price_regime",
    "t21": "observation_schema",
    "t22": "action_menu",
    "t23": "reward_formulation",
    "t31": "simulator",
    "t32": "schema_tuple",
}


# Resource locations each record carries so the environment can be rebuilt.
# They are reconstructed from the consumer's own checkout, so the stored value
# is redundant, and it names both the project and sometimes the host.
RESOURCE_PATH_KEYS = frozenset({
    "wofost_gym_path", "crop_traits_dir", "weather_cache_dir", "dssat_gym_path",
    "cycles_gym_path", "cycles_path", "save_folder", "log_saving_path",
    "model_path", "auxiliary_file_paths",
})

REDACTED = "<resolved from the consumer's checkout>"


def _looks_like_path(value: str) -> bool:
    """A filesystem location rather than prose.

    Model prompts and responses are prose and must survive untouched; a path has
    no spaces and either starts at the root, climbs out of the tree, or contains
    a separator.
    """
    if not value or " " in value or "\n" in value:
        return False
    return value.startswith(("/", "../", "./")) or "/" in value


def _redact_value(value: Any) -> Any:
    """Redact a resource value while preserving its shape."""
    if isinstance(value, list):
        return [REDACTED for _ in value]
    return REDACTED


def redact_resource_paths(value: Any) -> Any:
    """Remove filesystem locations from a decoded JSON structure.

    Keys known to hold resource locations are redacted outright. Everything else
    is redacted only when the value itself looks like a path, because listing
    keys one at a time kept missing new ones -- `cycles_gym_path` and a list
    under `env_params.auxiliary_file_paths` each surfaced only after the guard
    rejected a bundle. Prose is left alone, so prompts and responses survive.
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in RESOURCE_PATH_KEYS:
                out[key] = _redact_value(item)
            else:
                out[key] = redact_resource_paths(item)
        return out
    if isinstance(value, list):
        return [redact_resource_paths(item) for item in value]
    if isinstance(value, str) and _looks_like_path(value) and find_identity_leaks(value):
        return REDACTED
    return value


def redact_json_file(path: Path) -> None:
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    path.write_text(json.dumps(redact_resource_paths(payload), indent=2) + "\n")


class EvaluationError(RuntimeError):
    """Raised when the evaluation cannot be trusted to mean what it claims."""


@dataclass
class Reasoning:
    """The reasoning format a checkpoint was trained under."""

    require_think: bool
    thinking_mode: str | None
    think_tag: str | None
    source: str  # "checkpoint manifest" or "operator-supplied"

    def overrides(self) -> dict[str, Any]:
        out: dict[str, Any] = {"require_think": self.require_think}
        if self.thinking_mode:
            out["thinking_mode"] = self.thinking_mode
        if self.think_tag:
            out["think_tag"] = self.think_tag
        return out

    def assertions(self) -> dict[str, Any]:
        # thinking_mode is deliberately excluded: "think" and
        # "grounding_decision" are aliases that normalize to the same prompt, so
        # asserting the literal string would fail on an equivalent suite.
        out: dict[str, Any] = {"require_think": self.require_think}
        if self.think_tag:
            out["think_tag"] = self.think_tag
        return out


@dataclass
class EvalSpec:
    checkpoint_repo: str
    revision: str
    suite_repo: str
    tier: str
    results_repo: str
    suite_groups: tuple[str, ...] = ()
    max_samples: int | None = None
    num_trials: int = 1
    forbid_terms: tuple[str, ...] = field(default_factory=tuple)
    # Weights already on disk, for a model that is not one of our published
    # checkpoints -- the untrained base model, evaluated as a control. The
    # identity written into every manifest still comes from checkpoint_repo and
    # revision, so a local path never reaches an artifact.
    checkpoint_dir: Path | None = None
    # Audited suites already present on the evaluation host. The repository
    # identity is still recorded in the bundle, but no network fetch is needed.
    suite_dir: Path | None = None


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #

def fetch_checkpoint(spec: EvalSpec, work_dir: Path, token: str | None) -> Path:
    if spec.checkpoint_dir is not None:
        if not (spec.checkpoint_dir / "run_manifest.json").is_file():
            raise EvaluationError(
                f"{spec.checkpoint_dir} has no run_manifest.json. A local model "
                f"still has to declare what it is; write one rather than letting "
                f"the bundle claim an identity nothing checked."
            )
        return spec.checkpoint_dir

    from huggingface_hub import snapshot_download

    target = work_dir / "checkpoint"
    snapshot_download(
        spec.checkpoint_repo,
        revision=spec.revision,
        local_dir=str(target),
        token=token,
    )
    return target


def fetch_suites(spec: EvalSpec, work_dir: Path, token: str | None) -> Path:
    if spec.suite_dir is not None:
        missing = [
            group
            for group in spec.suite_groups
            if not (spec.suite_dir / group).is_dir()
        ]
        if missing:
            raise EvaluationError(
                f"{spec.suite_dir} is missing suite groups: {', '.join(missing)}"
            )
        return spec.suite_dir

    from huggingface_hub import snapshot_download

    target = work_dir / "suites"
    snapshot_download(
        spec.suite_repo,
        repo_type="dataset",
        local_dir=str(target),
        allow_patterns=[f"{group}/*" for group in spec.suite_groups],
        token=token,
    )
    return target


def read_checkpoint_manifest(checkpoint_dir: Path) -> dict[str, Any]:
    path = checkpoint_dir / "run_manifest.json"
    if not path.is_file():
        raise EvaluationError(
            f"{path} is absent, so the checkpoint cannot say which run or "
            f"reasoning format it came from"
        )
    return json.loads(path.read_text())


def resolve_reasoning(
    manifest: dict[str, Any],
    override_require_think: bool | None,
    override_think_tag: str | None,
) -> Reasoning:
    """Take the training condition from the checkpoint whenever it records one."""
    recorded = ((manifest.get("training") or {}).get("reasoning") or {})
    require_think = recorded.get("require_think")

    if require_think is not None:
        if override_require_think is not None and override_require_think != require_think:
            raise EvaluationError(
                f"--require-think={override_require_think} contradicts the "
                f"checkpoint manifest, which records {require_think}. Resolve the "
                f"disagreement rather than overriding it."
            )
        return Reasoning(
            require_think=bool(require_think),
            thinking_mode=recorded.get("thinking_mode"),
            think_tag=recorded.get("think_tag"),
            source="checkpoint manifest",
        )

    if override_require_think is None:
        raise EvaluationError(
            "the checkpoint manifest records no reasoning format and none was "
            "supplied. Pass --require-think explicitly; evaluating under the "
            "suite default would silently score the policy in the wrong mode."
        )
    return Reasoning(
        require_think=override_require_think,
        thinking_mode=None,
        think_tag=override_think_tag,
        source="operator-supplied",
    )


def discover_splits(suites_dir: Path, groups: Sequence[str]) -> list[tuple[str, Path]]:
    splits: list[tuple[str, Path]] = []
    for group in groups:
        group_dir = suites_dir / group
        if not group_dir.is_dir():
            raise EvaluationError(f"suite group {group} is absent from {suites_dir}")
        found = sorted(group_dir.glob("test*.parquet"))
        if not found:
            raise EvaluationError(f"suite group {group} contains no test parquet")
        splits.extend((group, path) for path in found)
    return splits


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

def _hydra_bool(value: bool) -> str:
    return "True" if value else "False"


def evaluate_split(
    spec: EvalSpec,
    reasoning: Reasoning,
    checkpoint_dir: Path,
    group: str,
    parquet: Path,
    out_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Evaluate one split and report what happened, success or not."""
    cmd = [
        "bash", str(project_root / "entrypoints" / "eval" / "eval.sh"),
        f"data.inference_file={parquet}",
        f"data.validation_axis={VALIDATION_AXIS[spec.tier]}",
        f"model.path={checkpoint_dir}",
        f"output.dir={out_dir}",
        f"runtime.num_trials={spec.num_trials}",
    ]
    if spec.max_samples is not None:
        cmd.append(f"data.max_samples={spec.max_samples}")
    for key, value in reasoning.overrides().items():
        rendered = _hydra_bool(value) if isinstance(value, bool) else value
        cmd.append(f"+data.env_config_overrides.{key}={rendered}")
    for key, value in reasoning.assertions().items():
        rendered = _hydra_bool(value) if isinstance(value, bool) else value
        cmd.append(f"+data.assert_env_config.{key}={rendered}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # The raw evaluator log is full of this host's absolute paths, so it stays
    # outside the bundle. Failures are still diagnosable: the manifest keeps a
    # tail, and the log itself remains on the machine that produced it.
    log_dir = out_dir.parents[1].parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{group}__{parquet.stem}.log"
    print(f"  evaluating {group}/{parquet.stem}", flush=True)
    with log_path.open("w") as log:
        result = subprocess.run(
            cmd, cwd=str(project_root), stdout=log, stderr=subprocess.STDOUT, check=False
        )

    record: dict[str, Any] = {
        "group": group,
        "split": parquet.stem,
        "suite_file": f"{group}/{parquet.name}",
        "suite_sha256": _sha256(parquet),
        "status": "ok" if result.returncode == 0 else "failed",
        "exit_code": result.returncode,
    }
    if result.returncode != 0:
        tail = log_path.read_text().splitlines()[-15:]
        record["log_tail"] = [
            line for line in tail if not find_identity_leaks(line)
        ] or ["(log tail withheld: every line carried a host path)"]
        return record

    metrics_path = out_dir / "trial_1" / "metrics.json"
    if not metrics_path.is_file():
        candidates = sorted(out_dir.rglob("metrics.json"))
        metrics_path = candidates[0] if candidates else None
    if metrics_path is not None and metrics_path.is_file():
        record["metrics"] = json.loads(metrics_path.read_text())
    else:
        record["status"] = "failed"
        record["detail"] = "evaluation exited 0 but wrote no metrics.json"

    env_config_path = out_dir / "eval_env_config.json"
    if env_config_path.is_file():
        effective = json.loads(env_config_path.read_text())
        # The rollout records where the weights were read from, which on this
        # machine is a local absolute path. That is useful while debugging and
        # forbidden in a shared artifact, so the bundle keeps the identity the
        # weights actually have: the repository and revision they came from.
        effective["model_path"] = f"{spec.checkpoint_repo}@{spec.revision}"
        env_config_path.write_text(json.dumps(effective, indent=2) + "\n")
        record["effective_env_config"] = effective

    # Per-scenario records embed the environment's resource locations. Those are
    # rebuilt from the consumer's checkout anyway, so redacting them costs no
    # information and keeps the project name out of a shared artifact.
    for json_path in sorted(out_dir.rglob("*.json")):
        if json_path.name != "eval_env_config.json":
            redact_json_file(json_path)
    return record


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #

def build_manifest(
    spec: EvalSpec,
    reasoning: Reasoning,
    checkpoint_manifest: dict[str, Any],
    records: list[dict[str, Any]],
    suite_revision: str | None,
) -> dict[str, Any]:
    ok = [r for r in records if r["status"] == "ok"]
    return {
        "manifest_version": "1",
        "kind": "e5-checkpoint-evaluation",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": {
            "repo": spec.checkpoint_repo,
            "revision": spec.revision,
            "cell": (checkpoint_manifest.get("experiment") or {}).get("cell"),
            "run_name": (checkpoint_manifest.get("experiment") or {}).get("run_name"),
            "artifact_id": checkpoint_manifest.get("artifact_id"),
            "training": checkpoint_manifest.get("training"),
            "provenance": checkpoint_manifest.get("provenance"),
        },
        "suite": {
            "repo": spec.suite_repo,
            "revision": suite_revision,
            "groups": list(spec.suite_groups),
        },
        "protocol": {
            "tier": spec.tier,
            "validation_axis": VALIDATION_AXIS[spec.tier],
            "reasoning_source": reasoning.source,
            "env_config_overrides": reasoning.overrides(),
            "asserted": reasoning.assertions(),
            "num_trials": spec.num_trials,
            "max_samples": spec.max_samples,
            "selection_rule": "terminal or last-valid checkpoint, never best-validation",
        },
        "results": records,
        "summary": {
            "splits_total": len(records),
            "splits_ok": len(ok),
            "splits_failed": len(records) - len(ok),
        },
        "versions": _versions(),
    }


def run_evaluation(
    spec: EvalSpec,
    work_dir: Path,
    project_root: Path,
    *,
    override_require_think: bool | None = None,
    override_think_tag: str | None = None,
    upload: bool = True,
    keep_work_dir: bool = False,
    token: str | None = None,
) -> dict[str, Any]:
    _guard_text("results repository id", spec.results_repo, spec.forbid_terms)

    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"fetching {spec.checkpoint_repo}@{spec.revision}", flush=True)
    checkpoint_dir = fetch_checkpoint(spec, work_dir, token)
    checkpoint_manifest = read_checkpoint_manifest(checkpoint_dir)

    reasoning = resolve_reasoning(
        checkpoint_manifest, override_require_think, override_think_tag
    )
    print(
        f"reasoning: require_think={reasoning.require_think} "
        f"think_tag={reasoning.think_tag} (from {reasoning.source})",
        flush=True,
    )

    print(f"fetching suites {list(spec.suite_groups)}", flush=True)
    suites_dir = fetch_suites(spec, work_dir, token)
    splits = discover_splits(suites_dir, spec.suite_groups)
    print(f"{len(splits)} splits to evaluate", flush=True)

    bundle = work_dir / "bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    records = []
    for group, parquet in splits:
        out_dir = bundle / group / parquet.stem
        records.append(
            evaluate_split(
                spec, reasoning, checkpoint_dir, group, parquet, out_dir, project_root
            )
        )

    suite_revision = None
    if spec.suite_dir is None:
        try:
            from huggingface_hub import HfApi

            suite_revision = HfApi(token=token).dataset_info(spec.suite_repo).sha
        except Exception:  # noqa: BLE001 - a missing revision is recorded as null
            pass

    manifest = build_manifest(spec, reasoning, checkpoint_manifest, records, suite_revision)
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    _guard_text("eval_manifest.json", manifest_text, spec.forbid_terms)
    (bundle / "eval_manifest.json").write_text(manifest_text)

    failed = manifest["summary"]["splits_failed"]
    print(
        f"evaluated {manifest['summary']['splits_ok']}/{manifest['summary']['splits_total']} "
        f"splits" + (f", {failed} failed" if failed else ""),
        flush=True,
    )

    guard_folder(bundle, spec.forbid_terms)

    result: dict[str, Any] = {"local_dir": str(bundle), "manifest": manifest, "uploaded": False}
    if upload:
        # Addressed by path, not by revision. One results repository holds every
        # run in a tier, and two runs in a tier reach the same step, so a branch
        # named for the step would let the second arm overwrite the first with
        # no sign that anything was lost.
        from agrimanager.checkpoint.upload_results import upload_bundle

        url = upload_bundle(bundle, spec.results_repo, token, spec.forbid_terms)
        result.update(uploaded=True, url=url)
        print(f"uploaded (private): {url}", flush=True)
        if not keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
            result["local_dir"] = None
    else:
        print(f"upload skipped; bundle left at {bundle}", flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m agrimanager.checkpoint.evaluate",
        description="Evaluate a published checkpoint on the frozen E4 suites (E5).",
    )
    parser.add_argument("--checkpoint-repo", required=True)
    parser.add_argument("--revision", required=True, help="checkpoint revision, e.g. step-800")
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="evaluate weights already on disk instead of downloading "
                             "--checkpoint-repo. For the untrained base model, run as a "
                             "control. --checkpoint-repo/--revision still supply the "
                             "identity recorded in the bundle.")
    parser.add_argument("--suite-repo", required=True)
    parser.add_argument("--suite-dir", type=Path,
                        help="evaluate audited suites already on disk instead of "
                             "downloading --suite-repo. --suite-repo still supplies "
                             "the protocol identity recorded in the bundle.")
    parser.add_argument("--results-repo", required=True)
    parser.add_argument("--tier", required=True, choices=sorted(SUITE_GROUPS))
    parser.add_argument("--suite-group", action="append", default=[],
                        help="override the tier's default groups; repeatable")
    parser.add_argument("--require-think", choices=["True", "False"],
                        help="only for checkpoints whose manifest predates reasoning recording")
    parser.add_argument("--think-tag")
    parser.add_argument("--max-samples", type=int, help="prefix size, for pipeline checks")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--forbid", action="append", default=[])
    parser.add_argument("--work-dir", type=Path, default=Path("./.e5_work"))
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--keep-work-dir", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    spec = EvalSpec(
        checkpoint_repo=args.checkpoint_repo,
        revision=args.revision,
        suite_repo=args.suite_repo,
        tier=args.tier,
        results_repo=args.results_repo,
        suite_groups=tuple(args.suite_group) or SUITE_GROUPS[args.tier],
        max_samples=args.max_samples,
        num_trials=args.num_trials,
        forbid_terms=collect_forbid_terms(args.forbid),
        checkpoint_dir=args.checkpoint_dir,
        suite_dir=args.suite_dir,
    )
    try:
        result = run_evaluation(
            spec,
            work_dir=args.work_dir,
            project_root=args.project_root,
            override_require_think=(
                None if args.require_think is None else args.require_think == "True"
            ),
            override_think_tag=args.think_tag,
            upload=not args.no_upload,
            keep_work_dir=args.keep_work_dir,
            token=os.environ.get("HF_TOKEN"),
        )
    except (EvaluationError, PublishError) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0 if result["manifest"]["summary"]["splits_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
