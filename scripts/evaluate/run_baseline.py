"""VLADE에 등록된 외부 baseline의 LIBERO 평가를 실행합니다."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = REPO_ROOT / "experiments" / "baselines"
RESERVED_ARGUMENTS = {"pretrained_checkpoint", "task_suite_name", "num_trials_per_task", "seed"}


class BaselineConfigurationError(ValueError):
    """Baseline 설정이 실행 계약을 충족하지 않을 때 발생합니다."""


@dataclass(frozen=True)
class BaselineInvocation:
    """검증이 끝난 baseline 실행 정보입니다."""

    command: tuple[str, ...]
    working_directory: Path
    environment: Mapping[str, str]
    environment_name: str
    checkpoint_source: str
    local_checkpoint: Path | None
    log_directory: Path


def available_baselines() -> tuple[str, ...]:
    """LIBERO 설정을 가진 baseline 식별자를 반환합니다."""

    if not BASELINE_ROOT.is_dir():
        return ()
    return tuple(sorted(path.parent.name for path in BASELINE_ROOT.glob("*/libero.json")))


def load_baseline_config(baseline_id: str) -> dict[str, Any]:
    """baseline의 LIBERO JSON 설정을 읽고 기본 구조를 검증합니다."""

    config_path = BASELINE_ROOT / baseline_id / "libero.json"
    if not config_path.is_file():
        raise BaselineConfigurationError(f"baseline 설정을 찾을 수 없습니다: {config_path}")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineConfigurationError(f"JSON 문법 오류: {config_path}: {exc}") from exc

    if config.get("schema_version") != 1:
        raise BaselineConfigurationError("지원하는 schema_version은 1입니다.")
    if config.get("baseline_id") != baseline_id:
        raise BaselineConfigurationError("baseline_id가 설정 디렉터리 이름과 일치하지 않습니다.")

    required_sections = ("source", "runtime", "benchmark", "checkpoints", "evaluation")
    missing_sections = [section for section in required_sections if not isinstance(config.get(section), dict)]
    if missing_sections:
        raise BaselineConfigurationError(f"필수 설정 섹션이 없거나 객체가 아닙니다: {missing_sections}")

    _validate_repository_paths(config)
    _validate_benchmark_contract(config)
    return config


def _resolve_repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise BaselineConfigurationError(f"저장소 밖을 가리키는 경로는 사용할 수 없습니다: {value}") from exc
    return path


def _validate_repository_paths(config: Mapping[str, Any]) -> None:
    source_path = _resolve_repo_path(config["source"]["path"])
    working_directory = _resolve_repo_path(config["evaluation"]["working_directory"])
    dependency_manifest = _resolve_repo_path(config["runtime"]["dependency_manifest"])

    if not source_path.is_dir():
        raise BaselineConfigurationError(f"외부 원본 디렉터리가 없습니다: {source_path}")
    if working_directory != source_path:
        raise BaselineConfigurationError("evaluation.working_directory는 source.path와 같아야 합니다.")
    if not dependency_manifest.is_file():
        raise BaselineConfigurationError(f"의존성 명세가 없습니다: {dependency_manifest}")

    entrypoint = (working_directory / config["evaluation"]["entrypoint"]).resolve()
    try:
        entrypoint.relative_to(working_directory)
    except ValueError as exc:
        raise BaselineConfigurationError("평가 진입점은 외부 원본 디렉터리 안에 있어야 합니다.") from exc
    if not entrypoint.is_file():
        raise BaselineConfigurationError(f"평가 진입점이 없습니다: {entrypoint}")


def _validate_benchmark_contract(config: Mapping[str, Any]) -> None:
    benchmark = config["benchmark"]
    task_suites = benchmark.get("task_suites")
    if benchmark.get("name") != "libero" or not isinstance(task_suites, list) or not task_suites:
        raise BaselineConfigurationError("현재 실행기는 하나 이상의 LIBERO task suite가 필요합니다.")
    if benchmark.get("default_task_suite") not in task_suites:
        raise BaselineConfigurationError("default_task_suite가 task_suites에 포함되어야 합니다.")

    checkpoints = config["checkpoints"]
    for task_suite in task_suites:
        checkpoint = checkpoints.get(task_suite)
        if not isinstance(checkpoint, dict):
            raise BaselineConfigurationError(f"checkpoint 설정이 없습니다: {task_suite}")
        if not checkpoint.get("hub_repository") or not checkpoint.get("local_directory"):
            raise BaselineConfigurationError(f"checkpoint 경로가 완전하지 않습니다: {task_suite}")

    arguments = config["evaluation"].get("arguments", {})
    if not isinstance(arguments, dict):
        raise BaselineConfigurationError("evaluation.arguments는 객체여야 합니다.")
    duplicated = RESERVED_ARGUMENTS.intersection(arguments)
    if duplicated:
        raise BaselineConfigurationError(f"실행기가 관리하는 인자를 설정에 중복할 수 없습니다: {sorted(duplicated)}")

    environment = config["evaluation"].get("environment", {})
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in environment.items()
    ):
        raise BaselineConfigurationError("evaluation.environment의 키와 값은 문자열이어야 합니다.")


def _cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        raise BaselineConfigurationError("CLI 인자 값으로 null을 사용할 수 없습니다.")
    return str(value)


def build_invocation(
    config: Mapping[str, Any],
    *,
    task_suite: str | None = None,
    checkpoint_override: str | None = None,
    use_hub_checkpoint: bool = False,
    num_trials_per_task: int | None = None,
    seed: int | None = None,
    python_executable: str | None = None,
) -> BaselineInvocation:
    """설정과 CLI override를 결합해 실행 정보를 만듭니다."""

    benchmark = config["benchmark"]
    selected_suite = task_suite or benchmark["default_task_suite"]
    if selected_suite not in benchmark["task_suites"]:
        raise BaselineConfigurationError(f"지원하지 않는 task suite입니다: {selected_suite}")

    checkpoint_config = config["checkpoints"][selected_suite]
    local_checkpoint: Path | None = None
    if checkpoint_override:
        checkpoint = checkpoint_override
        checkpoint_source = "override"
    elif use_hub_checkpoint:
        checkpoint = checkpoint_config["hub_repository"]
        checkpoint_source = "hugging_face"
    else:
        local_checkpoint = _resolve_repo_path(checkpoint_config["local_directory"])
        checkpoint = str(local_checkpoint)
        checkpoint_source = "local"

    trials = num_trials_per_task if num_trials_per_task is not None else benchmark["num_trials_per_task"]
    selected_seed = seed if seed is not None else benchmark["seed"]
    if trials <= 0:
        raise BaselineConfigurationError("num_trials_per_task는 양수여야 합니다.")

    evaluation = config["evaluation"]
    working_directory = _resolve_repo_path(evaluation["working_directory"])
    entrypoint = (working_directory / evaluation["entrypoint"]).resolve()
    log_directory = _resolve_repo_path(evaluation["arguments"]["local_log_dir"])

    command = [
        python_executable or sys.executable,
        str(entrypoint),
        "--pretrained_checkpoint",
        checkpoint,
        "--task_suite_name",
        selected_suite,
        "--num_trials_per_task",
        str(trials),
        "--seed",
        str(selected_seed),
    ]
    for name, value in evaluation["arguments"].items():
        if name == "local_log_dir":
            value = str(log_directory)
        command.extend((f"--{name}", _cli_value(value)))

    environment = dict(os.environ)
    environment.update(evaluation.get("environment", {}))
    return BaselineInvocation(
        command=tuple(command),
        working_directory=working_directory,
        environment=environment,
        environment_name=config["runtime"]["environment_name"],
        checkpoint_source=checkpoint_source,
        local_checkpoint=local_checkpoint,
        log_directory=log_directory,
    )


def format_invocation(invocation: BaselineInvocation) -> str:
    """사람이 검토할 수 있는 실행 요약을 반환합니다."""

    return "\n".join(
        (
            f"Environment: {invocation.environment_name}",
            f"Working directory: {invocation.working_directory}",
            f"Checkpoint source: {invocation.checkpoint_source}",
            f"Log directory: {invocation.log_directory}",
            f"Command: {shlex.join(invocation.command)}",
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", choices=available_baselines())
    parser.add_argument("--task-suite", help="평가할 LIBERO task suite")
    parser.add_argument("--checkpoint", help="설정의 체크포인트 대신 사용할 로컬 경로 또는 Hub 식별자")
    parser.add_argument(
        "--use-hub-checkpoint",
        action="store_true",
        help="로컬 경로 대신 설정에 기록된 Hugging Face 식별자를 사용합니다.",
    )
    parser.add_argument("--num-trials-per-task", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--python", dest="python_executable", help="baseline 환경의 Python 실행 파일")
    parser.add_argument("--dry-run", action="store_true", help="평가를 시작하지 않고 명령만 출력합니다.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.checkpoint and args.use_hub_checkpoint:
        parser.error("--checkpoint와 --use-hub-checkpoint는 함께 사용할 수 없습니다.")

    try:
        config = load_baseline_config(args.baseline)
        invocation = build_invocation(
            config,
            task_suite=args.task_suite,
            checkpoint_override=args.checkpoint,
            use_hub_checkpoint=args.use_hub_checkpoint,
            num_trials_per_task=args.num_trials_per_task,
            seed=args.seed,
            python_executable=args.python_executable,
        )
    except BaselineConfigurationError as exc:
        parser.error(str(exc))

    print(format_invocation(invocation))
    if args.dry_run:
        return 0

    if invocation.local_checkpoint is not None and not invocation.local_checkpoint.is_dir():
        parser.error(
            "로컬 체크포인트가 없습니다: "
            f"{invocation.local_checkpoint}. 체크포인트를 준비하거나 --use-hub-checkpoint를 명시하십시오."
        )

    invocation.log_directory.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        invocation.command,
        cwd=invocation.working_directory,
        env=invocation.environment,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
