"""Generate the single Colab client for all config-driven research workflows."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_NAME = "colab_vlade.ipynb"
PROFILES = {
    "prepare_inputs": "HF 데이터·가중치 준비",
    "prepare_one_episode": "Canonical Train 한 Episode 준비",
    "fullframes": "전체 프레임 CPU 처리·공개 HF 게시",
    "ava_one_step": "AVA K1 한 step·별도 재로딩 검사",
    "ava_fft": "AVA K4 FFT",
    "ava_suite": "AVA K4 Suite-balanced pilot4/main",
    "ava_paper": "AVA 논문 LoRA·TBPTT pilot/main",
    "oft_fft": "OpenVLA-OFT FFT",
    "select_teacher": "검증 보고서 기반 T1 선택",
    "cache_teacher": "고정 T1 순차 cache 생성·게시",
    "qwen_task": "Qwen Task-only·파일럿",
    "qwen_kd": "Qwen Action KD·파일럿",
    "evaluate_ava": "AVA LIBERO+ paired rollout",
    "evaluate_oft": "OFT LIBERO+ paired rollout",
    "evaluate_qwen": "Qwen LIBERO+ paired rollout",
    "evaluate_libero_ava": "AVA 원본 LIBERO 평가",
    "evaluate_libero_oft": "OFT 원본 LIBERO 평가",
    "inference_ava": "실제 AVA 가중치 단일 추론",
}


def cell(kind: str, text: str, name: str) -> dict:
    """Build an output-free notebook cell and check Python syntax."""
    source = textwrap.dedent(text).strip() + "\n"
    value = {
        "cell_type": kind,
        "id": name,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        compile(source, name, "exec")
        value.update(execution_count=None, outputs=[])
    return value


def bootstrap(source_ref: str = "main", **kwargs) -> str:
    """Load one immutable source checkout without replacing running code."""
    return textwrap.dedent("""
        from pathlib import Path
        import os, re, subprocess, sys, time
        os.environ.setdefault('VLADE_SESSION_STARTED_UNIX', str(time.time()))
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        sys.dont_write_bytecode = True
        SOURCE_REF = __REF__  # 재개는 최초 실행의 SOURCE_COMMIT을 지정합니다.
        REPOSITORY = 'https://github.com/eastha10/VLADE.git'
        SOURCE_COMMIT = SOURCE_REF if re.fullmatch(r'[0-9a-f]{40}', SOURCE_REF) else subprocess.check_output(['git','ls-remote',REPOSITORY,SOURCE_REF], text=True).split()[0]
        if not re.fullmatch(r'[0-9a-f]{40}', SOURCE_COMMIT):
            raise ValueError('불변 Git commit이 필요합니다.')
        CODE = Path('/content/vlade_sources') / SOURCE_COMMIT
        if any((name == 'vlade' or name.startswith('vlade.')) and getattr(module,'__file__',None) and not Path(module.__file__).resolve().is_relative_to(CODE) for name,module in list(sys.modules.items())):
            raise RuntimeError('다른 소스가 로드되어 있습니다. Colab 런타임을 다시 시작하십시오.')
        if not CODE.exists():
            subprocess.run(['git','clone','--no-checkout','--filter=blob:none',REPOSITORY,str(CODE)],check=True)
            subprocess.run(['git','-C',str(CODE),'fetch','--depth','1','origin',SOURCE_COMMIT],check=True)
            subprocess.run(['git','-C',str(CODE),'checkout','--detach',SOURCE_COMMIT],check=True)
        sys.path[:0] = [str(CODE/'src'),str(CODE)]
        from vlade.utils.source_identity import source_identity
        assert source_identity(CODE,require_clean=True)['git_commit']==SOURCE_COMMIT
        print('SOURCE_COMMIT:',SOURCE_COMMIT)
    """).replace("__REF__", repr(source_ref)).strip()


def write_unified_notebook(path: Path) -> Path:
    """Write a selector-based client; model and data logic stay in Python modules."""
    catalog = "\n".join(f"| {key} | {title} |" for key, title in PROFILES.items())
    choices = json.dumps(list(PROFILES), ensure_ascii=False)
    controls = f"""
        import copy, datetime, uuid
        from vlade.research.client import job

        PROFILE = "prepare_inputs"  #@param {choices}
        WORKSPACE = "/content/vlade_work"  #@param {{type:"string"}}
        EXPERIMENT_ID = globals().get("EXPERIMENT_ID") or (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-" + uuid.uuid4().hex[:8]
        )
        RUN_GROUP_OVERRIDE = ""  #@param {{type:"string"}}
        OVERRIDES = {{}}  # 예: {{"train.max_steps": 1000, "train.warmup_steps": 100}}
        RUN_SETUP = False  #@param {{type:"boolean"}}
        RUN_EXPERIMENT = False  #@param {{type:"boolean"}}
    """
    plan = """
        AVAILABLE_PROFILES = __PROFILES__
        if PROFILE not in AVAILABLE_PROFILES:
            raise ValueError("목록에서 PROFILE을 선택하십시오.")
        CONFIG = CODE / "configs/research" / (PROFILE + ".json")
        RUN_GROUP = RUN_GROUP_OVERRIDE or (EXPERIMENT_ID + "-" + PROFILE)
        EFFECTIVE_OVERRIDES = {**OVERRIDES, "run_group": RUN_GROUP}
        # setup/run 셀은 마지막으로 확인한 계획만 실행합니다.
        print("PROFILE:", PROFILE, "| RUN_GROUP:", RUN_GROUP)
        job(CONFIG, action="plan", workspace=WORKSPACE, overrides=EFFECTIVE_OVERRIDES)
        _PLANNED = (str(CONFIG), WORKSPACE, copy.deepcopy(EFFECTIVE_OVERRIDES))
        _PLANNED_SELECTION = (PROFILE, EXPERIMENT_ID, RUN_GROUP_OVERRIDE, copy.deepcopy(OVERRIDES))
    """.replace("__PROFILES__", repr(list(PROFILES)))
    guard = """
        if (PROFILE, EXPERIMENT_ID, RUN_GROUP_OVERRIDE, OVERRIDES) != _PLANNED_SELECTION or WORKSPACE != _PLANNED[1]:
            raise RuntimeError("설정을 바꿨습니다. 계획 확인 셀부터 다시 실행하십시오.")
    """
    def action_source(action: str, flag: str) -> str:
        body = textwrap.indent(textwrap.dedent(guard).strip(), "    ")
        return (
            f"if {flag}:\n" + body
            + f'\n    job(_PLANNED[0], action="{action}", workspace=_PLANNED[1], overrides=_PLANNED[2])'
            + f'\nelse:\n    print("{action} 대기: {flag}=False")'
        )
    cells = [
        cell("markdown", """
            # VLADE · 통합 연구 실행
            데이터 준비·AVA/OFT 학습·Student 증류·Teacher cache·평가를 이 노트북에서 선택합니다.
            학습·데이터 로더·저장/재개는 프로젝트 Python 모듈을 그대로 호출합니다.

            순서: 소스 준비 → HF 인증 → 작업 선택·설정 → 계획 확인 → 환경 설치 → 실행.
            설치와 실행 스위치는 기본 False입니다. 모든 셀을 실행해도 실험은 자동 시작하지 않습니다.
            작업을 바꿀 때는 설정 셀의 PROFILE을 선택하고 계획부터 다시 확인하십시오.
            GPU 훈련·전체 프레임 처리·공개 게시 작업은 실행 전에 해당 계획을 확인하십시오.
        """, "purpose"),
        cell("code", bootstrap(), "source"),
        cell("code", """
            from google.colab import userdata
            os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
            print("HF_TOKEN을 연결했습니다. 토큰 값은 출력하지 않습니다.")
        """, "authentication"),
        cell("markdown", "## 작업 목록\n\n| PROFILE | 기능 |\n|---|---|\n" + catalog, "profiles"),
        cell("markdown", """
            ## 작업 선택과 설정
            PROFILE 하나만 실행합니다. 설정은 configs/research/<PROFILE>.json에서 읽으며
            변경할 항목만 OVERRIDES에 적습니다. 모델별 환경은 setup이 각각 준비합니다.

            EXPERIMENT_ID는 셀을 다시 실행해도 유지됩니다. 새 실험은 이 값을 새 이름으로 바꾸십시오.
            기본 run group은 EXPERIMENT_ID와 PROFILE을 합쳐 작업별로 분리합니다.
            같은 Suite pilot4/main, 논문 pilot/main은 EXPERIMENT_ID를 유지합니다.
            재개는 RUN_GROUP_OVERRIDE에 최초 실행의 정확한 run group을 넣고 원래 설정을 복원합니다.
            같은 run group/output에 여러 프로세스를 동시에 실행하지 마십시오.
        """, "selection-help"),
        cell("code", controls, "config"),
        cell("code", plan, "plan"),
        cell("markdown", """
            ## 환경 설치
            RUN_SETUP=True로 바꾸고 설정·계획 셀을 다시 실행한 뒤 아래 셀을 실행합니다.
            이미 준비된 같은 환경은 재사용합니다. 세션 예산은 설치부터 계산합니다.
        """, "setup-help"),
        cell("code", action_source("setup", "RUN_SETUP"), "setup"),
        cell("markdown", """
            ## 선택한 작업 실행
            RUN_EXPERIMENT=True로 바꾸고 설정·계획 셀을 다시 실행한 뒤 아래 셀을 실행합니다.
            선택한 작업만 실행되며, 다른 프로필의 데이터 준비·훈련·게시를 함께 시작하지 않습니다.
        """, "run-help"),
        cell("code", action_source("run", "RUN_EXPERIMENT"), "run"),
        cell("markdown", """
            ## AVA K4를 4,000스텝 학습하는 방법

            ### Suite-balanced: 파일럿 후 본 학습

            처음에는 "작업 선택과 설정" 셀의 해당 값을 아래처럼 바꿉니다.
            소스·HF 인증 이후 설정 → 계획 확인 → 환경 설치 → 실행 셀 순서로 실행합니다.

            ```python
            PROFILE = "ava_suite"
            OVERRIDES = {}  # 기본 pilot4: optimizer 4 step
            RUN_SETUP = True
            RUN_EXPERIMENT = True
            ```

            pilot4가 완료되면 같은 SOURCE_COMMIT·WORKSPACE·EXPERIMENT_ID·RUN_GROUP_OVERRIDE를
            유지한 채, 같은 설정 셀의 값을 아래처럼 바꿉니다.
            설정 → 계획 확인 → 실행 셀 순서로 실행합니다. 설치가 끝났으므로 setup은 생략합니다.

            ```python
            PROFILE = "ava_suite"
            OVERRIDES = {
                "settings.phase": "main",
                "train.max_steps": 4000,
                "train.warmup_steps": 400,
                "train.save_steps": 250,
                "train.eval_steps": 100,
            }
            RUN_SETUP = False
            RUN_EXPERIMENT = True
            ```

            warmup 400·저장 주기 250·검증 주기 100은 이 실행 예시의 값입니다.
            파일럿 4 step과 본 학습 4,000 step은 별도 실행입니다.
            main은 초기 T0에서 시작하므로 pilot checkpoint를 resume에 넣지 않습니다.
            기존 1,000-step main을 시작한 동일 run group의 목표만 4,000으로 변경하지 마십시오.
            그 경우 새 EXPERIMENT_ID로 파일럿부터 실행합니다.

            ### 일반 AVA FFT를 사용하는 경우

            새 실험 ID에서 아래 값을 설정하고 계획 → 환경 설치(필요할 때) → 실행합니다.
            일반 FFT는 Suite의 pilot 보고서를 요구하지 않으므로 최소 GPU 파일럿을 먼저 별도 실행으로 확인하십시오.

            ```python
            PROFILE = "ava_fft"
            OVERRIDES = {
                "train.max_steps": 4000,
                "train.warmup_steps": 400,
                "train.save_steps": 250,
                "train.eval_steps": 100,
            }
            RUN_SETUP = False  # 아직 환경이 없으면 True
            RUN_EXPERIMENT = True
            ```

            K4·batch 1·누적 8은 기존 기본값을 사용합니다. 4,000은 optimizer step 수이며,
            이 설정에서 총 32,000개 시퀀스·128,000개 query 예측에 해당합니다.
            복원 추출이므로 고유 데이터 수나 epoch 수를 뜻하지 않습니다.
            기본 세션 상한은 설치 포함 8시간입니다. 4,000 step을 그 시간 안에 완료한다는 보장은 없습니다.
            예산 종료 시 final로 기록하지 않으며, 같은 4,000-step 목표·원래 설정·run group으로
            검증된 Hub checkpoint에서 재개합니다. 기존 settings 비교·Suite pilot 복구 제약도 유지합니다.
        """, "ava-4000-steps"),
        cell("markdown", """
            ## 작업별 실행 순서
            - 일반 AVA/OFT: prepare_inputs → ava_fft 또는 oft_fft.
            - 최소 AVA 진단: prepare_one_episode → ava_one_step. 서로 같은 WORKSPACE를 사용합니다.
            - Suite-balanced: ava_suite 기본 pilot4(4 step) 완료 → 같은 EXPERIMENT_ID에서
              OVERRIDES={"settings.phase": "main", "train.max_steps": 1000,
              "train.warmup_steps": 100, "train.save_steps": 250}.
              1,000 step은 예시입니다. pilot checkpoint를 main의 resume에 넣지 않습니다.
            - 논문 LoRA: ava_paper에서 OVERRIDES={"pilot": True}를 먼저 실행하고
              같은 실험에서 {"pilot": False}로 본 학습합니다. 기본 목표는 100,000 step입니다.
            - Teacher 증류: 검증·게시한 고정 T1과 평가 보고서 준비 → select_teacher →
              cache_teacher → qwen_kd. select_teacher의 arguments는 실제 보고서·선정 이유로 지정합니다.
              없는 HF 키나 pilot cache를 전체 Teacher cache로 대체하지 않습니다.
            - Qwen Task-only/KD는 기존 query-head trainer이며 별도 QwenVLA 모델 학습과 구분합니다.
              기존 1-query 재로딩·overfit 조건과 pilot.only 설정을 유지합니다.
            - 평가: evaluate_ava/oft/qwen은 LIBERO+, evaluate_libero_ava/oft는 원본 LIBERO입니다.
              evaluation.rollouts_per_task와 checkpoints를 설정합니다.
            - inference_ava는 단일 액션 추론이며 rollout 성공률 평가가 아닙니다.
            - fullframes는 별도 CPU 전체 프레임 내보내기와 공개 HF 게시입니다.
              이미 완료된 전체 프레임 데이터는 재처리할 필요 없이 고정 HF 키를 사용합니다.

            출력은 WORKSPACE/runs/<run group>/ 아래에 생성됩니다. Suite는 pilot4/main 및 Hub
            -pilot4/-main 구분을 유지합니다. train.max_steps는 optimizer step 수이며 K4 길이와 다릅니다.
            같은 source commit·데이터·정규화·학습 설정으로만 재개합니다.
            기존 CLI의 settings 파일 비교와 Suite pilot 증거 검증도 그대로 적용합니다.
        """, "workflow-help"),
        cell("markdown", """
            ## 재개·명령줄·상세 안내
            Hub의 불변 revision으로 다운로드·무결성 검사한 checkpoint만 resume에 지정합니다.
            원래 run group·목표 step·K·sampling·detach·입력 identity를 유지하십시오.
            resume 추가 시 기존 settings 파일과 충돌하는 경우 설정 파일을 삭제하여 우회하지 않습니다.
            CLI도 같은 구현을 사용합니다. 저장소 루트에서 다음처럼 실행합니다.

            python -m scripts.research.run --config configs/research/ava_suite.json --workspace /content/vlade_work --action plan

            자세한 안내는 checkout의 docs/environment/research_runs.md,
            ava_sequence_training.md, hugging_face_checkpoints.md를 확인하십시오.
            Drive의 기존 파일과 실행 중인 Colab은 자동 갱신하지 않습니다.
            실제 GPU 학습·수렴·rollout 성능은 별도 검증 대상입니다.
        """, "reference-help"),
    ]
    value = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": path.name, "provenance": []},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def generate() -> list[Path]:
    """Regenerate only the maintained unified notebook and source manifest."""
    path = write_unified_notebook(ROOT / "experiments" / NOTEBOOK_NAME)
    manifest = ROOT / "experiments/manifests/colab_source.json"
    value = {
        "schema_version": 2,
        "distribution": "git_commit",
        "repository": "https://github.com/eastha10/VLADE.git",
        "default_ref": "main",
        "runtime_source": "project_owned",
        "submodules_required": False,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return [path, manifest]


def main() -> None:
    """Regenerate the unified Colab notebook without running experiments."""
    for path in generate():
        print(path.relative_to(ROOT))
