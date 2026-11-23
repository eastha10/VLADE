"""Generate the two config-only LIBERO+ plan evaluation notebooks."""
from pathlib import Path
import json
import sys
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vlade.utils.colab_notebooks import bootstrap, cell


ALIGNMENT = """
    # 1. T1 플랜과 액션의 일치도

    T1을 LIBERO+ Spatial에서 실행합니다. 같은 관측에서 생성한 플랜과 액션 청크를 저장하고,
    실제로 실행한 동작을 두 카메라 GIF와 위치·그리퍼 상태로 확인합니다.
    이 평가는 T1의 두 출력이 서로 맞는지를 측정합니다.

    1. BF16 CUDA GPU 런타임을 선택하고 Colab Secrets에 `HF_TOKEN`을 등록하세요.
    2. 소스 셀의 `SOURCE_REF`에 이 평가 코드가 포함된 원격 Git 커밋을 넣으세요.
    3. 설정의 `ACTION`을 `plan → setup → run` 순서로 바꾸며 설정 셀과 실행 셀을 실행하세요.
       먼저 `TRIALS=1`, `MAX_STEPS=16`으로 파일럿을 하고, 본 평가에는 새 `RUN_NAME`을 사용하세요.
    4. `review.csv`를 복사해 `review-labeled.csv`로 저장하고 각 GIF와 상태를 확인해 아래 기준으로 채점하세요.

    | 항목 | 판정 기준 |
    |---|---|
    | `target_motion` | 플랜이 지정한 물체를 향해 접근하는지, 또는 지정된 방향으로 움직이는지 확인합니다. |
    | `gripper` | 지시된 열기·닫기가 실제 그리퍼 상태에 나타나는지 확인합니다. |
    | `order` | 플랜이 순서를 명시한 경우 관측된 동작이 그 순서를 따르는지 확인합니다. |

    각 항목에 `match`, `mismatch`, `unclear`, `na`를 입력하세요.
    관측이 가려졌거나 청크 종료 전에 판정할 수 없으면 `unclear`, 그 항목을 플랜이 언급하지 않았으면 `na`입니다.
    빈 셀은 미채점으로 집계합니다. 정해진 간격으로 앞 구간부터 태스크별 최대 8개를 저장하며,
    일찍 종료된 태스크는 더 적게 저장됩니다. 열고 닫는 방향은 GIF와 실제 `gripper_qpos`를 보고 판정하세요.
"""

STUDENT = """
    # 2. S1·S2 플랜 효과 비교

    S1은 Action-level Distillation이고 S2는 Action+CoT Distillation입니다.
    S2에서는 정상 플랜, 같은 태스크의 과거 플랜, 플랜 생략을 평가합니다.
    S1은 Action-only 기준선으로 평가합니다.
    모든 조건에서 같은 LIBERO+ 태스크·초기 상태·시드를 사용합니다.

    1. BF16 CUDA GPU 런타임을 선택하고 Colab Secrets에 `HF_TOKEN`을 등록하세요.
    2. `SOURCE_REF`에 이 평가 코드가 포함된 원격 Git 커밋을 넣으세요.
    3. S1과 S2 체크포인트를 연결하세요. 동일한 초기 S0, 데이터, T1, 학습 설정, optimizer step을 사용해야 합니다.
       이미 등록한 HF asset 키는 설정 셀에서 `COT_CHECKPOINT`, `ACTION_CHECKPOINT`로 지정하세요.
       훈련 완료 로그에 출력된 S1과 S2의 `CHECKPOINT_PINS`를 하나의 딕셔너리로 합쳐 넣으세요.
       직접 입력하려면 키별로 `repo_id`, 불변 `revision` SHA, `prefix`를 지정하세요.
       `prefix` 형식은 `runs/<그룹>/S1/checkpoints/step-<7자리 번호>` 또는
       `runs/<그룹>/S2/checkpoints/step-<7자리 번호>`입니다.
    4. `ACTION`을 `plan → setup → run` 순서로 바꾸며 설정 셀과 실행 셀을 실행하세요.
       먼저 `TRIALS=1`, `MAX_STEPS=32`로 파일럿을 하고, 본 평가에는 새 `RUN_NAME`을 사용하세요.

    - **현재 vs 과거 플랜**은 현재 관측에 맞는 내용의 효과를 봅니다. 과거 조건은 각 에피소드에서 두 쿼리 전의 플랜을 사용합니다.
      처음 두 쿼리는 정상 플랜으로 시작합니다. **실제로 다른 플랜이 들어간 횟수**도 확인하세요.
    - **현재 vs 플랜 생략**은 같은 체크포인트가 입력 플랜에 얼마나 의존하는지 봅니다.
    - **현재 vs Action-only KD**는 최종 작업 성공률을 같은 초기 상태에서 비교합니다.

    `paired_success.csv`의 성공률 차이와 두 조건 중 한쪽만 성공한 횟수를 확인하세요.
    `action_probes.csv`는 정상 rollout 관측에서 **S0의 상태를 고정한 채** 플랜만 바꾼 정규화 액션 L1 차이입니다.
    현재 플랜과 과거 플랜이 모두 유효하고 내용이 다른 쿼리만 이 파일에 포함합니다.
    플랜을 바꿔 동작이 달라진 것과 성공률이 바뀐 것은 각각 해석하세요.
    생성된 플랜이 비어 있으면 해당 쿼리에서 플랜 없이 실행하고 그 횟수를 별도로 남깁니다.
"""


def generate(name: str, *, alignment: bool) -> None:
    config = "t1_plan_alignment" if alignment else "s1_s2_plan_effect"
    checkpoint_settings = "" if alignment else """
        COT_CHECKPOINT = "hfasset://S2"
        ACTION_CHECKPOINT = "hfasset://S1"
        CHECKPOINT_PINS = {}
        # 예: {"S2": {"repo_id": "VLADE-konkuk/VLADE-checkpoints",
        #              "revision": "<40자리 HF commit>", "prefix": "runs/<그룹>/S2/checkpoints/step-0000100"}}
        OVERRIDES.update(cot_checkpoint=COT_CHECKPOINT, action_checkpoint=ACTION_CHECKPOINT,
                         checkpoint_pins=CHECKPOINT_PINS)
    """
    cells = [
        cell("markdown", ALIGNMENT if alignment else STUDENT, "guide"),
        cell("code", bootstrap(source_ref="refs/heads/main"), "source"),
        cell("code", dedent(f"""
            WORKSPACE = "/content/vlade_distillation"
            RUN_NAME = "{config.replace('_', '-')}-seed42"
            ACTION = "plan"  #@param ["plan", "setup", "run"]
            TRIALS = {5 if alignment else 10}
            MAX_STEPS = 600
            OVERRIDES = dict(trials=TRIALS, max_steps=MAX_STEPS)
        """) + dedent(checkpoint_settings), "config"),
        cell("code", f"""
            from vlade.research.plan_evaluation import job

            out = job(CODE / 'configs/experiments/{config}.json', workspace=WORKSPACE,
                      run_name=RUN_NAME, action=ACTION, overrides=OVERRIDES, colab=True)
        """, "run"),
        cell("code", """
            import json
            from IPython.display import JSON, display
            from vlade.research.plan_evaluation import output_path

            out = output_path(Path(WORKSPACE), RUN_NAME)
            if (out / "results.json").exists():
                display(JSON(json.loads((out / "results.json").read_text(encoding="utf-8"))))
            else:
                print("결과 파일은 run 단계가 완료되면 생성됩니다.")
        """, "results"),
    ]
    if alignment:
        cells.extend([
            cell("code", """
                from IPython.display import Image, display

                CLIP_ID = ""  # review.csv에서 확인할 chunk_id 입력
                if CLIP_ID:
                    display(Image(filename=str(out / "clips" / (CLIP_ID + ".gif"))))
            """, "review-clip"),
            cell("code", """
                from vlade.evaluation.plan_rollout import summarize_alignment

                scored = out / "review-labeled.csv"
                if scored.exists():
                    display(JSON(summarize_alignment(scored)))
            """, "review-summary"),
        ])
    notebook = {"cells": cells, "metadata": {
        "accelerator": "GPU", "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}
    (ROOT / "experiments" / f"{name}.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    generate("colab_t1_plan_alignment", alignment=True)
    generate("colab_s1_s2_plan_effect", alignment=False)
