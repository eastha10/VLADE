"""Generate the dedicated HF T1 -> S0 Action KD research notebook."""
from __future__ import annotations

import json
from pathlib import Path

from vlade.utils.colab_notebooks import ROOT, bootstrap, cell


def write_notebook(path: Path) -> Path:
    """Keep authentication, input pins and execution switches explicit and output-free."""
    cells = [
        cell("markdown", """
            # HF T1 → S0 · Action-level Distillation

            T1 가중치를 HF에서 받아 고정하고, 매 query의 추론 Action Chunk로 S0를 학습합니다.
            Teacher cache를 만들 필요가 없습니다. 손실은 노션 **11-06 Tasks**의
            `L = L_BC(S, GT) + λ · L_KD(S, T1)`이며 두 항 모두 padding을 제외한 정규화 L1입니다.

            **순서:** 소스 → HF 인증 → 설정 → 입력 연결 → 계획 → 환경 준비 → 입력 검사 → 파일럿 → 본 학습.
            아래 실행 스위치는 모두 기본 False입니다. 필요한 셀만 True로 바꿔 실행하십시오.
            실제 가중치 다운로드·GPU 학습·VRAM/성공률 검증은 이 노트북 생성 작업에서 수행하지 않았습니다.

            이 노트북과 구현이 포함된 Git commit을 먼저 배포해야 합니다.
            소스 셀의 `SOURCE_REF`에 그 commit SHA 또는 배포한 branch를 지정하고 출력되는 SHA를 기록하십시오.
            미커밋 코드를 정식 실험에 사용하거나 실행 중인 소스를 덮어쓰지 않습니다.
        """, "purpose"),
        cell("code", bootstrap(), "source"),
        cell("markdown", """
            ## 1. HF 연결

            1. HF 계정에 `VLADE-konkuk/VLADE-checkpoints` 접근 권한을 부여합니다.
            2. [HF Access Tokens](https://huggingface.co/settings/tokens)에서 해당 비공개 model repo의
               **읽기 + 쓰기** 권한이 있는 토큰을 준비합니다. 입력은 읽기, KD checkpoint 자동 저장은 쓰기 권한을 사용합니다.
            3. Colab 왼쪽 Secrets에 `HF_TOKEN`으로 저장하고 이 노트북의 접근을 허용합니다.
               토큰을 코드·설정·출력에 붙여 넣지 않습니다.
            4. Colab 이외 환경은 `hf auth login` 또는 외부에서 주입한 `HF_TOKEN`을 사용합니다.

            kernel에 `huggingface_hub`가 없는 경우에만 다음 선택 셀을 실행합니다.
            AVA/Student 모델 의존성은 뒤의 환경 준비 셀이 별도로 설치합니다.
        """, "authentication-help"),
        cell("code", """
            INSTALL_HF_CLIENT = False
            if INSTALL_HF_CLIENT:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub==1.16.1'], check=True)
        """, "client"),
        cell("code", """
            try:
                from google.colab import userdata
            except ImportError:
                pass  # local: hf auth login / externally injected HF_TOKEN
            else:
                os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
            print('HF 인증 설정을 연결했습니다. 토큰 값은 출력하지 않습니다.')
        """, "authentication"),
        cell("markdown", """
            ## 2. T1·S0 가중치 선택

            **이미 asset registry에 등록했다면:** `T1_ASSET`, `S0_ASSET`에 실제 키를 적고
            해당 registry의 불변 commit을 `REGISTRY_REVISION`에 적습니다. `T1_HF`/`S0_HF`는 빈 값으로 둡니다.

            **HF `runs/`에만 있고 asset 키가 없다면:** HF repo의 **Files and versions**에서
            T1/S0 각각의 `runs/<group>/<output>/checkpoints/step-XXXXXXX` 경로를 찾습니다.
            그 파일이 존재하는 **40자리 commit SHA**를 `revision`, 경로를 `prefix`에 적습니다.
            `main`, 웹페이지 URL, 로컬 경로를 넣지 않습니다. T1과 S0의 SHA가 서로 달라도 됩니다.

            입력 연결 셀은 작은 seal·metadata와 원격 LFS hash를 확인하여 런타임 registry에 연결합니다.
            원격 HF 파일을 복제·업로드하거나 정식 lock을 덮어쓰지 않습니다.
            모델 본문 다운로드는 이후 입력 검사/학습 셀에서 수행합니다.

            T1은 선택한 비양자화 AVA FFT checkpoint, S0는 기존 schema-1 QwenStudent의 전체 checkpoint입니다.
            S0의 학습된 action head까지 복원합니다. schema-2 QwenVLA는 다른 구조이므로 거부합니다.
            T1·S0·데이터 split/정규화는 일치해야 합니다. T0나 `qwen_initial`을 대신 넣지 마십시오.
        """, "weights-help"),
        cell("code", """
            import copy, datetime, uuid
            from vlade.research.client import job

            WORKSPACE = '/content/vlade_action_kd'
            RUN_GROUP = globals().get('RUN_GROUP') or (
                'action-kd-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                + '-' + uuid.uuid4().hex[:8])
            REGISTRY_REVISION = None  # registered assets only: immutable registry commit, or committed lock
            T1_ASSET = 'teacher_t1'
            S0_ASSET = 'student_s0'
            HF_REPO = 'VLADE-konkuk/VLADE-checkpoints'
            T1_HF = {'repo_id': HF_REPO, 'revision': '', 'prefix': ''}
            S0_HF = {'repo_id': HF_REPO, 'revision': '', 'prefix': ''}
            DATA_ASSET = 'prepared_d_seed42'

            LAMBDA_KD = 1.0
            MAX_STEPS = 1000
            ACCUMULATION_STEPS = 8
            LEARNING_RATE = 2e-5
            SAVE_STEPS = 250
            EVAL_STEPS = 100
            TEACHER_DEVICE = 'cuda:0'  # another GPU can be selected, e.g. cuda:1
            RESUME_CHECKPOINT = None  # only an existing KD run; S0 belongs in S0_HF above
            RESUME_HF_RUN = ''  # optional: <original-group>/<output-name>, normally group/student
            RESUME_HF_REVISION = ''  # immutable HF commit
            RESUME_HF_ALIAS = 'last'  # last / best / final

            def input_selection():
                return {'workspace': WORKSPACE, 'group': RUN_GROUP, 'registry': REGISTRY_REVISION,
                        'teacher_key': T1_ASSET, 'student_key': S0_ASSET, 'teacher': T1_HF,
                        'student': S0_HF, 'data': DATA_ASSET}

            print('RUN_GROUP:', RUN_GROUP)
        """, "config"),
        cell("code", """
            RUN_CONNECT_INPUTS = False
            if RUN_CONNECT_INPUTS:
                from vlade.runtime.environment import runtime_environment
                from vlade.storage.assets import read_registry, asset_record
                from vlade.storage.checkpoint_inputs import pin_checkpoint_inputs
                from vlade.research.registry import prepare_registry

                work = Path(WORKSPACE).resolve()
                os.environ.update(runtime_environment(work))
                from huggingface_hub import HfApi
                info = HfApi().model_info(HF_REPO)
                if info.private is not True:
                    raise ValueError('가중치·출력 저장소는 private이어야 합니다.')
                base = read_registry(prepare_registry(CODE, work, REGISTRY_REVISION)
                                     if REGISTRY_REVISION else CODE / 'configs/storage/hf_assets.lock.json')
                selections = {}
                for key, spec in ((T1_ASSET, T1_HF), (S0_ASSET, S0_HF)):
                    if spec['prefix'] or spec['revision']:
                        selections[key] = spec
                    elif key not in base['assets']:
                        raise ValueError(f'{key}: 등록 키를 지정하거나 HF revision/prefix를 입력하십시오.')
                if T1_ASSET == S0_ASSET:
                    raise ValueError('T1과 S0는 서로 다른 asset 키를 사용하십시오.')
                REGISTRY_PATH = work / 'registries' / (RUN_GROUP + '-inputs.json')
                registry = pin_checkpoint_inputs(base, selections, REGISTRY_PATH)
                os.environ['VLADE_ASSET_REGISTRY'] = str(REGISTRY_PATH)
                for key in (T1_ASSET, S0_ASSET, DATA_ASSET):
                    record = asset_record(key)
                    print(key, '|', record['repo_id'], '| revision:', record['revision'])
                _CONNECTED = copy.deepcopy(input_selection())
                print('입력 연결 완료. 아직 모델 본문은 다운로드하지 않았습니다.')
        """, "connect-inputs"),
        cell("markdown", """
            ## 3. 계획 확인

            `checkpoint`=S0, `distillation.teacher_checkpoint`=T1,
            `distillation.action_kd_weight`=λ입니다. `configs/research/qwen_kd.json`을 기본값으로 사용합니다.
            아래 계획은 다운로드·설치·학습을 실행하지 않습니다.
            설정을 바꿨으면 입력 연결과 계획 셀을 다시 확인하십시오. 가중치를 바꾸면 새 run group을 사용합니다.
        """, "plan-help"),
        cell("code", """
            CONFIG = CODE / 'configs/research/qwen_kd.json'
            def execution_overrides():
                return {'run_group': RUN_GROUP, 'registry_revision': None,
                    'checkpoint': 'hfasset://' + S0_ASSET,
                    'distillation.teacher_checkpoint': 'hfasset://' + T1_ASSET,
                    'distillation.action_kd_weight': LAMBDA_KD,
                    'distillation.teacher_device': TEACHER_DEVICE,
                    'train.max_steps': MAX_STEPS, 'train.accumulation_steps': ACCUMULATION_STEPS,
                    'train.learning_rate': LEARNING_RATE, 'train.save_steps': SAVE_STEPS,
                    'train.eval_steps': EVAL_STEPS, 'data': 'hfasset://' + DATA_ASSET,
                    'resume': RESUME_CHECKPOINT}
            OVERRIDES = execution_overrides()
            job(CONFIG, action='plan', workspace=WORKSPACE, overrides=OVERRIDES)
            _PLANNED = copy.deepcopy((input_selection(), OVERRIDES))

            def require_current_inputs():
                if globals().get('_CONNECTED') != input_selection():
                    raise RuntimeError('현재 설정으로 입력 연결 셀을 먼저 실행하십시오.')
                if _PLANNED != (input_selection(), execution_overrides()):
                    raise RuntimeError('설정이 바뀌었습니다. 계획 셀부터 다시 실행하십시오.')
                os.environ['VLADE_ASSET_REGISTRY'] = str(REGISTRY_PATH)
        """, "plan"),
        cell("markdown", """
            ## 4. 환경 준비·HF 저장 권한 확인

            AVA Teacher와 Qwen Student는 Transformers 버전이 다르므로 각각 환경을 준비합니다.
            둘은 별도 프로세스이며 기본적으로 같은 GPU에 상주합니다. 합산 VRAM 여유를 파일럿에서 확인하십시오.
            이 setup은 모델 다운로드·학습은 시작하지 않지만 패키지를 설치하므로 시간이 걸립니다.
        """, "setup-help"),
        cell("code", """
            RUN_SETUP = False
            if RUN_SETUP:
                require_current_inputs()
                job(CONFIG, action='setup', workspace=WORKSPACE, overrides=OVERRIDES)
        """, "setup"),
        cell("code", """
            RUN_HUB_CHECK = False
            if RUN_HUB_CHECK:
                require_current_inputs()
                from vlade.runtime.environment import locate
                student_python = locate(CODE, Path(WORKSPACE), 'student')
                os.environ['VLADE_HUB_REPO'] = HF_REPO
                os.environ['VLADE_HUB_RUN_ID'] = RUN_GROUP
                subprocess.run([str(student_python), '-B', str(CODE / 'scripts/train/hub_checkpoints.py'),
                    'check', '--output', str(Path(WORKSPACE) / 'runs' / RUN_GROUP / 'student')], check=True)
        """, "hub-check"),
        cell("markdown", """
            ## 5. 실제 입력 검사 · 여기부터 모델/데이터를 다운로드합니다

            `run_student_kd.py --preflight`는 HF 무결성, S0 전체 구성, T1 FFT 상태,
            데이터 split·정규화를 검사합니다. GPU forward는 하지 않습니다.
            실행할 파일·설정은 다음 셀에서 그대로 확인할 수 있습니다.
        """, "preflight-help"),
        cell("code", """
            RUN_PREFLIGHT = False
            if RUN_PREFLIGHT:
                require_current_inputs()
                from vlade.runtime.environment import locate
                from vlade.data.episode_data import write_json
                student_python = locate(CODE, Path(WORKSPACE), 'student')
                teacher_python = locate(CODE, Path(WORKSPACE), 'ava')
                base = json.loads(CONFIG.read_text(encoding='utf-8'))
                student = json.loads((CODE / base['model_config']).read_text(encoding='utf-8'))['student']
                flat = {**base['train'], **base['distillation'], 'student': student,
                    'seed': base['seed'], 'initialization': 'student_checkpoint',
                    'data': OVERRIDES['data'], 'initial_checkpoint': OVERRIDES['checkpoint'],
                    'teacher_checkpoint': OVERRIDES['distillation.teacher_checkpoint'],
                    'teacher_device': TEACHER_DEVICE, 'teacher_python': str(teacher_python),
                    'action_kd_weight': LAMBDA_KD, 'max_steps': MAX_STEPS,
                    'accumulation_steps': ACCUMULATION_STEPS, 'learning_rate': LEARNING_RATE,
                    'save_steps': SAVE_STEPS, 'eval_steps': EVAL_STEPS,
                    'output': str(Path(WORKSPACE) / 'runs' / RUN_GROUP / 'student')}
                PREFLIGHT_CONFIG = Path(WORKSPACE) / 'configs' / (RUN_GROUP + '-preflight.json')
                write_json(PREFLIGHT_CONFIG, flat)
                command = [str(student_python), '-B', str(CODE / 'scripts/train/run_student_kd.py'),
                           '--config', str(PREFLIGHT_CONFIG), '--preflight']
                print('실행 파일:', CODE / 'scripts/train/run_student_kd.py')
                subprocess.run(command, check=True)
        """, "preflight"),
        cell("markdown", """
            ## 6. 파일럿 → 본 학습

            파일럿은 기존 1-step 저장/재로딩 검사와 한 train Episode의 50-step overfit 검사를 사용합니다.
            GT·Teacher L1과 λ 가중합을 기록합니다. 파일럿 완료만으로 rollout 성공률을 주장하지 않습니다.
            본 학습은 파일럿 가중치를 이어받지 않고 원래 S0에서 시작합니다.
            본 학습 셀도 파일럿 통과를 확인하므로 먼저 파일럿 셀을 실행하십시오.
        """, "training-help"),
        cell("code", """
            RUN_PILOT = False
            if RUN_PILOT:
                require_current_inputs()
                if RESUME_CHECKPOINT:
                    raise ValueError('재개 시 기존 파일럿을 유지하고 본 학습 셀을 사용하십시오.')
                job(CONFIG, action='run', workspace=WORKSPACE, overrides={**OVERRIDES, 'pilot.only': True})
        """, "pilot"),
        cell("code", """
            RUN_TRAIN = False
            if RUN_TRAIN:
                require_current_inputs()
                job(CONFIG, action='run', workspace=WORKSPACE, overrides={**OVERRIDES, 'pilot.only': False})
        """, "train"),
        cell("markdown", """
            ## 7. 저장·재개

            KD 출력은 `runs/<RUN_GROUP>/student`이고, 봉인 checkpoint와 last/best/final pointer는
            private HF `runs/<RUN_GROUP>/student/`에 게시됩니다. 목표 step을 마친 경우에만 final입니다.
            pending 전송은 `scripts/train/hub_checkpoints.py retry --output <원래 output>`으로 복구합니다.

            재개는 최초 SOURCE_COMMIT·RUN_GROUP·가중치·설정·데이터를 그대로 사용합니다.
            아래 선택 셀에서 HF의 원래 run과 **불변 revision**을 지정하여 checkpoint를 복구한 뒤,
            `RESUME_CHECKPOINT`가 설정되면 **계획 셀을 다시 실행하고 본 학습 셀**을 실행합니다.
            S0를 지정하는 것과 KD optimizer/cursor를 복원하는 재개는 별개입니다.
            기존 파일럿 보고서는 같은 WORKSPACE에서 보존하십시오. 런타임이 바뀌면 파일럿을 다시 검증합니다.
        """, "resume-help"),
        cell("code", """
            RUN_DOWNLOAD_RESUME = False
            if RUN_DOWNLOAD_RESUME:
                require_current_inputs()
                if not re.fullmatch(r'[0-9a-f]{40}', RESUME_HF_REVISION):
                    raise ValueError('재개 HF commit SHA가 필요합니다.')
                if RESUME_HF_RUN != RUN_GROUP + '/student':
                    raise ValueError('원래 RUN_GROUP/student를 지정하십시오.')
                from vlade.runtime.environment import locate
                student_python = locate(CODE, Path(WORKSPACE), 'student')
                destination = Path(WORKSPACE) / 'restored' / RUN_GROUP
                command = [str(student_python), '-B', str(CODE / 'scripts/train/hub_checkpoints.py'),
                    'download', '--repo', HF_REPO, '--run', RESUME_HF_RUN, '--alias', RESUME_HF_ALIAS,
                    '--revision', RESUME_HF_REVISION, '--output', str(destination)]
                subprocess.run(command, check=True)
                checkpoints = sorted(destination.glob('step-*'))
                if len(checkpoints) != 1:
                    raise ValueError('복구한 checkpoint 경로를 직접 확인하십시오.')
                RESUME_CHECKPOINT = str(checkpoints[0])
                print('RESUME_CHECKPOINT:', RESUME_CHECKPOINT, '| 계획 셀부터 다시 실행하십시오.')
        """, "download-resume"),
    ]
    value = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
             "language_info": {"name": "python"}, "colab": {"name": path.name}}, "nbformat": 4, "nbformat_minor": 5}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def main() -> None:
    print(write_notebook(ROOT / "experiments/colab_action_distillation.ipynb").relative_to(ROOT))
