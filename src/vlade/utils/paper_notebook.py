"""Author the independent 04 paper-reference notebook without running training."""
from __future__ import annotations

from pathlib import Path


def generate_paper_notebook(root: Path, digest: str) -> Path:
    from vlade.utils.colab_notebooks import cell,notebook,bootstrap,ava_install_source
    path = root/"experiments/colab_04_ava_paper_best.ipynb"
    notebook(path,[
        cell("markdown", """
        # 04 · AVA-VLA 논문 최고 보고 설정 · 100K / batch 256
        [논문 v4 Appendix D.1/Table 7](https://arxiv.org/html/2511.18960v4)은 사전학습 OpenVLA에서
        100,000 gradient step·유효 batch 256으로 학습한 AVA-VLA의 LIBERO 평균 성공률 98.3%를 보고합니다.
        이 숫자는 논문 보고값이며 우리 측정값이나 LIBERO+ 보장 성능이 아닙니다.
        나머지 하이퍼파라미터는 Appendix A의 rank 32·K=4·2/3 경계 detach·LR 5e-4·10% warm-up·cosine을 따릅니다.
        D.1이 나머지 조건을 모두 재명시하지 않았다는 점은 설정에 기록합니다.
        우리 실행은 01의 LIBERO+ train 분할/정규화·query 간격 8을 쓰므로 데이터 조건이 다릅니다.
        02의 zero-context FFT와 별도 실험입니다. 기존 AVA/OFT 가중치를 초기값으로 대신 쓰지 않습니다.
        **준비 전용 기본값:** 설치·다운로드·파일럿·본 학습 플래그는 모두 False입니다. 모두 실행해도 대규모 작업은 시작하지 않습니다.
        100K 목표를 8시간 예산에 맞춰 축소하지 않습니다. 8시간은 실행 세션별 상한이며 전체 실험에는 여러 세션이 필요할 수 있습니다.
        """, "purpose"),
        cell("code", """
        import time, os
        PAPER_SESSION_START = globals().get('PAPER_SESSION_START',time.time())
        os.environ['VLADE_DEADLINE_UNIX'] = str(PAPER_SESSION_START+8*3600)
        """, "paper-session-start"),
        cell("code",bootstrap(digest,isolated_uv=True),"drive-and-source"),
        cell("code", """
        RUN_INSTALL = False
        RUN_DOWNLOAD = False
        RUN_PILOT = False
        RUN_PAPER_TRAINING = False
        config = json.loads((CODE/'configs/experiments/ava_paper_best.json').read_text())
        config['initial_checkpoint'] = str(ROOT/config['initial_checkpoint'])
        config['data'] = str(ROOT/config['data'])
        config['output'] = str(ROOT/config['output'])
        CONFIG_PATH = ROOT/'run-configs/ava-paper-best-seed42.json'
        CONFIG_PATH.parent.mkdir(exist_ok=True)
        if CONFIG_PATH.exists() and json.loads(CONFIG_PATH.read_text()) != config:
            raise ValueError('기존 04 실행 설정과 다릅니다. 기존 설정/결과를 덮어쓰지 않습니다.')
        CONFIG_PATH.write_text(json.dumps(config,indent=2))
        print(json.dumps(config,indent=2,ensure_ascii=False))
        print('1 optimizer step = 256 K=4 sequences = 1024 query 예측')
        """, "paper-config"),
        cell("markdown", """
        ## 환경 설치 (선택)
        02와 같은 검증된 AVA Transformers 패치를 사용합니다. K=4의 두 프레임 BPTT를 위해
        non-reentrant gradient checkpointing을 사용합니다. 실제 GPU 메모리 적합성은 파일럿에서 확인해야 합니다.
        준비만 할 때는 RUN_INSTALL=False를 유지하십시오.
        """, "install-help"),
        cell("code","if RUN_INSTALL:\n    import threading\n    from google.colab import runtime\n"
             "    _paper_release_timer = threading.Timer(max(0,PAPER_SESSION_START+8*3600-time.time()),runtime.unassign)\n"
             "    _paper_release_timer.daemon = True\n    _paper_release_timer.start()\n"+
             "\n".join("    "+line for line in ava_install_source().splitlines())+
             "\nelse:\n    print('환경 설치 미실행: RUN_INSTALL=False')", "install-ava"),
        cell("code", """
        if RUN_INSTALL:
            import threading
            from google.colab import runtime
            if globals().get('_paper_release_timer'):
                _paper_release_timer.cancel()
            _paper_release_timer = threading.Timer(max(0,PAPER_SESSION_START+8*3600-time.time()),runtime.unassign)
            _paper_release_timer.daemon = True
            _paper_release_timer.start()
            print('8시간 세션 마감 시 런타임 반환을 요청합니다.')
        """, "paper-session-limit"),
        cell("code", """
        if RUN_DOWNLOAD:
            require_ava_environment()
            program = "from pathlib import Path; import sys; sys.path.insert(0,'src'); from vlade.data.colab_assets import download; download(sys.argv[1],sys.argv[2],Path(sys.argv[3]))"
            run([AVA_PY,'-c',program,config['initial_repository'],config['initial_revision'],config['initial_checkpoint']])
        else:
            print('가중치 다운로드 미실행: RUN_DOWNLOAD=False (사전학습 OpenVLA 약 15.1 GB)')
        """, "download-pretrained"),
        cell("code", """
        if RUN_PILOT:
            require_ava_environment()
            run([AVA_PY,'scripts/train/run_paper_ava.py','--config',CONFIG_PATH,'--preflight'])
            pilot = Path(config['output']+'-pilot')
            if not (pilot/'pilot_report.json').is_file():
                run([AVA_PY,'scripts/train/run_paper_ava.py','--config',CONFIG_PATH,'--pilot'])
            print((pilot/'pilot_report.json').read_text())
        else:
            print('K=4 backward·LoRA·정규화·저장/재로딩 파일럿 미실행')
        """, "paper-pilot"),
        cell("markdown", """
        ## 본 학습/재개 (선택)
        파일럿은 1 sequence로 K=4 역전파·저장/재로딩 경로를 점검하며 batch 256의 성능 검증은 아닙니다.
        본 학습은 microbatch 1×누적 256을 사용합니다. 첫 step·5,000 step 간격·세션 종료 전에
        LoRA adapter, head/proprio, AVA generator, optimizer/scheduler/RNG와 sequence cursor를 Drive에 저장합니다.
        100,000 step과 warm-up 10,000은 재개 중 바꾸지 않습니다. 중단 시 데이터 cursor를 복원합니다.
        CPU 테스트만 통과한 자체 학습 어댑터이며 저자 학습 코드 재현 또는 98.3% 달성을 검증한 것은 아닙니다.
        기존 02의 T1/cache 도구는 FFT 형식 전용이므로 04 adapter checkpoint를 직접 입력하지 마십시오.
        """, "paper-train-help"),
        cell("code", """
        if RUN_PAPER_TRAINING:
            require_ava_environment()
            # 재개할 때는 이 경로만 지정하고 목표 step/LR schedule은 유지합니다.
            RESUME_CHECKPOINT = ''
            command = [AVA_PY,'scripts/train/run_paper_ava.py','--config',CONFIG_PATH]
            if RESUME_CHECKPOINT:
                command += ['--resume',RESUME_CHECKPOINT]
            else:
                report = Path(config['output']+'-pilot')/'pilot_report.json'
                if not report.is_file():
                    raise RuntimeError('K=4 실제 파일럿을 먼저 실행하십시오.')
                check = "import json,sys; from pathlib import Path; sys.path.insert(0,'src'); from vlade.training.paper_ava import require_pilot; require_pilot(json.loads(Path(sys.argv[1]).read_text()))"
                run([AVA_PY,'-c',check,CONFIG_PATH,report])
            run(command)
        else:
            print('준비 완료. 본 학습은 실행하지 않았습니다. RUN_PAPER_TRAINING=False')
        """, "paper-train"),
    ])
    return path
