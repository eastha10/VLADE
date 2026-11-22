# 기존 HF 정본의 읽기 전용 입력 등록

이미 내려받은 모델·prepared 데이터를 이동·복사·덮어쓰기하거나 다시 다운로드하지
않고 사용하는 절차입니다. 임의의 로컬 입력을 허용하는 옵션이 아닙니다. 등록 전에
현재 registry의 고정 HF revision·원격 지문과 실제 기존 파일을 검증합니다.

등록은 sealed directory asset, 게시된 `hub_checkpoint`, 고정 `upstream_files`와 schema2 prepared dataset archive의 기존 해제본을
지원합니다. 원본 root 밖의 별도 JSON receipt에 asset key, 전체 고정 record,
검증한 원격 저장소/revision과 기존 절대 경로를 보관합니다. 이 receipt는 실행 캐시
메타데이터이며 Git·Drive 배포 대상이 아닙니다. 개인 경로를 정식 설정에 넣지 않습니다.

## 등록

소스 checkout의 루트에서 실제 기존 파일 경로와 고정 registry를 지정합니다.

```powershell
python scripts/data/bind_local_input.py --key S0 `
  --path checkpoints/student/qwen-vla-s0-seed42/S0 `
  --bindings data/hf_assets/local_inputs.json `
  --registry configs/quantization/s0_assets.json

python scripts/data/bind_local_input.py --key prepared_d_seed42 `
  --path data/processed/libero_plus_episode80_d_seed42 `
  --bindings data/hf_assets/local_inputs.json `
  --registry configs/quantization/s0_assets.json
```

등록 시 인증된 HF API로 고정 revision의 원격 파일 크기·LFS SHA256 또는 작은
metadata 파일 SHA256을 확인합니다. 가중치·데이터 archive를 다운로드하지 않습니다.
모델은 record에 선언한 모든 파일의 실제 SHA256을 검사합니다. Prepared 데이터는
HF에 고정된 manifest/normalization/split을 먼저 대조한 뒤, 그 manifest가 인증하는
모든 trajectory·query·image 파일을 전수 검사합니다. dataset COMPLETE와 train 내부
calibration 경계도 검사합니다. 로컬 tar archive를 재구성하여 비교하는 절차는 아닙니다.

입력은 읽기만 하며 등록 파일은 반드시 입력 밖에 둡니다. 기존 등록 항목을 유지하고
lock 및 원자적 파일 교체로 등록을 갱신합니다. 검증 실패 시 성공 receipt를 쓰지 않고
이전 등록을 보존합니다. 다른 등록 프로세스의 lock을 임의로 지우지 않습니다.

## 실행

```powershell
$env:VLADE_LOCAL_INPUTS = (Resolve-Path data/hf_assets/local_inputs.json).Path
$env:HF_HUB_OFFLINE = '1'
python scripts/quantization/run_s0_comparison.py `
  --config configs/quantization/s0_comparison.json --run
```

`hfasset://S0`, `hfasset://prepared_d_seed42`는 그대로 사용합니다. 공통
`resolve_input`/HF URI의 `resolve_path`가 등록을 확인하고 기존 원본 경로를 반환합니다.
같은 registry에서 검증한 정확한 등록 root의 로컬 경로도 `resolve_input`으로 허용합니다.
그 아래의 임의 하위 폴더나 미등록 로컬 경로는 여전히 외부 입력으로 허용하지 않습니다.

receipt만 신뢰해 검증을 생략하지 않습니다. 각 프로세스의 사용 시점에 모델 파일 또는
prepared metadata의 실제 hash를 다시 검사합니다. 데이터 reader는 원래대로 실제
읽는 trajectory/query/image의 SHA256을 검증합니다. registry record·revision 변경,
root 경로 변경·삭제·symlink 우회, byte 변조는 실패하며 조용히 다른 모델을 받지 않습니다.

`VLADE_LOCAL_INPUTS`를 지정하지 않으면 기존 canonical HF 캐시 동작을 유지합니다.
지정한 파일이 없거나 손상된 경우는 오류입니다. 파일에 등록되지 않은 HF 키는 기존
HF resolver를 사용하므로, 새 다운로드를 금지할 실행에서는 위처럼 offline을 지정합니다.
등록 자체는 원격 지문을 확인해야 하므로 offline 모드에서 수행하지 않습니다.

독립 비교 실행기는 등록된 root도 출력 금지 경로로 취급합니다. 등록 파일의 SHA256을
manifest와 worker의 입력 identity에 기록하므로, 등록 정보가 바뀌면 이전 실험의
resume/reference를 그대로 섞지 않습니다. 새 등록을 만들었으면 새 run group을 사용합니다.
