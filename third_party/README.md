# Third-party sources

외부 원본 소스를 Git submodule로 연결합니다. 원본은 직접 수정하지 않고
`src/vlade/`의 어댑터에서 통합합니다.

| 경로 | 저장소 | 고정 커밋 | 라이선스 |
| --- | --- | --- | --- |
| `ava_vla/` | https://github.com/eastha10/AVA-VLA | `69ae5dfb90f4eb714259693bd2c222e0b3de6456` | `ava_vla/LICENSE`의 Apache 2.0 및 포함된 OpenVLA-OFT MIT 고지 |
| `openvla_oft/` | https://github.com/moojink/openvla-oft | `e4287e94541f459edc4feabc4e181f537cd569a8` | `openvla_oft/LICENSE`의 MIT |

AVA-VLA 사용자 포크의 공식 원본은 https://github.com/LiAuto-DSR/AVA-VLA 이며,
OpenVLA-OFT는 https://github.com/openvla/openvla 에서 파생된 저장소입니다.

```bash
git submodule update --init --recursive
```

소스만 받는 명령입니다. 두 LIBERO 공개 가중치는 이 로컬 작업 공간에 별도로
준비했지만 Git에 포함되지 않습니다. LIBERO+ 데이터·평가 assets는 로컬에서 삭제했고,
실행 의존성과 GPU 추론은 아직 준비·검증하지 않았습니다.

AVA-VLA Teacher 연결 설정은 `configs/models/teacher/ava_vla.json`에 기록합니다.
두 원본의 baseline 설정은 각각 `experiments/baselines/ava_vla/libero.json`과
`experiments/baselines/openvla_oft/libero.json`, 소스 검증 상태는
`experiments/manifests/`에서 관리합니다. 의존성 설치, 가중치 다운로드와 실행 검증은
baseline 등록과 구분해 기록합니다.
