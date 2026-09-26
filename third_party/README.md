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

소스만 받는 명령입니다. 모델 가중치, 벤치마크 데이터와 실행 의존성은 별도로
준비해야 합니다. 가중치는 다음 단계에서 다운로드하고 검토합니다.

AVA-VLA 연결 설정과 검증 상태는 `configs/models/teacher/ava_vla.json` 및
`experiments/manifests/ava_vla_source.json`에 기록합니다. OpenVLA-OFT는 현재
원본 소스만 이양했으며 의존성 설치, 가중치 다운로드와 실행 검증은 수행하지 않았습니다.
