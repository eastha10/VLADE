# Third-party source

AVA-VLA 원본 소스를 `ava_vla/` Git submodule로 연결합니다.
원본은 직접 수정하지 않고 `src/vlade/`의 어댑터에서 통합합니다.

| 항목 | 값 |
| --- | --- |
| 사용자 포크 | https://github.com/eastha10/AVA-VLA |
| 공식 원본 | https://github.com/LiAuto-DSR/AVA-VLA |
| 고정 커밋 | `69ae5dfb90f4eb714259693bd2c222e0b3de6456` |
| 원본 고지 | `ava_vla/LICENSE`의 Apache 2.0 및 OpenVLA-OFT MIT 고지 보존 |

```bash
git submodule update --init --recursive
```

소스만 받는 명령입니다. 모델 가중치, 벤치마크 데이터와 실행 의존성은 별도로
준비해야 합니다. 가중치는 다음 단계에서 다운로드하고 검토합니다.

연결 설정과 검증 상태는 `configs/models/teacher/ava_vla.json` 및
`experiments/manifests/ava_vla_source.json`에 기록합니다.
