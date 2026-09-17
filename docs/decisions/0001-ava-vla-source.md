# 0001: AVA-VLA를 Teacher 및 baseline 소스로 연결

날짜: 2026-09-17

VLADE의 첫 기반 소스로 사용자 포크 `eastha10/AVA-VLA`를 채택합니다. 검토 시
공식 `LiAuto-DSR/AVA-VLA`의 main과 같은 커밋을 가리킵니다.

원본은 `third_party/ava_vla/` Git submodule로 연결하고 커밋을 고정합니다. VLADE
자체 구현은 `src/vlade/`에 두어 원본 평가 코드와 라이선스 고지를 보존합니다.

이번 변경은 소스 이양과 Teacher 연결 설정까지 포함합니다. Hugging Face 가중치는
나중에 다운로드해 `.pt` 부속 모듈과 전체 모델 구성의 호환성을 검토합니다.
Teacher·Student 어댑터와 증류 학습은 해당 검토 이후 구현합니다.

실험 결과를 보고할 때 원 논문의 수치를 자체 재현 결과로 사용하지 않습니다.
실행 환경, 모델·데이터 revision과 가중치 해시는 실제 실험 manifest에 추가합니다.
