# Jetson packed PTQ 준비

- 로컬 `39ca0807e87042d0f82fc55d7369d5d6f3c5e924`에서 `python -B scripts/quantization/run_packed_ptq.py --config configs/quantization/packed_student.json`의 계획 출력을 확인했다. S0와 LIBERO+ 입력이 고정되어 있으며 실제 GPU 실행은 수행하지 않았다.
- Jetson의 유선 기본 경로는 설정됐으나, Windows에서 HTTPS가 확인된 GitHub IPv4로 Jetson이 연결할 때 443번 포트 시간 초과가 발생했다. Windows 실행 환경에서 기존 Jetson SSH 주소로 접속을 시도하면 `Permission denied`를 반환한다.
- `git bundle create <bundle> refs/heads/main`으로 Git 경로만 전송할 수 있는 `scratch/vlade-jetson-39ca080-main.bundle`을 준비했다. `git bundle verify`로 사전 커밋이 필요 없는 번들임을 확인했고, `git bundle list-heads`로 지정 커밋이 포함됐는지 확인했다. 2,048,795 bytes이며 LAN 전송에 필요해 scratch에 남겼다. 가중치·데이터는 포함하지 않는다.
- Jetson으로의 전송과 PTQ 실행은 미실행이다. Windows 인터넷 공유 설정을 이 세션에서 변경하지 않았다.
