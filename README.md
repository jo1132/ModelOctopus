# ModelOctopus — 모델 학습·경량화·배포 라이브러리 설계

작성일: 2026-10-28. `ModelOctopus`, `modeloctopus`, `ef`는 작업용 이름이며 패키지명·상표 사용 가능 여부를 확인한 이름은 아닙니다.

M0 core를 구현했습니다. 설치·실행 방법은 [M0 API](docs/m0-api.md), 완료 범위와 검증은 [상태](docs/status.md), 실제 환경은 [환경 matrix](docs/environment-matrix.md)를 참고하세요. `ef plan`과 `ef doctor`, strict schema, 로컬 artifact store, plugin registry와 TinyCNN fixture를 제공합니다. 아래 전체 흐름의 학습·압축·배포 API는 M1 이후 목표이며 아직 실행할 수 없습니다.

사용자의 기존 YOLO Edge Lab을 다중 모델 라이브러리로 확장합니다. 모델 생성, 체크포인트 로드, 학습, 데이터셋 준비, pruning, quantization, knowledge distillation, PyTorch/ONNX/TensorRT 실행을 연결합니다. 개인 연구의 핵심은 새로운 quantization 및 KD 방법을 plugin으로 추가하는 것입니다.

여기서 요청의 “Tensor”는 기존 대화의 TensorRT로 해석했습니다. TensorFlow/Keras를 뜻했다면 별도의 학습·모델·export adapter 범위를 추가해야 합니다. ONNX를 통해 모든 프레임워크의 학습을 통합할 수 있다고 가정하지 않습니다.

| 파일 | 용도 |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 목표, 모듈 구조, 지원 범위, 데이터·모델 흐름, 배포 의미 |
| [API_CONTRACTS.md](API_CONTRACTS.md) | 타입, plugin 계약, 학습·변환 API, 실패 처리 |
| [RESEARCH_EXTENSIONS.md](RESEARCH_EXTENSIONS.md) | 사용자 정의 quantization·KD 추가 방법과 비교 실험 |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | M0~M9 개발 순서, 작업 단위와 통과 조건 |
| [AGENTS.md](AGENTS.md) | 새 저장소 루트에 둘 Codex 작업 규칙 |
| [CODEX_START.md](CODEX_START.md) | 처음 Codex에 넣을 프롬프트, 단계별 후속 프롬프트 |
| [examples/classification_kd_ptq.yaml](examples/classification_kd_ptq.yaml) | 분류 teacher/student 학습과 KD·PTQ의 목표 recipe |
| [examples/yolo_ptq.yaml](examples/yolo_ptq.yaml) | 기존 YOLO 학습·PTQ 흐름의 목표 recipe |
| [SOURCES.md](SOURCES.md) | 확인한 공식 자료와 설계 판단의 근거 |

시작 방법:

1. 새 Git 저장소에 이 문서 묶음을 복사합니다. 기존 프로그램이 있는 저장소라면 기존 `AGENTS.md`를 덮어쓰지 말고 규칙을 통합합니다.
2. Codex에 `CODEX_START.md`의 첫 프롬프트를 전달합니다.
3. M0에서 버전·실행 환경·계약을 확정하고 M1의 작은 실행 경로를 완성합니다.
4. 각 milestone의 실제 증거가 있는 기능만 supported로 승격합니다.

첫 목표는 **분류 모델 2종과 YOLO 1종이 같은 모델/데이터/산출물 계약을 사용하고, ONNX 내보내기·실행·그래프 조회까지 되는 것**입니다. 이후 pruning과 KD를 결합하고 사용자 quantization 방법을 실험합니다. 모든 모델에 모든 기능을 일괄 지원한다고 선언하지 않습니다.
