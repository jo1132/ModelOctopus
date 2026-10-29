# Codex에 전달할 작업 프롬프트

아래 첫 프롬프트는 설계 문서 묶음을 새 저장소에 복사한 뒤 전달한다. 현재 사용자의 요청은 설계이므로 이 문서 작성 단계에서 라이브러리 구현을 실행하지 않는다.

## 첫 작업: M0

```text
이 저장소의 README.md, AGENTS.md, ARCHITECTURE.md, API_CONTRACTS.md,
RESEARCH_EXTENSIONS.md, IMPLEMENTATION_PLAN.md를 읽고 ModelOctopus 라이브러리의
M0만 실제로 구현해줘.

목표는 모델 학습·경량화·배포를 지원할 Python 라이브러리의 최소 core다.
이번 범위는 다음과 같아.

1. src layout과 pyproject, 최소 core dependency, dev test 설정.
2. ArtifactRef, ModelSpec, DatasetSpec, TrainSpec, capability/error schema.
3. 해시와 atomic publish를 갖는 로컬 artifact store.
4. plugin registry와 optional backend의 lazy loading.
5. recipe DAG의 참조·cycle·입출력 kind 검사와 ef plan/doctor.
6. TinyCNN fixture와 모델 round-trip 계약의 시작점.
7. 의미 있는 단위·계약 테스트, 실제 실행 가능한 작은 사용 예.

M1 이후 기능은 구현한 척하지 말고 planned/unsupported로 표시해줘.
Docker/GPU/runtime을 사용할 수 있는지 먼저 확인하고 실제 가능한 검증을 수행해줘.
준비되지 않은 backend 테스트를 성공한 것으로 기록하지 마.
현재 환경에서 확인한 의존성 조합과 미검증 조합을 구분해 기록해줘.

수정 전 저장소 상태와 기존 파일을 확인하고, 작업이 끝나면 docs/status.md에
구현된 항목, 실제 실행한 테스트, 미실행 이유, 다음 작업을 기록해줘.
계획만 제시하지 말고 이 범위의 코드와 검증까지 완료해줘.
```

## 다음 작업: M1의 작은 실행 경로

```text
AGENTS.md와 설계 문서, docs/status.md를 확인해줘.
M1에서 TinyCNN 및 torchvision 분류 모델의 create/load/train/export/predict 경로를 구현해줘.
먼저 random weights와 작은 synthetic classification 데이터로 offline 테스트를 완성하고,
실제 다운로드를 허용하는 integration test는 분리해줘.
PyTorch 모델 저장·새 프로세스 복원, epoch 경계 resume, FP32 ONNX/ORT 출력 비교,
Netron 기반 graph view와 종료 처리를 검증해줘.
core가 YOLO/AIMET/TensorRT에 의존하지 않도록 유지해줘.
구현 증거가 없는 조합은 supported로 표시하지 마.
```

## YOLO와 PTQ를 연결할 때

```text
현재 완료된 milestone과 capability를 먼저 확인하고 M2의 YOLOv8n adapter를 구현해줘.
기존 YOLO Edge Lab의 코드는 참고할 수 있지만 학습·양자화 E2E가 검증됐다고 가정하지 마.
Ultralytics native training과 고정된 전후처리/출력 계약을 연결하고
COCO8 실제 학습·FP32 ONNX export·ORT 수치 비교를 실행해줘.
YOLO pruning/KD는 아직 unsupported로 유지해줘.
그 뒤 별도 작업으로 M3의 작은 Conv AIMET round-trip을 먼저 통과시킨 다음
분류 모델과 YOLO의 PTQ로 확장할 수 있도록 결과를 남겨줘.
```

## 사용자 quantization 방법 추가

```text
RESEARCH_EXTENSIONS.md와 현재 QuantizationStrategy 계약을 읽어줘.
아래에 설명할 내 quantization 방법을 외부 plugin으로 구현해줘.

[내 방법의 수식/의사코드/논문 기준/변경할 baseline을 여기에 작성]

구현 전에 알고리즘 입력·출력, labels/gradient 필요 여부, weight mutation,
quantization state, export 가능 표현과 대상 runtime 조건을 명확히 정리해줘.
방법에서 정해지지 않은 핵심 수학 규칙은 임의로 만들어 성능을 주장하지 마.
단순한 default로 진행 가능한 항목은 가정을 기록하고 진행해줘.
작은 수치·gradient fixture, save/load, 가능한 경우 standard Q/DQ round-trip을 검증해줘.
기존 baseline과 동일 모델/seed/data/bitwidth/budget으로 비교하는 recipe를 추가해줘.
표준 ONNX나 대상 kernel이 지원하지 않으면 research_only로 표시해줘.
```

## 사용자 KD 방법 추가

```text
현재 DistillationObjective와 TaskAdapter 계약을 확인하고 아래 KD 방법을 plugin으로 추가해줘.

[teacher/student, 사용할 logits/features, 정렬 방법, loss 수식을 여기에 작성]

teacher freeze와 BN state 보존, student/projection gradient,
feature shape·class mapping·augmentation geometry를 테스트해줘.
objective의 trainable state가 optimizer와 checkpoint/resume에 포함되도록 해줘.
같은 student 초기값과 학습 budget으로 supervised-only 및 baseline KD와 비교할 수 있게 해줘.
분류용 loss를 탐지 모델에 자동 적용하지 말고 task-specific 계약을 확인해줘.
```

## 리뷰를 요청할 때

```text
현재 변경을 설계와 대조해 리뷰해줘.
특히 artifact 불변성, pruned 구조 복원, best/last checkpoint 혼합,
data leakage, teacher gradient/state 변화, fake quant와native kernel의 혼동,
미지원 조합의 silent fallback, TensorRT target profile, 실제 검증 없는 supported 표시를 확인해줘.
실제 문제와 증거를 우선해서 정리하고 수정 가능한 문제는 관련 테스트와 함께 고쳐줘.
```
