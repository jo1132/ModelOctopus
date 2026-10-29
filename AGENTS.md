# Codex repository instructions

이 파일은 새 ModelOctopus 구현 저장소를 위한 지침이다. 현재 문서 묶음은 설계이며 구현 완료 상태가 아니다. 사용자의 명시적 현재 요청이 이 문서보다 우선한다.

## 구현 전 읽기

1. README.md
2. ARCHITECTURE.md
3. API_CONTRACTS.md
4. RESEARCH_EXTENSIONS.md
5. IMPLEMENTATION_PLAN.md
6. 존재하면 docs/status.md와 현재 작업 범위의 소스·테스트

## 범위

- 사용자가 요청한 milestone/task를 구현한다. 첫 요청이 전체 골격이라면 M0부터 완성한다.
- 각 단계의 결과를 다음 단계 입력으로 사용할 수 있게 끝낸다.
- unsupported 기능을 성공처럼 보이게 하는 임시 구현을 만들지 않는다.
- 공개 API와 recipe가 바뀌면 문서/schema/example/test를 함께 고친다.
- 기존 Git 변경과 파일을 확인하고 사용자 작업을 덮어쓰지 않는다.

## 아키텍처

- src layout을 사용한다. core는 backend dependency를 import하지 않는다.
- application service는 model/task/algorithm/runtime adapter를 통해 실행한다.
- Model ArtifactRef와 live nn.Module을 구분한다. worker 사이에 live model 객체를 전달하지 않는다.
- YOLO와 분류 모델에 동일한 loss/output 계약을 강제하지 않는다.
- 모델 생성 가능성과 pruning/KD/export 지원 가능성을 따로 선언한다.
- Python API와 CLI는 같은 request/service 구현을 사용한다.
- third-party package의 site-packages를 직접 수정하지 않는다. adapter나 version pin으로 해결한다.

## 모델·데이터·상태

- 입력 artifact와 teacher는 불변이다. 변경은 새 artifact로 기록한다.
- scratch/finetune/resume를 구분한다. epoch-boundary resume부터 검증한다.
- best model과 last training state의 epoch/optimizer를 혼합하지 않는다.
- pruning 후 축소 구조를 재생성하는 ArchitectureRecipe를 저장하고 새 process에서 검증한다.
- calibration과 val/test의 누출을 검사한다. synthetic/COCO8 결과를 실제 성능 근거로 사용하지 않는다.
- transform/color/layout/shape/class mapping/NMS의 의미를 manifest로 고정한다.
- 임의의 data/model download를 import 시점에 실행하지 않는다.

## Quantization과 KD

- fake quant simulation과 실제 low-precision kernel 실행을 구분한다.
- PTQ의 gradient/label 요구 조건을 plugin별로 선언한다. 무조건 no_grad로 감싸지 않는다.
- 표준 Q/DQ가 불가능한 method는 research_only와 명확한 export 오류를 반환한다.
- dtype만 바꾸거나 Q/DQ 노드만 추가해 정상 양자화 완료로 표시하지 않는다.
- teacher parameter와 BN state를 고정하고 student/objective projection의 gradient를 검증한다.
- KD objective가 optimizer.step을 독자적으로 호출하지 않는다.
- YOLO KD에 분류 softmax loss를 검증 없이 사용하지 않는다.
- 구조적 pruning과 unstructured mask의 size/performance 효과를 구분한다.

## 실행과 검증

- 작업 전 Docker/GPU/Python/runtime 조건을 확인한다.
- 의존성은 기능별 profile로 고정하고 공식 문서/API와 실제 설치 버전을 대조한다.
- CPU/GPU fallback은 명시적 policy와 report가 있을 때만 허용한다.
- 테스트는 의미 있는 반례와 실제 round-trip 위주로 작성한다.
- native backend가 없으면 hardware 검증을 not_run/skip로 보고한다. mock 통과로 supported로 승격하지 않는다.
- 결과 숫자를 만들지 않는다. latency/accuracy/memory는 실제 측정값과 조건을 함께 기록한다.
- 학습 loss 감소와 서비스 정확도 개선을 구분한다.
- stage 실패 시 diagnostic을 남기고 성공 artifact를 publish하지 않는다.
- TensorRT engine과 ONNX external data를 포함한 전체 bundle의 hash/호환성을 검증한다.
- 오래 걸리는 학습·다운로드는 실행 상태를 보여주고 불필요하게 반복하지 않는다.

## 작업 완료 시

- 변경한 파일과 공개 API를 정리한다.
- 실제 실행한 테스트와 미실행 검증을 구분한다.
- docs/status.md의 milestone/checklist와 capability evidence를 갱신한다.
- 필요한 다음 작업을 한두 개로 좁혀 기록한다.
- 사용자가 요청하지 않은 원격 배포·외부 게시·메시지 전송은 수행하지 않는다.
