# Codex 구현 계획

단계별로 작은 실제 실행 경로를 완성한다. 현재 M0 구현과 검증 상태는 [docs/status.md](docs/status.md)에 기록한다. M1 이후는 구현 완료로 표시하지 않는다. 목표 API와 YAML은 계약이며 설치 가능한 라이브러리가 아니다.

각 작업은 코드·관련 테스트·사용 예·검증 결과를 함께 끝낸다. 후속 단계는 선행 artifact/schema를 재사용한다. hardware가 없어 not_run인 단계는 supported로 승격하지 않는다.

## M0 — 프로젝트 골격·계약·환경

목표: 지원하지 않는 기능을 정확히 설명하는 core와 최소 fixture를 만든다.

| 작업 | 구현 | 완료 조건 |
|---|---|---|
| M0-1 | src layout, pyproject, lint/type/test, 최소 core import | torch/AIMET 없이 core import 가능 |
| M0-2 | ArtifactRef, spec schema, errors, capability registry | unknown field/type mismatch/중복 plugin 거부 |
| M0-3 | 로컬 artifact store, atomic publish, digest, lock | 손상·partial write·중복 writer 검증 |
| M0-4 | plan parser, DAG 참조·cycle 검사 | 유효 recipe 계획 생성, 잘못된 참조·cycle 실패 |
| M0-5 | doctor와 env profile, TinyCNN fixture | 실제 설치/CPU forward와 미검증 backend 구분 |

버전은 공식 metadata와 실제 install/import/forward 검증을 거쳐 profile별 lock으로 고정한다. 최신 문서의 예제를 과거 package에 그대로 적용하지 않는다. 이전 YOLO 패키지의 단위 테스트 통과만으로 AIMET 전체 호환성을 선언하지 않는다.

산출물: core, application/planner, 로컬 artifact store, schema fixtures, 최소 CLI, docs/status.md, docs/environment-matrix.md.

## M1 — 분류 모델 생성·학습·ONNX 실행

목표: TinyCNN/ResNet18/MobileNetV3에서 모델·데이터·학습·export 공통 계약을 검증한다.

| 작업 | 구현 | 완료 조건 |
|---|---|---|
| M1-1 | torchvision/custom provider, random/pretrained/local weight | 생성·head 교체·strict reload, key mismatch 탐지 |
| M1-2 | synthetic/CIFAR-10/ImageFolder data | split 재현·class mapping·중복/annotation 검증 |
| M1-3 | generic torch training engine | 작은 문제 학습, best/last 분리, epoch resume |
| M1-4 | Torch artifact와 Eager runtime | 새 process에서 원본 factory/recipe로 복원·추론 |
| M1-5 | FP ONNX exporter + ORT runtime | 고정 input/output contract, FP 수치 비교 |
| M1-6 | graph inspect + Netron wrapper | 저장된 ONNX 조회, viewer 시작/종료, graph report |

offline CI는 synthetic data와 random weights만 쓰고 download를 막는다. 실제 CIFAR-10/pretrained integration은 별도 marker로 실행한다. 정확도 실험과 wiring smoke를 구분한다.

완료 시연: 작은 분류 모델 생성 → 학습 → 저장 → 새 process reload → ONNX → ORT predict → graph view. CLI 종료 코드 외에 gradient/update/output 비교 증거가 필요하다.

## M2 — YOLO adapter와 기존 프로젝트 이전

목표: YOLOv8n이 같은 라이브러리 표면에서 학습·평가·ONNX를 지원한다.

- M2-1: 고정한 Ultralytics 버전의 create/load/train/export/evaluate adapter 구현.
- M2-2: YOLO YAML과 COCO8 provider, label/bbox/class/중복 검사.
- M2-3: TransformSpec, decoded pre-NMS output, NMS 후 Prediction 계약.
- M2-4: native trainer의 best/last/model/config를 공통 artifact로 변환.
- M2-5: PyTorch/ONNX raw output 및 동일 evaluator의 mAP 비교.

실제 COCO8 한 epoch와 FP ONNX round-trip이 통과해야 한다. COCO8 성능은 서비스 품질 지표로 쓰지 않는다. native trainer의 내부 모델 생성 방식을 검토하고 학습 가능한 artifact와 wrapper identity를 맞춘다. 아직 YOLO pruning/KD는 unsupported다.

## M3 — AIMET PTQ·Q/DQ·graph 비교

목표: 분류와 YOLO FP ONNX를 같은 QuantizationStrategy 계약으로 양자화한다.

- M3-1: 작은 Conv에서 실제 AIMET → encodings → 표준 Q/DQ → ORT 검증.
- M3-2: split과 전처리를 공유하는 calibration loader, 재사용 cache key.
- M3-3: AIMET adapter, signed/symmetric 등 target별 profile.
- M3-4: Q/DQ dtype/scale/zero-point/axis/custom-op/external-data 검사.
- M3-5: simulation/QDQ 수치 비교, FP/QDQ task metric 보고서.
- M3-6: graph diff/quantization summary와 Netron 연결.

완료 조건: 레이어별 quant config와 보정 목록을 기록하고, source model/transform이 바뀌면 이전 encoding 재사용을 거부한다. unsupported profile은 사전 검사나 구조 검사에서 명시적으로 실패한다. CPU Q/DQ 실행을 GPU INT8 가속 결과로 보고하지 않는다.

여기까지가 **첫 사용 가능한 baseline release**다. 생성·학습·PTQ·Torch/ONNX 실행·graph를 지원하며 pruning/KD/TensorRT는 아직 지원 범위에서 제외된다.

## M4 — 구조적 pruning과 복원

목표: TinyCNN/ResNet18을 구조적으로 줄이고 다시 학습·export한다.

- M4-1: importance와 budget을 명시한 PruningPlan.
- M4-2: Torch-Pruning dependency group 적용, residual/concat/depthwise fixture 검증.
- M4-3: ArchitectureRecipe/retained indices/shape 저장 및 새 process 복원.
- M4-4: pruned student의 supervised recovery와 새 optimizer 생성.
- M4-5: 학습된 pruned model의 ONNX/ORT round-trip, 실제 parameter/tensor-size 비교.

완료 조건: mask만 0으로 만든 경우와 실제 shape 축소를 구분하고 teacher/source는 변경하지 않는다. 원래 architecture에 축소 state_dict를 로드하는 버그를 테스트로 차단한다. 채널 20% 제거를 latency 20% 감소라고 보고하지 않는다.

## M5 — 분류 KD와 외부 objective plugin

목표: 작은 architecture student와 pruned student를 같은 objective 계약으로 학습한다.

- M5-1: CE + temperature KL objective, teacher frozen/eval.
- M5-2: feature taps와 student projection, objective parameters/state 관리.
- M5-3: 외부 entry-point plugin으로 새 KD objective 등록.
- M5-4: teacher mutation/gradient 누출/hook 누수/다른 class order 실패 테스트.
- M5-5: 동일 student 초기값·budget의 supervised-only와 KD 비교 recipe.

완료 조건: teacher weight와 BN buffers는 한 step 후 동일하고 student는 바뀐다. projection optimizer 등록과 resume 복원을 확인한다. KD가 항상 정확도를 개선해야 테스트가 통과한다는 조건은 두지 않는다.

## M6 — 연구용 quantization과 QAT 확장

목표: library core를 수정하지 않고 사용자 method를 추가한다.

- M6-1: observer/encoding estimator/quantizer/policy/optimization hook 분리.
- M6-2: minmax와 자작 affine range estimator 예제, registry 추가만으로 실행.
- M6-3: fake-quant/STE를 지원하는 QAT engine hooks와 observer freeze schedule.
- M6-4: QAT + KD의 순서·gradient·state resume 검증.
- M6-5: standard Q/DQ lowering이 가능한 method의 export round-trip.
- M6-6: 비표준 quantizer의 research_only 상태·export 거부·후속 kernel 계약.

완료 조건: plugin이 필요한 labels/gradient/weight mutation/export 능력을 선언한다. PTQ라는 이름만으로 gradient를 일괄 금지하지 않는다. 연구 simulation의 값과 gradient를 작은 수치 예제로 검증하고 baseline ablation을 제공한다.

## M7 — TensorRT와 local deployment bundle

목표: 모델 파일 저장에 그치지 않고 대상 runtime에서 실행한다.

- M7-1: RuntimeProfile/BuildSpec, TensorRT/CUDA/GPU 호환성 검사.
- M7-2: FP 및 Q/DQ ONNX의 TensorRT build, build log/operator report.
- M7-3: TensorRT runner 및 같은 pre/postprocess의 predict API.
- M7-4: 실제 engine과 FP/ORT 출력·task metric 비교.
- M7-5: warmup/synchronization/latency/memory 측정 및 bundle 생성.
- M7-6: PyTorch/ORT/TRT bundle을 각각 fresh process에서 load/run.

GPU integration job이 필요하다. hardware 부족으로 skip한 job은 완료 증거가 아니다. 호환 profile cache key가 다르면 engine을 재사용하지 않는다. target builder를 local training env와 분리할 수 있어야 한다.

M4/M5/M7까지 완료하면 **분류 기준 요구 기능 전체**가 연결된다. YOLO 기본 학습·PTQ·배포도 포함되지만 YOLO 구조적 pruning/KD는 M8의 별도 검증을 요구한다.

## M8 — YOLO 연구용 pruning/KD

목표: task별 연구 adapter를 구축한다.

- M8-1: native trainer가 pruned topology를 원복하지 않는 module injection/restore 경로.
- M8-2: 보호할 detection head와 dependency 기준, 축소 architecture serializer.
- M8-3: 같은 YOLO family의 teacher/student raw outputs와 feature taps.
- M8-4: class/box/feature distillation, foreground/geometry/stride alignment.
- M8-5: pruned → recovery/KD → PTQ → ONNX/TRT 전체 검증.

완료 조건: detection loss와 assignment가 정상이며 forward/backward가 통과한다. class order·box 좌표·feature stride 불일치를 거부한다. pruning 후 모델 구조가 저장/로드 후에도 유지된다. 다른 YOLO 세대나 Faster R-CNN과의 KD는 자동 지원하지 않는다.

## M9 — 문서화·연구 재현·운영 연결

- capability matrix를 실제 CI 증거와 연결하고 supported 조합만 문서화한다.
- custom plugin template와 recipe gallery, graph report 사용법을 정리한다.
- real dataset/복수 seed ablation과 hardware benchmark 보고서를 작성한다.
- MLflow/remote artifact store는 optional tracking adapter로 추가한다.
- FastAPI job service, field feedback, 재학습, canary/rollback은 이후 별도 scope로 개발한다.
- ExecuTorch/ARM/NPU는 target별 plugin과 hardware acceptance를 추가한다.

## 테스트 계층과 완료 정의

| 계층 | 실행 위치 | 필수 검증 |
|---|---|---|
| unit | backend 없는 CPU CI | schemas, DAG, hashing, state transition, capability |
| contracts | CPU + 최소 torch/onnx | 모델 복원, tensor 계약, gradient, preprocessing, checkpoint |
| integration | profile별 Docker | 실제 torchvision/YOLO/AIMET/ORT, download 허용 marker 분리 |
| hardware | 지정 NVIDIA/edge | TensorRT build/run, latency, peak memory, compatibility |
| scientific | 고정 dataset/hardware | accuracy·ablation·반복 실험 |

공통 완료 조건은 구현, 반례 테스트, 직접 실행한 예시, 공개 API 문서, source/config/env/asset hash, 실제 검증 상태다. fake output·shape만 검사하는 mock·무조건 성공하는 stub은 native integration의 완료 증거가 아니다. 의미 없는 구현 복제 테스트를 늘리지 않는다.

## Codex 작업 크기

숫자 순서는 권장 진행 순서이며 모든 단계가 바로 앞 단계에 의존하는 것은 아니다. M4는 M1, M5의 독립 student KD는 M1, M7은 M3와 대상 GPU 환경만 있어도 시작할 수 있다. pruned student KD에는 M4가 필요하고, QAT+KD에는 M5/M6의 계약이 필요하다. M8의 YOLO 연구 통합에는 M2/M4/M5를 먼저 검증한다. 초기에는 한 흐름씩 진행해 공통 계약의 변경을 줄인다.

한 작업에는 하나의 operation 또는 adapter를 완성한다. 예를 들어 M3 전체 대신 M3-1의 작은 Conv AIMET Q/DQ round-trip부터 수행한다. API 설계 변경이 필요하면 문서·예시·테스트를 같이 수정한다. 요청 범위에서 필요한 작업은 진행하되 미검증 후속 기능을 한 번에 채우지 않는다.

작업 결과에는 문제와 결과 동작, 변경한 계약, 실행한 검증, not_run 이유를 남긴다. 플랫폼 전체를 완성했다고 주장하기보다 어떤 model/task/backend 조합이 새로 검증됐는지 정확히 표시한다.
