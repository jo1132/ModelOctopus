# API와 plugin 계약 초안

이 문서는 전체 API의 **구현 목표**다. M0에서 실행 가능한 schema/store/registry/plan/doctor 계약은 [M0 API](docs/m0-api.md)에 고정했다. 아래 Project와 M1 이후 service는 아직 구현하지 않았다. 이후 변경은 migration과 테스트를 함께 추가한다.

## 1. 사용자가 보는 API

공개 API는 artifact를 입력받고 새로운 artifact를 반환한다. 모델을 실제로 실행하는 worker 내부에서만 live `nn.Module`을 materialize한다. 서로 다른 env의 Python 객체를 pickle로 전달하지 않는다.

```python
# 설계 예시: 현재 실행 가능한 코드가 아니다.
from modeloctopus import Project, ModelSpec, DatasetSpec, TrainSpec, QuantizationSpec

project = Project(root="./workspace")
data = project.datasets.prepare(DatasetSpec(
    provider="torchvision", name="cifar10",
    split_policy="official_train_90_10_plus_official_test", seed=42,
))
model = project.models.create(ModelSpec(
    provider="torchvision", architecture="resnet18", task="classification",
    num_classes=10, initialization={"kind": "random"},
))
trained = project.training.fit(model=model, dataset=data, spec=TrainSpec(
    engine="torch", mode="scratch", epochs=20, batch_size=64,
))
fp = project.exports.create(model=trained.model, format="onnx", target="ort_cpu_fp32")
quantized = project.compression.quantize(
    source=fp.artifact, dataset=data,
    spec=QuantizationSpec(plugin="aimet_onnx.ptq", split="calibration",
                          target="trt_int8_symmetric", weight_bits=8, activation_bits=8),
)
report = project.evaluation.compare(
    reference=fp.artifact, candidate=quantized.artifact,
    dataset=data, split="val", runtime="onnxruntime",
)
graph = project.graphs.view(quantized.artifact)  # ViewerHandle(url, close)
bundle = project.deployment.package(quantized.artifact, runtime="onnxruntime")
runner = project.runtimes.load(bundle)
predictions = runner.predict(["example.jpg"])
```

`Project`는 facade다. 도메인 서비스는 별도 class로 테스트할 수 있어야 한다. 위 target은 policy id이며 실제 TensorRT 빌드 완료를 의미하지 않는다. ONNX checker·ORT validation·TensorRT target validation은 별개 report다.

학생 모델과 KD:

```python
student = project.models.create(ModelSpec(
    provider="torchvision", architecture="mobilenet_v3_small",
    task="classification", num_classes=10, initialization={"kind": "random"},
))
kd = project.training.fit(
    model=student, dataset=data, teacher=trained.model,
    spec=TrainSpec(engine="torch", mode="scratch", epochs=20,
                   objective={"plugin": "classification.kl_kd", "temperature": 4.0,
                              "hard_weight": 0.5, "soft_weight": 0.5}),
)

pruned = project.compression.prune(
    source=trained.model, dataset=data,
    spec={"plugin": "torch_pruning.channels", "channel_ratio": 0.2,
          "preserve_task_head": True},
)
recovered = project.training.fit(
    model=pruned.model, dataset=data, teacher=trained.model,
    spec=TrainSpec(engine="torch", mode="finetune", epochs=10,
                   objective={"plugin": "classification.kl_kd", "temperature": 4.0,
                              "hard_weight": 0.5, "soft_weight": 0.5}),
)
```

`channel_ratio`는 plugin이 정의하는 목표 채널 비율이다. FLOPs·파라미터·속도 감소율과 같지 않다. plan에는 채널 수 반올림과 의존성을 적용한 실제 변경을 기록한다. `pruned.model`은 축소 구조 recipe와 weight를 포함하며 원본 `trained.model`은 변경하지 않는다.

## 2. ArtifactRef와 주요 spec

Core는 backend에 의존하는 tensor 타입을 갖지 않는다. manifest는 JSON 호환이며 schema_version을 필수로 둔다.

| 타입 | 필수 필드와 의미 |
|---|---|
| ArtifactRef | id, kind, digest. 저장 위치 URI는 store가 해석 |
| ModelSpec | provider/revision, architecture, task, initialization, head config, input contract, serializer |
| InitializationSpec | random/pretrained/weights/resume_ref, source ref, strict/head policy |
| ArchitectureRecipe | base factory/revision, architecture kwargs, 순서가 있는 structural edits, 예상 tensor shapes |
| DatasetSpec | provider/name/revision, source, task, split policy, download/cache policy |
| DatasetManifest | class_map, sample 목록, splits/grouping, content digests, source/license metadata |
| TransformSpec | version, color/layout/dtype/range, resize/pad/normalize, random seed, geometry mapping |
| TrainSpec | engine, mode, optimizer/scheduler, epochs, batch, AMP, seed, objective/QAT specs, resume scope |
| QuantizationSpec | strategy/plugin revision, precision policy, observer/estimator, rounding, 제외 노드, target profile |
| PruningSpec | strategy/plugin revision, 목표 ratio/budget 종류, importance, 보호 module, 채널 정렬 조건 |
| DistillationSpec | teacher ref, objective plugin, weights/temperature, feature taps, alignment, cache policy |
| ExportSpec | format, opset/exporter profile, shape constraints, pre/postprocess 경계, target profile |
| RuntimeProfile | backend/version, EP priority, fallback policy, device, shape profiles, precision capabilities |
| EvaluationSpec | evaluator revision, split, metric protocol, tolerances, reference, shape/batch, 측정 조건 |

unknown config key는 오류다. weights와 resume, random 초기화와 local checkpoint처럼 모순되는 지정을 거부한다. 상대 경로는 recipe 파일의 부모를 기준으로 해석하고, 실행 기록에는 해석된 경로를 남긴다. 데이터셋 내부 경로는 dataset root 기준이다. artifact identity는 절대 경로에 의존하지 않는다.

## 3. 입출력과 task 계약

### Classification

- Batch: images `[N,C,H,W]`, targets `[N]` int64, sample IDs, TransformRecord.
- ForwardOutput: logits `[N,K]`, 선택적인 named features.
- Prediction: class IDs/probabilities. class 순서는 manifest에 고정한다.
- Loss: hard-label CE 또는 등록한 objective.
- Metrics: top1/top5, 클래스별 지표. K가 5보다 작으면 실제 top-k의 k를 명시한다.

### Detection

- 논리 Batch: images, sample별 boxes `[Mi,4]`, labels `[Mi]`, image IDs, original sizes, geometry.
- 내부 Batch: task adapter가 native engine에 맞게 변환한다. 모든 detector에 같은 raw head shape를 요구하지 않는다.
- ForwardOutput: adapter version이 있는 raw head, 선택적인 feature maps, decode 전후를 명시한 prediction view.
- 외부 Prediction: 원본 이미지 좌표의 xyxy, score, class id. 빈 검출을 허용한다.
- ExportOutput: 고정한 decoded pre-NMS tensor 또는 NMS가 포함된 출력. 둘을 명확히 구분한다.
- Loss: detector 고유 assignment, box/class/DFL 등을 task adapter가 담당한다.
- Metrics: bbox 형식, category mapping, IoU, NMS를 포함한 평가 protocol을 고정한다.

`ForwardOutput.features`는 semantic key(예: backbone.stage3), layout, stride, module provenance를 갖는다. pruning/export로 module 이름이 바뀌면 feature mapping을 다시 검증한다.

## 4. Plugin 단위

| Protocol | 최소 계약 |
|---|---|
| ModelProvider | list_models, resolve_weights, create, import_checkpoint |
| ModelAdapter | export_view, feature_views, describe_architecture, rebuild, task_binding |
| TaskAdapter | collate/validate, task_loss, decode, metric_adapter, distillation_views |
| DatasetProvider | resolve/download, index, build_manifest |
| DataModule | train/val/test/calibration loader, transform revision, split 검증 |
| TrainingEngine | capabilities, fit, save_training_state, restore_training_state |
| PruningStrategy | validate, plan, apply, architecture export, 변경 report |
| QuantizationStrategy | validate, prepare, calibrate_or_optimize, finalize, simulation report, export capability |
| DistillationObjective | requirements, setup, forward→LossBundle, trainable_parameters, state save/load, teardown |
| Exporter | can_export, export, inspect, reference comparison |
| RuntimeBackend | can_load, load, infer_tensor, synchronize, close |
| GraphInspector | inspect, diff, view→ViewerHandle |

모든 plugin은 id/API version/config schema/capabilities를 가진다. discovery만으로 heavy backend를 import하지 않고 선택한 worker에서 lazy load한다. API version 불일치와 중복 id는 오류다.

외부 plugin은 Python entry points로 등록한다. group 후보는 `modeloctopus.models`, `modeloctopus.datasets`, `modeloctopus.quantization`, `modeloctopus.pruning`, `modeloctopus.distillation`, `modeloctopus.runtimes`다. YAML의 임의 Python import 문자열을 실행하지 않고 등록된 id를 해석한다.

### ModelAdapter 복원 계약

1. describe_architecture(module)은 재생성 가능한 ArchitectureRecipe를 반환한다.
2. rebuild(recipe)는 새 process에 같은 parameter/buffer 구조를 만든다.
3. state_dict를 strict load하며 missing/unexpected key가 없어야 한다.
4. reference inputs에서 원본과 출력을 비교한다.
5. custom module 정의가 필요하면 plugin distribution/version을 요구한다.

임의 Python forward를 JSON만으로 완전히 저장할 수 있다고 가정하지 않는다. rebuild를 제공하지 않는 모델은 메모리 안의 연구 또는 지원하는 export format으로 사용 범위를 제한한다.

### QuantizationStrategy 실행 계약

```python
# Protocol 설명용. 구체적인 구현은 backend plugin에 둔다.
class QuantizationStrategy:
    def validate(self, source, spec, target) -> "CapabilityReport": ...
    def prepare(self, source, spec, context) -> "PreparedQuantization": ...
    def calibrate_or_optimize(self, prepared, data, context) -> "QuantizationState": ...
    def finalize(self, prepared, state, context) -> "QuantizationResult": ...
```

PreparedQuantization과 backend tensor는 worker 안의 임시 객체다. 저장할 state는 schema를 가진다. QAT는 prepare 이후 학습 가능한 student와 TrainingHooks를 engine에 넘기고, 학습 완료 후 finalize한다. requires_grad, requires_labels, mutates_weights, supports_qat, export_formats를 각각 선언한다.

QuantizationResult에는 artifact, source ref, config/encoding refs, simulation 비교, graph provenance, exportability를 포함한다. research_only 결과를 native artifact인 것처럼 반환하지 않는다.

### DistillationObjective 실행 계약

setup(student_adapter, teacher_adapter, spec)에서 feature tap과 학습용 alignment module을 준비한다. engine은 해당 parameters를 optimizer에 포함한다. teacher는 eval/frozen이고 forward의 gradient는 끈다. student와 필요한 projection은 gradient를 유지한다. teacher의 BN running statistics도 바뀌지 않아야 한다.

forward(student_output, teacher_output, targets, context)는 LossBundle(total tensor, 이름별 component tensors, detached logging metrics)을 반환한다. objective가 optimizer.step을 직접 호출하지 않는다. AMP·gradient accumulation·분산 실행의 제어는 engine이 맡으며 초기 구현은 single-device로 제한한다.

resume 시 objective state를 복원한다. 종료 시 hooks를 제거하고 batch 사이에 teacher tensors를 계속 보관하지 않는다. 양자화와 feature hook의 적용 순서는 setup에서 결정한다.

## 5. Result 타입

| operation | result |
|---|---|
| datasets.prepare | Dataset ArtifactRef |
| models.create/import_weights | trainable Model ArtifactRef |
| training.fit | TrainResult(model, training_state, metrics, run_id) |
| compression.prune | PruningResult(model, plan, report) |
| compression.quantize | QuantizationResult(artifact, encodings, report, exportability) |
| exports.create | ExportResult(artifact, validation) |
| deployment.build | BuildResult(artifact, target_profile, validation) |
| deployment.package | Bundle ArtifactRef |
| evaluation.compare | EvaluationReport ArtifactRef |
| graphs.inspect | GraphReport ArtifactRef |
| graphs.view | ViewerHandle(url, server handle, close) |

TrainResult.model은 평가로 선택한 weight를 가진다. training_state는 재개용 상태다. best model과 마지막 epoch의 optimizer state를 섞지 않는다. best/last를 별도 ref로 기록하고 각각의 epoch를 명시한다.

## 6. Capability negotiation

모델 provider가 YOLO를 생성할 수 있어도, engine에 feature tap이 없으면 custom feature KD를 거부한다. planner는 필요한 features, task/class map, 입력 geometry, export format의 조합을 검사한다.

CapabilityReport는 status, reasons, 필요한 extras/env profile, tested versions, evidence refs를 가진다. 대표 reason code는 다음과 같다.

- MODEL_TASK_MISMATCH
- HEAD_CLASS_MAPPING_MISMATCH
- TRAINING_STATE_INCOMPATIBLE
- PRUNED_ARCHITECTURE_NOT_RECONSTRUCTABLE
- FEATURE_ALIGNMENT_UNDEFINED
- QUANTIZER_NO_TARGET_LOWERING
- UNSUPPORTED_OPERATOR_OR_OPSET
- RUNTIME_PROFILE_UNVERIFIED
- CALIBRATION_EVALUATION_OVERLAP
- DEPENDENCY_NOT_INSTALLED

CPU/GPU 미검증, 미지원 bitwidth/KD를 성공으로 표시하지 않는다. fallback은 사용자가 policy로 지정한 경우만 허용하며 실제 provider/precision을 기록한다.

## 7. CLI와 recipe DAG

구현 목표 CLI:

```bash
ef doctor --profile train-cpu
ef models list --task classification
ef datasets list
ef plan examples/classification_kd_ptq.yaml
ef run examples/classification_kd_ptq.yaml
ef artifacts inspect ARTIFACT_ID
ef evaluate --reference REF_ID --candidate CANDIDATE_ID --dataset DATA_ID --split val
ef graph inspect ONNX_ARTIFACT_ID
ef graph view ONNX_ARTIFACT_ID --host 127.0.0.1 --port 8080
ef graph diff FP_ARTIFACT_ID QDQ_ARTIFACT_ID
ef build ONNX_ARTIFACT_ID --backend tensorrt --profile local-nvidia
```

recipe에는 schema_version/project/environment_profiles/steps를 둔다. step은 id, kind, inputs, params, profile을 가진다. `@step_id.output_name`으로 선행 결과를 참조한다. plan은 구문·참조·cycle·타입·능력을 검사하며 학습이나 큰 download는 수행하지 않는다. 실제 모델 구조나 hardware가 필요한 검사는 deferred로 기록하고 해당 stage 전에 수행한다.

output 이름은 다음으로 고정한다.

| kind | outputs |
|---|---|
| model.create | model |
| dataset.prepare | dataset |
| train.fit | model, training_state |
| export.create | artifact |
| compression.quantize | artifact |
| deployment.build | artifact |
| compression.prune | model, plan |
| evaluation.compare | report |
| graph.inspect | report |

입출력 kind가 맞지 않는 연결을 plan에서 거부한다. Python API와 CLI는 같은 request와 service를 사용한다. 최초 실행기는 순차 DAG로 충분하며 자동 병렬 실행과 cluster scheduler는 후속 범위다.

## 8. 실패·재개·캐시

정상 artifact는 성공했을 때만 publish한다. 검증 실패 시 diagnostic과 실패 report를 보존한다. 검증 전 결과를 보관하려면 candidate 상태를 ready 상태와 구분한다.

job은 run ID와 stage ID로 식별하고 중복 writer를 lock한다. manifest를 원자적으로 갱신한다. crash 당시의 staging을 완성된 artifact와 구분한다. 재실행은 입력·설정·plugin·환경 digest가 같은 성공 artifact만 명시적으로 재사용한다.

학습이 정상 종료했지만 accuracy가 기준에 미달하면 run 성공과 품질 gate 불합격을 별도로 기록한다. 구문 오류·shape 불일치·NaN·export 불가능은 operation 실패다. GPU가 없어 검증하지 못했다면 not_run이다. 미측정 benchmark는 null과 이유를 반환하며 0초를 넣지 않는다.
