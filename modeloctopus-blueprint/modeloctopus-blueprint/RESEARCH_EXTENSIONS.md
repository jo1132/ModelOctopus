# 사용자 Quantization·KD 방법을 추가하는 설계

다음 내용은 확장 설계이며 새 연구 방법의 구현·성능 개선·신규성을 주장하지 않는다. 기존 방법의 재현과 사용자가 제안하는 변경을 구분해 기록한다.

## 1. 연구 변경점을 작게 유지하기

| 바꾸려는 것 | 확장 지점 | 함께 검증할 것 |
|---|---|---|
| activation range 결정 | Observer/EncodingEstimator | clipping, scale, 대표 데이터 |
| tensor별 bitwidth | QuantizationPolicy | exporter/runtime이 지원하는 bitwidth |
| 기본 반올림 대체 | RoundingStrategy/optimization pass | 최종 weight/encoding과 simulation 일치 |
| 이전 레이어 오차 반영 | Calibration/OptimizationStrategy | 참조 입력, 레이어 순서, 오차 전파 |
| vector/codebook 표현 | Quantizer + serializer + lowering | tensor 표현, kernel, runtime 실행 |
| teacher 정보 사용 방식 | DistillationObjective | teacher freeze, class/feature 정렬 |
| 중간층 증류 대상 | FeatureSelector/FeatureAligner | shape, stride, geometry, projection |
| 양자화와 KD 공동 최적화 | QAT strategy + objective + hooks | observer schedule, STE, exportability |

작은 변경은 해당 hook을 교체한다. 여러 레이어를 순차 최적화하는 등 상태를 가진 방법은 QuantizationStrategy 전체를 구현한다. 모든 알고리즘을 observer 하나에 넣지 않는다.

## 2. Affine quantization 연구 baseline

기본 표현은 `q = clip(round(x / scale) + zero_point, qmin, qmax)`와 `x_hat = scale * (q - zero_point)`다. scale, zero-point, dtype, axis, rounding, clipping 범위를 QuantizationState에 저장한다.

첫 비교는 같은 모델·데이터·bitwidth에서 minmax와 사용자 range estimator만 바꾸는 것이다. AIMET baseline과 자체 simulation의 차이를 기록한다. 표준 Q/DQ로 export할 수 있다면 동일 입력에서 simulation과 ORT 출력을 비교한다.

```python
# 설계 예시. M6에서 구현할 plugin metadata.
descriptor = {
    "id": "myresearch.quant.error_aware_affine",
    "api_version": 1,
    "algorithm_version": "0.1.0",
    "tasks": ["classification"],
    "source_kinds": ["trainable_torch"],
    "requires_grad": True,
    "requires_labels": False,
    "mutates_weights": True,
    "export_formats": ["onnx_qdq"],
    "status": "experimental",
}
```

PTQ에서도 gradient 기반 최적화를 사용할 수 있다. 일반적인 task-label QAT와 라벨 없는 재구성 손실 PTQ를 metadata로 구분한다. export_formats 선언만으로 supported가 되는 것은 아니며 round-trip 검증이 필요하다.

source 모델을 복사해서 materialize한 뒤 최적화한다. FP 참조 모델, 양자화 모델 입력, teacher target, 레이어별 error metric과 최적화 budget을 기록한다. backend 전용 custom op가 표준 ONNX에 남으면 export 검사를 통과시키지 않는다.

## 3. 새로운 수치 표현을 만들 때

nonuniform/codebook quantization은 하나의 scale과 zero-point만으로 표현할 수 없을 수 있다. 연구 수식과 bit 수를 실제 packed storage/kernel 구현과 구분한다.

1. simulation_only: 실수 연산으로 근사를 재현하고 accuracy를 평가한다.
2. exportable: ONNX 표준 연산으로 분해하거나 custom op schema를 만들어 저장·실행할 수 있다.
3. native_supported: 대상 runtime kernel이 있으며 실제 memory/latency와 accuracy를 검증했다.

표준 연산으로 분해해 실행할 수 있어도 원래 FP 연산보다 빠르다는 뜻은 아니다. packed weight 크기, 펼치는 데 필요한 workspace, 메모리 전송, kernel 선택을 측정한다. 아직 없는 kernel의 예상 속도를 측정값으로 기록하지 않는다.

## 4. Classification KD baseline

첫 objective는 `L = alpha * CE(student_logits, labels) + beta * T^2 * KL(p_teacher(T) || p_student(T))`다. alpha/beta를 명시하고 KL reduction은 batchmean으로 고정한다. teacher/student의 class 순서가 같아야 한다.

```python
# 손실 계산 예시. plugin 등록과 training engine은 별도로 구현한다.
import torch.nn.functional as F

def classification_kd(student_logits, teacher_logits, labels, temperature, alpha, beta):
    hard = F.cross_entropy(student_logits, labels)
    soft = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits.detach() / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2
    return alpha * hard + beta * soft, {"hard_ce": hard, "soft_kl": soft}
```

temperature > 0, class 수 일치, 유한한 출력, teacher gradient 없음, student gradient 존재를 검사한다. teacher를 eval/frozen으로 실행해 BN 통계도 보존한다. KD 유무를 비교할 때는 같은 초기 student·optimizer·학습 budget을 사용한다.

## 5. Feature KD와 YOLO KD

teacher/student의 feature C/H/W나 stride가 다르면 정렬이 필요하다. student 측 1×1 projection, resampling, foreground mask의 의미를 objective spec에 적는다. 학습 가능한 projection은 optimizer/checkpoint/resume에 포함한다. feature tensor를 매 batch 영구 보관하지 않는다.

YOLO에서는 기존 task loss와 다음 후보를 함께 실험한다.

| 항목 | 초기 방침 |
|---|---|
| class distillation | detector score 의미에 맞는 loss. 분류 softmax KL을 자동 사용하지 않음 |
| box distillation | 같은 좌표계와 대응하는 foreground 위치에서 비교 |
| multi-scale features | stride별 mapping, C/H/W 정렬 |
| distribution regression | bin 및 parameterization이 맞을 때만 비교 |
| task supervision | GT 기반 detector 고유 loss 유지 |

첫 구현은 같은 family의 teacher/student로 제한한다. NMS 후 서로 다른 개수의 검출을 단순 MSE로 비교하지 않는다. pseudo-label 학습과 raw-logit/feature KD는 별도 strategy다. mosaic 같은 증강에서는 teacher/student view의 대응을 명시한다.

## 6. QAT + KD의 hook 순서

1. student와 teacher를 서로 다른 artifact에서 복원한다.
2. student에 fake-quant/observer를 준비한다.
3. semantic feature taps와 alignment를 설정한다.
4. optimizer에 student, projection, 학습 가능한 quantizer parameters를 등록한다.
5. teacher를 freeze/eval로 두고 no_grad로 forward한다.
6. student forward → task loss + KD loss → backward/update를 수행한다.
7. 정해진 schedule에 따라 observer/range/BN을 freeze한다.
8. training state, QAT state, objective state를 저장한다.
9. 추론용 view를 만들고 지원하는 lowering으로 export한 뒤 수치·accuracy를 재검증한다.

round 연산을 넣으면 gradient가 저절로 흐른다고 생각하지 않는다. STE/gradient 근사와 학습 range의 규칙을 plugin에 명시하고 검사한다. 처음 teacher는 FP32로 사용하며 양자화 teacher는 별도 실험으로 추가한다.

## 7. 비교 실험

최소 ablation은 다음 5개다. 수치는 실제 실행 전까지 비워둔다.

| 실험 | student | 학습/처리 | 최종 정밀도 | 비교 목적 |
|---|---|---|---|---|
| A | 고정된 작은 구조 | supervised-only | FP32 | 작은 모델 baseline |
| B | A와 동일 | baseline KD | FP32 | KD의 기여 |
| C | B와 동일 | B의 weight에 baseline PTQ | INT8 | 기존 양자화 손실 |
| D | C와 동일 | B의 weight에 사용자 quant method | C와 동일 | quant method의 기여 |
| E | A와 같은 초기값 | 사용자 KD 후 C와 같은 PTQ | INT8 | KD method의 기여 |

pruning을 추가하면 pruning plan도 고정한 비교 행을 만든다. QAT에서는 추가 학습 budget의 효과를 별도로 비교한다. 가능하면 여러 seed의 평균과 변동을 보고한다. calibration·val·final test를 나누고 test로 알고리즘이나 hyperparameter를 선택하지 않는다.

accuracy, artifact bytes, parameter 수, 이론 MACs, 실측 p50/p95, peak memory, compile 시간을 구분한다. AIMET simulator 시간을 native INT8 성능으로 보고하지 않는다. teacher 계산 비용도 학습 비용에 포함한다.

## 8. Plugin 추가 완료 조건

- metadata/config schema/capability report가 있다.
- 작은 결정적 fixture로 값과 gradient를 검사했다.
- 입력 artifact와 teacher가 변하지 않는다.
- 저장 후 로드해 같은 encoding/objective state를 복원한다.
- export 지원을 선언했다면 실제 runtime round-trip이 통과한다.
- 미지원 task/shape/target은 이유와 함께 실패한다.
- baseline과 같은 budget으로 비교할 recipe가 있다.
- supported와 experimental을 구분한다.
