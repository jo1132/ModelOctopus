# M0에서 실행 가능한 API

패키지 및 import 이름은 `modeloctopus`, 표시 이름은 ModelOctopus다. CLI는 `ef`와 `modeloctopus`를 제공한다. 원래 `src/ModelOctopus/` 표기는 Python 패키지 관례에 맞춰 소문자로 정리했다. `API_CONTRACTS.md`의 Project facade는 M1 이후 목표이며 M0에서 제공하지 않는다.

```bash
python -m pip install -r configs/core-ci.lock
python -m pip install -e . --no-deps
pytest -q
ruff check src tests examples/m0_roundtrip.py
ruff format --check src tests examples/m0_roundtrip.py
mypy
ef doctor --profile core-ci
ef plan examples/classification_kd_ptq.yaml
```

CPU fixture가 필요한 환경에서는 `configs/train-cpu.lock`을 설치하고 `python examples/m0_roundtrip.py`를 실행한다. 큰 데이터나 pretrained weight 다운로드는 없다. lock은 현재 Linux/Python 조합의 관측값이며 다른 플랫폼 보장이 아니다.

```python
from modeloctopus import LocalArtifactStore, ModelSpec

spec = ModelSpec(provider="custom", architecture="tinycnn", task="classification", num_classes=3)
store = LocalArtifactStore("./workspace")
ref = store.publish("dataset", {"index.json": b"{}"}, {"revision": "fixture-v1"})
assert store.read(ref, "index.json") == b"{}"
store.verify(ref)
```

- `ModelSpec`, `DatasetSpec`, `TrainSpec`, `InitializationSpec`, `ArtifactRef`, `ArchitectureRecipe`, `CapabilityReport`는 strict Pydantic schema다. 알 수 없는 필드와 문자열→정수 강제 변환을 거부한다. JSON schema는 `docs/schemas/`에 있다. schema_version 1을 사용하고 recipe와 저장된 manifest에는 버전을 반드시 기록한다.
- `LocalArtifactStore.publish(kind, files, metadata)`는 상대 파일 이름→bytes mapping을 받는다. canonical JSON manifest와 모든 파일의 SHA-256으로 content identity를 만든다. 절대 저장 경로는 digest에 포함되지 않는다. 동일 내용 재게시는 검증 후 같은 ref를 반환한다. 다른 내용은 새 ref다.
- manifest와 실제 파일 목록·내용을 모두 검증한다. 외부 tensor 파일도 mapping에 포함해야 한다. metadata에는 producer가 source/config/env/plugin/parent provenance를 기록해야 한다. M0 store가 모델별 provenance의 완전성을 판정하지는 않는다.
- Linux `flock`으로 같은 artifact의 process writer를 직렬화한다. staging에서 파일·디렉터리 fsync, 검증, rename 후 artifacts 디렉터리를 fsync한다. 실패 staging은 diagnostics로 이동한다. process 강제 종료 후 남은 staging은 게시되지 않으며 자동 재사용하지 않는다. 여러 호스트/NFS·Windows 보장은 없다. 로컬 store 디렉터리는 신뢰된 writer만 사용한다.
- `Registry.register(PluginDescriptor(...), loader)`는 중복 id/API 버전 불일치를 거부한다. `load(id, config)`는 descriptor의 config schema 검증 뒤 backend loader를 한 번 호출한다. 없는 backend는 `DEPENDENCY_NOT_INSTALLED` 오류다.
- 외부 entry point group은 `modeloctopus.models`, `.datasets`, `.quantization`, `.pruning`, `.distillation`, `.runtimes`다. entry point는 `(PluginDescriptor, backend_loader)`를 반환하는 **가벼운 metadata factory**다. metadata 모듈은 backend를 import하지 않아야 한다. 임의로 설치된 plugin 코드의 import 부작용까지 격리하는 sandbox는 제공하지 않는다. recipe는 Python import 문자열을 실행하지 않는다.
- `application.planner.plan(path)`와 `ef plan`은 같은 구현을 호출한다. schema·중복 YAML key/id·알 수 없는 profile/operation/input·참조/output·cycle·artifact kind를 검사한다. 알려진 task와 teacher/student class 수 불일치를 거부한다. recipe 상대 경로는 recipe 부모 기준이다. 실행 순서는 위상 정렬한다.
- `deployment.build` 입력 이름은 `source`, resume 입력은 `training_state`로 고정한다. resume에서 training_state는 필수이며 다른 mode에서는 금지한다. export format에 따라 output kind를 구분한다.
- plan 성공은 정적 계획 성공이다. 모든 실행 operation은 `planned`, `executable: false`이며 backend·실제 class mapping·모델 구조 검증은 deferred다. M0에는 `run`, 학습·압축·export·deployment service가 없다.
- `doctor(profile)`는 metadata와 도구 유무를 보고한다. train-cpu/train-cuda는 별도 Python process에서 TinyCNN 실제 forward를 실행한다. core-ci는 core 실행 가능 여부, train-cpu의 experimental은 fixture 검증만 의미한다. quant-aimet/deploy-tensorrt는 package가 있어도 integration은 not_run이다. CLI exit 0은 보고서 생성 성공이며 validation 상태를 함께 읽어야 한다. 잘못된 요청은 2, 실행 검증 실패는 1이다.
- `models.tinycnn`은 선택적 torch fixture 모듈이다. architecture recipe 재구성 + strict state_dict 로드를 검증하며 전체 nn.Module pickle을 저장하지 않는다. 모델 서비스나 학습 engine을 대신하지 않는다.

공식 API 확인: [Pydantic strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/), [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html). 현재 설치 버전에서 반례·forward·round-trip을 실제 확인했으며 최신 문서만으로 다른 버전의 호환성을 주장하지 않는다.
