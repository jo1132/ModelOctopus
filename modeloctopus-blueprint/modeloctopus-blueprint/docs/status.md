# ModelOctopus 구현 상태

2026-10-29 (Asia/Seoul). 이번 범위는 **M0만**이다. M1~M9 실행 기능은 planned/unsupported이며 supported로 승격하지 않았다.

| 항목 | 상태 | 구현·근거 |
|---|---|---|
| M0-1 | 완료 | src layout, pyproject, core/dev 및 CPU fixture lock, CLI entry points, lint/format/type/test 설정. torch/AIMET 없는 독립 venv에서 core import·테스트 통과 |
| M0-2 | 완료 | ArtifactRef/ModelSpec/DatasetSpec/TrainSpec/InitializationSpec/ArchitectureRecipe/capability/error schema. strict 타입·unknown field·잘못된 초기화·중복 plugin/API 버전 반례 검증. JSON schema fixtures 제공 |
| M0-3 | 완료 | canonical manifest 및 전체 파일 SHA-256, staging 검증→atomic rename, fsync, Linux process lock. 손상·manifest 변조·누락/추가 파일·symlink·partial write·6회 중복 process writer 테스트 |
| M0-4 | 완료 | YAML 중복 key/step·unknown config/input/profile·참조/output·cycle·입출력 kind·기본 task/class 수·resume 검사. 위상 정렬과 경로 해석. 제공된 두 recipe가 정적 plan을 생성하며 모든 실행 단계는 planned |
| M0-5 | 완료 | doctor profile과 환경 matrix. TinyCNN CPU forward/backward, architecture recipe+state_dict strict load, 새 process 출력 완전 일치, 직접 실행 예제. CUDA/backend 미검증을 구분 |

실제 실행한 검증:

- CPU 포함 `pytest -q`: **22 passed**. [결과](evidence/pytest-cpu.txt).
- torch/AIMET 없는 독립 `/tmp/modeloctopus-m0-core` venv `pytest -q -rs`: **19 passed, 3 skipped**. torch fixture/doctor 검증은 실제 미설치로 skip. [결과](evidence/pytest-core-isolated.txt).
- 독립 venv `pip check`: No broken requirements found. torch/AIMET module 미설치 확인 및 core import 성공.
- `ruff check src tests examples/m0_roundtrip.py`, `ruff format --check ...`: 통과.
- `mypy`: 12 source files 통과.
- `python examples/m0_roundtrip.py`: TinyCNN CPU 저장·복원 비교 통과.
- `ef plan examples/classification_kd_ptq.yaml`, `ef plan examples/yolo_ptq.yaml`: 실제 계획 JSON 생성. training/download 실행은 없으며 `executable: false`.
- `ef doctor --profile core-ci/train-cpu/train-cuda/quant-aimet/deploy-tensorrt`: 각각 실행하고 [evidence](evidence/)에 JSON 기록. 없는 torch 환경에서도 doctor가 DEPENDENCY_NOT_INSTALLED를 보고하는 것을 확인.
- 전체 문서의 이전 브랜드 표기를 ModelOctopus로 통일하고 import/package/entry point group은 `modeloctopus`로 변경. `ef`는 명시된 CLI 계약을 유지.

Capability evidence:

| 조합 | 상태 | 근거 |
|---|---|---|
| core/schema/store/planner/CLI · Linux/Python 3.12 | supported (M0) | tests/test_core.py, isolated core 결과 |
| external plugin metadata discovery/lazy loader | supported (계약만) | test_registry, test_registry_entry_point_discovery_is_lazy. 실 backend plugin integration 증거는 아님 |
| custom TinyCNN · CPU forward/backward/state_dict restore | experimental fixture | tests/test_tinycnn.py, doctor-train-cpu.json, m0_roundtrip.py |
| train/create/export/quantize/prune/KD/evaluate/build/graph 서비스 | planned | M1 이후. 현재 facade/runner 및 실 service를 제공하지 않음 |
| CUDA/AIMET/TensorRT 실행 | unverified_environment | doctor report의 not_run과 환경 matrix. nvidia-smi GPU 조회를 CUDA 실행 증거로 취급하지 않음 |

남은 한계:

- store의 동시 writer/atomicity 검증은 Linux 로컬 filesystem 기준이다. 물리적 전원 차단, NFS·멀티호스트·Windows는 검증하지 않았다. crash 잔여 staging 정리·수동 복구 도구는 없다.
- TinyCNN은 M0 fixture다. dataset 준비·훈련/checkpoint resume·ONNX·ORT·실제 model provider는 M1부터 검증한다. synthetic 결과를 실제 정확도나 latency로 보고하지 않았다.
- planner의 실제 모델 구조·class 순서·operator·hardware 조건은 deferred다. 정적 계획 성공이 backend 실행 지원을 뜻하지 않는다.
- optional extras 중 core-ci/train-cpu 이외에는 검증된 profile lock이 없다. 해당 extras의 설치 후보를 호환성 보장으로 해석하지 않는다.
- 최초 폴더에 `.git`이 없어 Git diff/commit은 제공하지 않았다. 기존 설계 내용은 보존하며 이름/현재 상태/API 보충 문서만 갱신했다.

다음 작업은 M1의 TinyCNN·분류 model provider와 synthetic dataset 경로를 연결하고, epoch 경계 resume 및 FP ONNX/ORT round-trip을 실제 검증하는 것이다. 이번 작업에서는 시작하지 않았다.
