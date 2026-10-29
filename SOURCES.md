# 공식 자료와 설계 근거

확인일: 2026-10-28. 아래 자료는 기능 및 제약을 확인한 근거다. 이 문서의 architecture, API 이름, roadmap, artifact schema는 본 프로젝트를 위해 새로 제안한 설계다. 링크의 최신 문서 버전이 실제 구현 environment의 버전과 같다고 가정하지 않는다.

| ID | 자료 | 설계에 사용한 사실 |
|---|---|---|
| S1 | [PyNetsPresso 공식 저장소](https://github.com/Nota-NetsPresso/PyNetsPresso) | trainer/compressor/quantizer/converter/profiler 공개 인터페이스, 계정/API key 사용 흐름 |
| S2 | [torchvision models](https://docs.pytorch.org/vision/stable/models.html) | 모델 architecture와 pretrained weight provider |
| S3 | [PyTorch pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html) | mask/reparameterization 기반 pruning과 구조 축소를 구분할 근거 |
| S4 | [Torch-Pruning 공식 저장소](https://github.com/VainF/Torch-Pruning) | 구조적 의존성, dependency graph의 autograd 요구, pruning 후 구조 변경 |
| S5 | [AIMET 2.40 quantization workflow](https://qualcomm.github.io/aimet-pages/releases/2.40.0/tutorials/quantization_workflow.html) | ONNX PTQ와 표준 Q/DQ export 흐름 |
| S6 | [PyTorch Knowledge Distillation tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html) | teacher freeze, temperature KD, feature alignment의 기본 방식 |
| S7 | [PyTorch TorchScript deprecation 안내](https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference) | 새 설계에서 TorchScript를 기본 경로로 삼지 않는 이유 |
| S8 | [TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html) | 대상 환경과 engine 호환성 profile이 필요한 이유 |
| S9 | [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/) | runtime/EP별 연산 할당 및 지원 경계 |
| S10 | [Netron 공식 저장소](https://github.com/lutzroeder/netron) | ONNX 등 모델 그래프 조회와 Python 실행 지원 |
| S11 | [Ultralytics train](https://docs.ultralytics.com/modes/train/) | architecture/weight 로드와 기본 학습 경로 |
| S12 | [Torch-TensorRT ExportedProgram](https://docs.pytorch.org/TensorRT/user_guide/compilation/dynamo_export.html) | export graph와 backend compilation을 구분할 근거 |

PyNetsPresso의 공개 README와 사용자 API를 검토했으며 proprietary 서비스 내부 구현을 분석하거나 복제했다고 주장하지 않는다. 외부 프로젝트 코드·모델·데이터를 실제 포함할 때는 각 라이선스와 attribution을 기록한다. 본 설계는 특정 외부 서비스 계정이나 원격 업로드를 요구하지 않는다.

기존 대화의 YOLO Edge Lab 및 `yolo-edge-project-design.md`는 프로젝트 요구와 이전 범위를 유지하기 위한 자료로 사용했다. 새 library의 다중 모델 지원·plugin API·pruning/KD 구현이 이미 존재한다는 근거로 사용하지 않았다.
