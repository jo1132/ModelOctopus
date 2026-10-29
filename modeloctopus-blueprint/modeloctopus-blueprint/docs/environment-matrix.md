# M0 환경 검증

검증일: 2026-10-29 (Asia/Seoul). 실행 환경: Linux x86_64, Python 3.12.12. 최초 폴더는 Git 저장소가 아니며 기존 소스·tests·status는 없었다. 문서·예시는 보존하고 이름과 M0 현재 상태를 갱신했다.

| Profile | 실제 환경 | 검증과 상태 |
|---|---|---|
| core-ci | Pydantic 2.12.3 / pydantic-core 2.41.4 / PyYAML 6.0.3 | core/schema/store/registry/planner/CLI 검증. backend import를 차단한 subprocess 통과 |
| train-cpu | torch 2.6.0+cpu / NumPy 2.5.4 | TinyCNN CPU forward/backward 및 state_dict 새 process 복원 통과. fixture만 experimental; 학습 service planned |
| train-cuda | RTX A6000 / driver 535.309.01 / nvidia-smi CUDA 12.2 | 장치는 조회 가능. torch CPU build로 CUDA_UNAVAILABLE, 실행 not_run. 별도 CUDA toolkit 설치가 torch CUDA 지원을 뜻하지 않음 |
| quant-aimet | AIMET 미설치. onnx 1.23.2 / onnxruntime 1.31.1 metadata만 확인 | PTQ/QDQ/runtime integration not_run |
| deploy-tensorrt | TensorRT 미설치 | build/run/hardware acceptance not_run |
| Docker | docker CLI 없음 | container 검증 not_run |

`configs/core-ci.lock`은 core/dev transitive 설치 버전을 고정한다. `configs/train-cpu.lock`은 CPU fixture의 torch 및 관측된 transitive 버전을 추가한다. 이 두 profile 외 extras는 설치 진입점 후보다. 특히 AIMET/TensorRT/pruning/viz의 버전 미지정은 호환성 미확정 상태이며 검증된 lock이 아니다. 전체 extras 동시 설치는 보장하지 않는다. vision/yolo/onnx pin도 설치 metadata에 근거한 후보이며 M1 이후 integration 증거가 없다.

`docs/evidence/environment.freeze`는 기존 환경 전체의 관측 snapshot이며 새 최소 환경의 추천 lock으로 쓰지 않는다. doctor JSON은 package presence와 실제 실행 결과를 구분한다. source/config/example/schema hash는 `docs/evidence/source-hashes.json`에 저장한다. fixture artifact는 테스트 임시 디렉터리에서 검증하며 배포 산출물로 게시하지 않았다.
