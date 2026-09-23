# vLLM Verifier — Jev 호환 의사결정 엔진

Jev의 `POST /v1/systemone` 요청을 받아 `choice`, `score`, `noul` 응답을 반환합니다.
Mac에서는 Decision Kai의 후보 점수 계산 또는 DiffusionGemma의 MLX 추론을 선택할 수 있고,
외부 vLLM 서버에도 연결할 수 있습니다.
공식 TypeSafe Python SDK는 API 주소와 키를 바꾸어 사용할 수 있습니다.

## 네이티브 결정 엔진

[네이티브 엔진](native-engine.md)은 HTTP 추론 서버 없이 vLLM Python 런타임에 토큰 작업을
직접 제출합니다. 질문별 격리를 유지하면서 배치 실행, 공유 state prefix, 토큰 예산,
실패한 질문만 재시도하는 경로를 제공합니다. vLLM 경로는 오프라인 실험용이며 NVIDIA GPU 검증이
남아 있습니다. Mac용 MLX 경로는 실제 추론과 API 동작을 확인했습니다. Decision 경로는 직접 후보 점수 계산과 요청 간 배치 처리를 제공합니다. Diffusion sampler 최적화와 분산 서빙은 후속 항목입니다.

## Mac에서 Decision Kai 실행

```sh
UV_PROJECT_ENVIRONMENT=.venv-decision uv sync --locked --extra decision --python 3.12
UV_PROJECT_ENVIRONMENT=.venv-decision uv run --extra decision \
  vllm-verifier --runtime decision --port 18080
```

가중치 약 2.3GB를 내려받아 PyTorch MPS/Metal에서 실행합니다. JSON 생성 없이 후보 확률을
계산하며 여러 요청의 질문을 타입과 길이별로 묶습니다. 전체 입력 한도는 질문당 1,024토큰이고,
Choice·Score는 최소 두 후보가 필요합니다. 확률은 보정되지 않았고 confidence는 최대 후보
확률입니다. MLX와 Transformers 버전이 달라 별도 가상환경을 사용합니다.
[설정·스케줄링·벤치마크 안내](decision-models.md)를 참고하세요.

## Mac에서 DiffusionGemma 실행

Apple Silicon에서는 `uv sync --locked --extra mac`으로 설치한 뒤
`uv run --extra mac vllm-verifier --runtime mlx --port 18080`으로 실행할 수 있습니다.
4bit 모델 가중치 약 16.5GB를 내려받으며, Metal에서 텍스트 질문을 순차 처리합니다.
24GB Mac에 권장하는 낮은 동시성 설정과 메모리 제한은 [Mac 실행 안내](macos.md)를 참고하세요.

## 실행

vLLM 서버 주소와 키를 설정한 뒤 게이트웨이를 시작합니다.

```sh
export VERIFIER_API_KEY=your-local-api-key
export VERIFIER_BASE_URL=http://your-vllm-host:8000/v1
docker compose up --build -d gateway
```

주소는 컨테이너에서 접근할 수 있어야 합니다. 같은 Mac에서 실행하는 vLLM에는
`http://host.docker.internal:8000/v1`을 사용합니다.

GPU 없이 API를 확인하려면 별도 데모 서버를 실행합니다.

```sh
docker compose -f compose.demo.yaml up --build -d --wait
```

데모는 `examples/demo/`에 분리되어 있으며 제품 패키지나 이미지에 포함되지 않습니다.
균등 분포만 반환하고 실제 추론을 하지 않습니다. 기본 키는 `local-demo-key`이며,
`VERIFIER_API_KEY`가 설정되어 있으면 해당 값을 사용합니다.

## 제공 기능

- Choice: 주어진 옵션의 확률 분포와 가장 높은 확률의 선택지
- Score: 순서가 있는 기준의 확률 가중 평균, 기준 설명 및 확률 분포
- Noul: 질문이 참일 추정 확률
- `/v1/vision/systemone`: 이미지를 한 번 텍스트로 해석한 뒤 여러 질문에 공통으로 사용
- 질문별 독립 추론, 전체 동시 호출 제한, 타임아웃, 오류 응답, 제한된 재생성
- Bearer 인증, 상태 점검, Prometheus 지표, Docker Compose 및 CI

이미지 요청은 원본 Jev API의 확장입니다. 이미지 URL을 서버가 내려받지 않으며,
base64 PNG/JPEG/WebP만 받습니다. 선택 결과에 따른 실제 명령 실행은 호출 애플리케이션이 담당합니다.

## 호환성과 검증 범위

Jev와 **요청·응답 프로토콜을 맞춘 독립 구현**입니다. Jev 모델 자체나 학습 방법을 재현하지 않습니다.
모델이 생성한 확률을 구조적으로 검증하며, 실제 정답 확률로 보정되어 있다고 보장하지 않습니다.
`confidence`는 정규화 엔트로피로 계산하고, Jev의 수치와 동일하다고 주장하지 않습니다.

현재 로컬 테스트는 실제 TypeSafe SDK와 가짜 upstream 응답으로 서버 동작을 검증합니다.
Docker ARM64 이미지 빌드·실행과 컨테이너 간 HTTP 연동도 검증했습니다.
GPU 추론 정확도·지연 시간은 해당 환경에서 [평가 도구](evaluation.md)로 확인해야 합니다.
잘못된 JSON이나 확률을 정상 결과로 위장하지 않고, 재시도 후에도 실패하면 오류를 반환합니다.

[전체 API 호환 범위](compatibility.md) · [설계](architecture.md) · [배포](deployment.md)
