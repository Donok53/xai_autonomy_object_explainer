# XAI Autonomy Driving Explainer

독립 ROS Noetic 프로젝트로, 주행 스택 자체를 수정하지 않고도
`왜 로봇이 이렇게 주행했는가`를 설명 가능한 형태로 재구성하는 것을 목표로 한다.

이 프로젝트는 `Modular_Approach_Autonomous_Driving-` 밖에서 동작한다.
핵심 아이디어는 다음과 같다.

- planner / LiDAR / explainability topic을 `faithful evidence`로 사용
- camera / VLM은 장면을 사람이 읽기 쉬운 언어로 풀어주는 `narration layer`로 사용
- 즉, `VLM-only`가 아니라 `planner-grounded hybrid explainability`를 지향
- exact object identity를 강하게 단정하지 않고, `matched visual region`과 `grounding confidence`를 함께 다룬다

## 프로젝트 범위

현재 첫 버전은 아래 두 가지를 제공한다.

- `build_bag_event_index.py`
  `/xai/event_log`를 기준으로 bag의 핵심 시점들을 JSONL 인덱스로 정리
- `xai_driving_explainer_node.py`
  `/xai/planner_snapshot`과 `/xai/event_log`를 받아 한국어 베이스라인 설명 번들을 publish
- `driving_vlm_prompt_builder.py`
  설명 번들을 받아 실제 VLM 입력용 `system_prompt`, `user_prompt`를 publish

현재는 로컬 VLM 호출까지 포함한다.
기본 backend는 `Ollama`이며, 기본 모델은 더 가벼운 `moondream`이다.

## 워크스페이스 구조

```text
xai_autonomy_driving_explainer/
├── src/
│   ├── CMakeLists.txt
│   └── xai_driving_explainer/
│       ├── CMakeLists.txt
│       ├── package.xml
│       ├── config/
│       ├── docs/
│       ├── launch/
│       └── scripts/
└── README.md
```

## Build

```bash
cd ~/code/xai_autonomy_driving_explainer
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
```

현재 환경처럼 `conda`나 `pyenv`가 기본 Python을 가리고 있으면,
ROS Noetic `catkin`이 `empy`를 못 찾을 수 있어서 위 방식이 더 안전하다.

## Runtime Baseline

기존 주행 스택이 `/xai/planner_snapshot`과 `/xai/event_log`를 publish 중일 때:

```bash
cd ~/code/xai_autonomy_driving_explainer
source devel/setup.bash
roslaunch xai_driving_explainer xai_driving_explainer.launch
```

출력 토픽:

- `/xai/driving_explanations`
- `/xai/driving_vlm_prompts`

기본 launch 설정에서는 이 설명이 토픽으로만 publish되는 것이 아니라,
터미널 로그에도 자동으로 출력된다.
즉 `roslaunch ...`를 띄워둔 터미널에서 아래 형태의 결합 설명을 바로 볼 수 있다.

```text
[XAI] event=path_blocked | planner_reason=... | camera_hint=... | matched_region=... | grounding_confidence=medium | final=...
```

현재 출력 철학:

- `planner_reason_ko`
  planner / LiDAR evidence를 기반으로 한 실제 판단 이유
- `scene_description_ko`
  카메라가 설명해야 할 장면 묘사
- `matched_visual_region_ko`
  planner evidence와 느슨하게 연결되는 카메라 내 대략적 영역
- `grounding_confidence`
  이 대응 힌트를 얼마나 강하게 믿을 수 있는지

추가 출력:

- `/xai/driving_vlm_explanations`
  실제 VLM scene description, detected objects, final combined explanation
- `/xai/driving_camera_overlay`
  카메라 프레임 위에 focus region과 설명을 얹은 annotated image

## Bag 으로 직접 돌려보기

이 bag에는 이미 `/xai/planner_snapshot`과 `/xai/event_log`가 들어 있으므로,
원래 주행 스택을 다시 띄울 필요 없이 이 프로젝트만 실행해도 된다.
기본 설정은 로컬 `Ollama` backend이므로 `OPENAI_API_KEY`가 필요하지 않다.

## Local VLM 준비

로컬 무료 VLM은 `Ollama + moondream`을 기본 추천으로 사용한다.

참고:

- Ollama API / local server: https://docs.ollama.com/api
- Ollama vision usage: https://docs.ollama.com/capabilities/vision
- Ollama structured outputs: https://docs.ollama.com/capabilities/structured-outputs
- Moondream on Ollama: https://ollama.com/library/moondream
- Qwen2.5-VL 3B on Ollama: https://ollama.com/library/qwen2.5vl:3b

### 1. Ollama 설치

Linux에서는 공식 설치 스크립트 또는 다운로드 페이지를 사용할 수 있다.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

또는:

- https://ollama.com/download

### 2. Ollama 서버 실행

```bash
ollama serve
```

별도 터미널에서 상태 확인:

```bash
curl http://127.0.0.1:11434/api/tags
```

### 3. 비전 모델 다운로드

```bash
ollama pull moondream
```

모델 목록 확인:

```bash
ollama list
```

터미널 1:

```bash
source /opt/ros/noetic/setup.bash
roscore
```

터미널 2:

```bash
cd ~/code/xai_autonomy_driving_explainer
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch xai_driving_explainer xai_driving_explainer.launch
```

터미널 3:

```bash
source /opt/ros/noetic/setup.bash
rosbag play --clock /home/byeongjae/bagfiles/record_real_20260422_180049.bag
```

설명 번들 확인:

```bash
source /opt/ros/noetic/setup.bash
rostopic echo -n 3 /xai/driving_explanations
```

설명 번들의 data 문자열만 보고 싶으면:

```bash
source /opt/ros/noetic/setup.bash
rostopic echo -n 1 /xai/driving_explanations/data
```

VLM 입력용 prompt 확인:

```bash
source /opt/ros/noetic/setup.bash
rostopic echo -n 1 /xai/driving_vlm_prompts
```

prompt의 `user_prompt` 텍스트만 보고 싶으면:

```bash
source /opt/ros/noetic/setup.bash
rostopic echo -n 1 /xai/driving_vlm_prompts/data
```

가볍게 구조만 보고 싶으면:

```bash
source /opt/ros/noetic/setup.bash
rostopic echo -n 1 /xai/driving_vlm_prompts/data | python3 -m json.tool
```

실제 VLM 결과 확인:

```bash
source /opt/ros/noetic/setup.bash
rostopic echo -n 1 /xai/driving_vlm_explanations/data | python3 -m json.tool
```

카메라 overlay 토픽 확인:

```bash
source /opt/ros/noetic/setup.bash
rostopic list | rg driving_camera_overlay
```

launch 터미널에서 기대하는 로그:

```text
[XAI] ...
[XAI-VLM] backend=ollama | event=... | status=ok | scene=... | objects=... | final=...
```

## Backend 전환

기본은 로컬 `Ollama` backend이다.

- `backend=ollama`
- `model=moondream`
- `endpoint=http://127.0.0.1:11434/api/chat`

필요하면 launch에서 바꿀 수 있다:

```bash
roslaunch xai_driving_explainer xai_driving_explainer.launch \
  vlm_backend:=ollama \
  vlm_model:=moondream \
  vlm_endpoint:=http://127.0.0.1:11434/api/chat
```

더 무거운 로컬 모델을 다시 시험하고 싶으면:

```bash
ollama pull qwen2.5vl:3b
roslaunch xai_driving_explainer xai_driving_explainer.launch \
  vlm_backend:=ollama \
  vlm_model:=qwen2.5vl:3b \
  vlm_request_timeout_s:=40.0 \
  vlm_min_request_interval_s:=4.0 \
  vlm_max_image_side_px:=448
```

OpenAI API로 다시 전환하고 싶으면:

```bash
export OPENAI_API_KEY=YOUR_KEY_HERE
roslaunch xai_driving_explainer xai_driving_explainer.launch \
  vlm_backend:=openai \
  vlm_model:=gpt-4.1-mini \
  vlm_endpoint:=https://api.openai.com/v1/chat/completions
```

## Offline Bag Index Build

기본 bag:

- `/home/byeongjae/bagfiles/record_real_20260422_180049.bag`

예시:

```bash
cd ~/code/xai_autonomy_driving_explainer
source devel/setup.bash
python src/xai_driving_explainer/scripts/build_bag_event_index.py \
  --bag /home/byeongjae/bagfiles/record_real_20260422_180049.bag \
  --output-dir generated/record_real_20260422_180049 \
  --max-match-dt-s 0.5
```

생성 파일:

- `generated/.../run_summary.json`
- `generated/.../event_index.jsonl`

## 데이터 철학

설명의 신뢰도를 높이기 위해 아래 우선순위를 따른다.

1. planner / control / emergency stop state
2. LiDAR obstacle evidence
3. global path / path change / blocked state
4. camera / VLM scene narration

즉, 카메라는 "보이는 장면"을 설명하고,
planner와 LiDAR는 "실제로 그렇게 판단한 이유"를 설명한다.

## 다음 단계

- image frame 추출기 추가
- point cloud evidence crop 저장
- VLM prompt builder 추가
- hybrid explanation scorer 추가
