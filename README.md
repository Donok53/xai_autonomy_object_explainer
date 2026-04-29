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

현재 기본 런타임은 아래 구성이다.

- `build_bag_event_index.py`
  `/xai/event_log`를 기준으로 bag의 핵심 시점들을 JSONL 인덱스로 정리
- `xai_driving_explainer_node.py`
  `/xai/planner_snapshot`과 `/xai/event_log`를 받아 planner-grounded 설명 번들을 publish
- `driving_scene_detector_node.py`
  카메라 프레임과 설명 번들을 받아 lightweight detector 기반의 실시간 시각 설명을 생성
- `driving_camera_overlay_viewer.py`
  카메라 프레임 위에 `event`, `final`, `objects`만 간단히 overlay

VLM 실험 경로는 남아 있지만, 기본값은 꺼져 있다.
현재 기본 backend는 `yolo_worker`이며, detector는 내부 판단용으로 사용한다.
또한 기본적으로 tracker를 켜서, 같은 물체가 몇 프레임 동안 라벨이 흔들리더라도
동일 객체로 최대한 유지하도록 설계했다.
추가로 `/cmd_vel`과 odometry를 함께 읽어, 로봇이 움직이는 동안에는
track hold 시간과 label smoothing을 더 보수적으로 적용한다.
최근에는 여기에 `/planning/linefit_ground/non_ground_cloud`와 `camera_info`를 함께 읽어
planner가 보고 있는 `anchor_xyz` 주변의 point cloud를 카메라 화면으로 투영하고,
YOLO는 `무엇으로 보이는지`, point cloud는 `실제로 어디에 있는지`를 함께 다루도록 확장했다.

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
- `/xai/driving_vlm_explanations`
- `/xai/driving_camera_overlay`

기본 launch 설정에서는 detector 결과가 토픽으로 publish될 뿐 아니라,
터미널 로그에도 자동으로 출력된다.
즉 `roslaunch ...`를 띄워둔 터미널에서 아래 형태의 실시간 설명을 바로 볼 수 있다.

```text
[XAI-DETECTOR] backend=yolo_worker | event=path_blocked | status=ok | scene=... | objects=... | final=...
```

현재 출력 철학:

- `planner_reason_ko`
  planner / LiDAR evidence를 기반으로 한 실제 판단 이유
- `scene_description_ko`
  detector가 focus region 안에서 본 장면 요약
- `detected_objects_ko`
  현재 프레임에서 잡힌 주요 객체 목록
- `final_combined_explanation_ko`
  planner evidence와 detector 결과를 합친 실시간 최종 설명

추가 출력:

- `/xai/driving_vlm_explanations`
  이름은 legacy이지만, 기본 설정에서는 detector 결과 payload를 담는다
- `/xai/driving_camera_overlay`
  기본값은 검은 배경의 `lidar_only` 화면 위에 전방 non-ground point cloud 전체를 BEV(top-down)로 렌더링하고, 선택된 장애물 cluster와 memory/track id를 다른 색으로 강조한다
  detector bbox는 내부 판단에만 사용한다

## Bag 으로 직접 돌려보기

이 bag에는 이미 `/xai/planner_snapshot`과 `/xai/event_log`가 들어 있으므로,
원래 주행 스택을 다시 띄울 필요 없이 이 프로젝트만 실행해도 된다.
기본 설정은 detector-only runtime이므로 `OPENAI_API_KEY`나 `Ollama`가 필요하지 않다.

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
roslaunch xai_driving_explainer xai_driving_explainer.launch log_combined_explanation:=false
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

detector 결과 확인:

```bash
source /opt/ros/noetic/setup.bash
rostopic echo -n 1 /xai/driving_vlm_explanations/data | python3 -m json.tool
```

overlay 토픽 확인:

```bash
source /opt/ros/noetic/setup.bash
rostopic list | rg driving_camera_overlay
```

launch 터미널에서 기대하는 로그:

```text
[XAI-DETECTOR] backend=yolo_worker | event=... | status=ok | scene=... | objects=... | final=...
```

## Camera-LiDAR Extrinsic 튜너

설계도면이나 정적 TF가 없을 때는, bag를 보면서 카메라 위에 point cloud를 직접 투영해
`tx/ty/tz/roll/pitch/yaw`를 수동으로 맞추는 편이 가장 현실적이다.

체커보드/ChArUco bag가 준비되어 있다면, 먼저 자동 추정으로 초기 extrinsic을 구해본 뒤
필요할 때만 수동 튜너로 미세조정하는 흐름을 권장한다.

자동 추정 실행:

```bash
cd ~/code/xai_autonomy_driving_explainer
source /opt/ros/noetic/setup.bash
source devel/setup.bash
/usr/bin/python3 src/xai_driving_explainer/scripts/auto_calibrate_camera_lidar_charuco.py \
  --bag /home/byeongjae/bagfiles/yongbong_checkerboard_v2.bag \
  --point-cloud-topic /ouster/points
```

기본값은 현재 작업에 맞춰 아래처럼 잡혀 있다.

- ChArUco `6 x 8`
- Checker size `25 mm`
- Marker size `18.75 mm`
- Dictionary `DICT_4X4_50`

자동 추정은 camera에서 ChArUco pose를 찾고, 현재 초기 extrinsic을 바탕으로
보드 근처 점군을 자동 선택한 뒤 plane 기반으로 refinement한다.
완벽하지 않을 수 있으므로, 결과 YAML을 얻은 뒤 필요하면 아래 수동 튜너로 마무리하면 된다.

튜너 실행:

```bash
cd ~/code/xai_autonomy_driving_explainer
source /opt/ros/noetic/setup.bash
source devel/setup.bash
/usr/bin/python3 src/xai_driving_explainer/scripts/tune_camera_lidar_extrinsic.py \
  --bag /home/byeongjae/bagfiles/record_real_20260422_180049.bag
```

`camera_info`가 없는 bag라면 다른 bag에서 intrinsic만 읽어와서 같이 쓸 수 있다.

```bash
/usr/bin/python3 src/xai_driving_explainer/scripts/tune_camera_lidar_extrinsic.py \
  --bag /home/byeongjae/bagfiles/yongbong_checkerboard.bag \
  --point-cloud-topic /ouster/points \
  --camera-info-bag /home/byeongjae/bagfiles/record_real_20260422_180049.bag
```

또는 camera intrinsic을 직접 알고 있다면 수동 입력도 가능하다.

```bash
/usr/bin/python3 src/xai_driving_explainer/scripts/tune_camera_lidar_extrinsic.py \
  --bag /home/byeongjae/bagfiles/yongbong_checkerboard.bag \
  --point-cloud-topic /ouster/points \
  --camera-frame camera_color_optical_frame \
  --camera-width 640 \
  --camera-height 480 \
  --fx 384.9215 \
  --fy 384.4141 \
  --cx 328.9431 \
  --cy 245.7993
```

기본 입력:

- image: `/camera/color/image_raw`
- camera info: `/camera/color/camera_info`
- point cloud: `/planning/linefit_ground/non_ground_cloud`

주요 조작:

- `1/2`: `tx` 감소/증가
- `3/4`: `ty` 감소/증가
- `5/6`: `tz` 감소/증가
- `u/o`: roll 감소/증가
- `i/k`: pitch 감소/증가
- `j/l`: yaw 감소/증가
- `n/p`: 다음/이전 샘플
- `[` 또는 `-`: step 축소
- `]` 또는 `=`: step 확대
- `s`: YAML 저장 후 종료
- `q`: 저장 없이 종료

저장 결과에는 아래 값이 같이 들어간다.

- `translation_xyz`
- `rotation_xyzw`
- `rotation_rpy_deg`
- 현재 launch에 바로 붙일 수 있는 override 문자열

튜닝 팁:

- 복도 벽 모서리, 문틀, 바닥-벽 경계처럼 직선이 분명한 구간에서 맞추는 게 좋다.
- 사람처럼 움직이는 객체보다 우산, 벽, 문, 고정 가구 같은 정적 물체로 맞추는 게 안정적이다.
- 먼저 yaw와 pitch를 맞추고, 그 다음 `tx/ty/tz`를 세부 조정하는 편이 빠르다.
- 점군이 전체적으로 좌우로 밀리면 yaw/ty를 먼저 의심하고, 위아래로 뜨면 pitch/tz를 먼저 의심하는 게 좋다.

## Detector / VLM 전환

기본은 detector-only runtime이다.

- `enable_scene_detector=true`
- `enable_vlm_prompt_builder=false`
- `enable_vlm_inference=false`
- `scene_detector_backend=yolo_worker`

focus crop과 처리 주기만 바꾸고 싶으면:

```bash
roslaunch xai_driving_explainer xai_driving_explainer.launch \
  scene_detector_min_process_interval_s:=0.20 \
  scene_detector_focus_crop_margin_ratio:=0.10 \
  scene_detector_max_image_side_px:=416
```

YOLO detector를 다른 모델로 바꾸고 싶으면:

```bash
roslaunch xai_driving_explainer xai_driving_explainer.launch \
  scene_detector_backend:=yolo_worker \
  scene_detector_yolo_model:=yolo11n.pt \
  scene_detector_yolo_imgsz:=320
```

tracker 관련 파라미터를 조정하고 싶으면:

```bash
roslaunch xai_driving_explainer xai_driving_explainer.launch \
  scene_detector_yolo_use_tracker:=true \
  scene_detector_yolo_tracker_config:=botsort.yaml \
  scene_detector_track_hold_ttl_s:=0.9 \
  scene_detector_track_min_hits:=2
```

로봇 motion을 반영한 tracking 안정화를 더 조정하고 싶으면:

```bash
roslaunch xai_driving_explainer xai_driving_explainer.launch \
  scene_detector_cmd_vel_topic:=/cmd_vel \
  scene_detector_odom_topic:=/lio_localizer/odometry/optimization \
  scene_detector_use_motion_awareness:=true \
  scene_detector_motion_extra_hold_ttl_s:=0.9 \
  scene_detector_motion_extra_center_match_px:=140.0 \
  scene_detector_motion_selected_track_bonus:=0.45
```

## Optional: Local VLM 실험 경로

기본 detector가 아니라 예전 로컬 VLM 실험 경로를 다시 켜고 싶으면:

```bash
ollama serve
ollama pull moondream
roslaunch xai_driving_explainer xai_driving_explainer.launch \
  enable_scene_detector:=false \
  enable_vlm_prompt_builder:=true \
  enable_vlm_inference:=true \
  vlm_backend:=ollama \
  vlm_model:=moondream \
  vlm_endpoint:=http://127.0.0.1:11434/api/chat
```

OpenAI API로 다시 전환하고 싶으면:

```bash
export OPENAI_API_KEY=YOUR_KEY_HERE
roslaunch xai_driving_explainer xai_driving_explainer.launch \
  enable_scene_detector:=false \
  enable_vlm_prompt_builder:=true \
  enable_vlm_inference:=true \
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

즉, 카메라는 "지금 보이는 대상"을 빠르게 붙이고,
planner와 LiDAR는 "실제로 그렇게 판단한 이유"를 설명한다.

## 다음 단계

- detector 종류 확장
- 사람 외 객체용 lightweight detector 추가
- planner evidence와 detector bbox 매칭 개선
- overlay / report viewer 다듬기
