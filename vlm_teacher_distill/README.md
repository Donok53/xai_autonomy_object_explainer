# VLM Teacher Distill

`xai_autonomy_driving_explainer`의 런타임 경로는 그대로 두고,
오프라인에서만 VLM을 teacher로 써서 dataset을 만들기 위한 별도 프로젝트다.

핵심 아이디어:

- 실행 시점: `YOLO + LiDAR + planner + template`
- 오프라인 학습 시점: `VLM teacher`
- 목표: VLM이 추론에 직접 들어오지 않아도, 더 안정적인 semantic label과 설명 템플릿을 만들기

## 현재 포함된 기능

1. `scripts/export_teacher_dataset.py`
   - bag에서 `/xai/event_log`, `/xai/planner_snapshot`, 카메라 이미지, point cloud를 묶어
     teacher dataset을 만든다.
   - event 중심 샘플을 뽑는다.
   - 이미지와 downsampled point cloud를 함께 저장한다.
   - VLM용 prompt 초안도 metadata에 같이 넣는다.

2. `scripts/annotate_teacher_with_ollama.py`
   - export한 dataset을 읽고, 이미지 + planner/LiDAR 문맥을 VLM에 보내
     teacher annotation을 JSONL로 저장한다.
   - 런타임이 아니라 오프라인 데이터 생성용이다.

## 구조

```text
vlm_teacher_distill/
├── README.md
├── requirements.txt
├── data/
│   └── .gitkeep
└── scripts/
    ├── annotate_teacher_with_ollama.py
    └── export_teacher_dataset.py
```

## 권장 실행 환경

ROS bag를 직접 읽으므로 `/usr/bin/python3` 기준이 가장 안전하다.

```bash
cd ~/code/xai_autonomy_driving_explainer
source /opt/ros/noetic/setup.bash
source devel/setup.bash
```

## 1. Teacher Dataset Export

기본 예시:

```bash
/usr/bin/python3 vlm_teacher_distill/scripts/export_teacher_dataset.py \
  --bag /home/byeongjae/bagfiles/record_real_20260422_180049.bag \
  --output-dir /home/byeongjae/code/xai_autonomy_driving_explainer/vlm_teacher_distill/data/record_real_teacher
```

생성 결과:

- `images/*.jpg`
- `pointcloud/*.npz`
- `metadata/teacher_dataset.jsonl`

`teacher_dataset.jsonl` 한 줄에는 대략 이런 정보가 들어간다.

- `sample_id`
- `stamp`
- `event_label`
- `planner_reason`
- `motion_state`
- `path_blocked`
- `image_path`
- `pointcloud_path`
- `teacher_prompt_ko`
- 원본 `event_log`, `planner_snapshot`

## 2. Ollama Teacher Annotation

예시:

```bash
/usr/bin/python3 vlm_teacher_distill/scripts/annotate_teacher_with_ollama.py \
  --dataset-dir /home/byeongjae/code/xai_autonomy_driving_explainer/vlm_teacher_distill/data/record_real_teacher \
  --model moondream
```

출력:

- `annotations/teacher_labels.jsonl`

각 라인은:

- `sample_id`
- `model`
- `teacher_output_raw`
- `teacher_output_json`

형태로 저장된다.

## 의도한 다음 단계

이 프로젝트의 목표는 바로 student 학습까지 끝내는 것이 아니라,
우선 아래 파이프라인을 안정화하는 것이다.

1. bag -> teacher dataset export
2. dataset -> VLM teacher label
3. teacher label 품질 확인
4. 작은 student classifier / template selector 학습

## 참고

- 현재 class 해석은 `xai_autonomy_driving_explainer`에서 쓰는 whitelist 철학을 그대로 따른다.
- 런타임 project는 수정하지 않고, 여기서는 오프라인 teacher dataset만 다룬다.
