# Project Strategy

## Why Separate Project

이 프로젝트는 주행 스택을 바꾸지 않고 설명 계층만 독립적으로 실험하기 위해
별도 워크스페이스로 분리한다.

## Core Position

`VLM-only` 설명은 사람이 보기에는 그럴듯할 수 있지만,
실제 주행 의사결정의 원인과 어긋날 가능성이 높다.

그래서 설명은 두 층으로 나눈다.

1. `faithful layer`
   planner state, emergency stop, path blocked, LiDAR evidence, path change
2. `narration layer`
   camera frame / VLM을 이용해 사람이 이해하기 쉬운 자연어 해설 생성

## Phase 1

- bag의 `/xai/event_log`를 기준으로 설명 샘플 인덱스 생성
- `/xai/planner_snapshot`에서 구조화된 판단 근거 복원
- rule-based baseline explanation 생성

## Phase 2

- image frame export
- LiDAR evidence crop export
- VLM prompt template 설계
- planner-grounded hybrid explanation 생성

## Phase 3

- 설명 품질 평가
- event별 explanation consistency scoring
- failure case taxonomy 정리
