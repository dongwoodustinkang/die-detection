# Side 표면 OK/NG 분류 CNN 가이드

Side 검사 이미지(A/B 2페이지 TIFF)에서 **표면 타일을 잘라 → 학습 데이터셋을 만들고 → 작은 CNN을 학습하고 → 평가**하는 전체 과정을 정리한 문서입니다.

```
원본 TIFF (A/B 2페이지)
   │  ① tools/make_data.py        표면 크롭 + train/val/test 비율 분할
   ▼
dataset/infer/side_surface_v1/   A/, B/ PNG 타일 + samples.csv
   │  ② tools/train_classifier.py  B 타일로 CNN 학습
   ▼
models/side_surface_cnn_MMDDHH.pt (+ .png 학습 그래프)
   │  ③ tools/eval_classifier.py   OK/NG 폴더로 성능 확인
   ▼
정확도 · TP/FN/FP/TN · 방향별 오류 (터미널) + 오분류 CSV
```

---

## 0. 실행 환경

모든 명령은 **프로젝트 루트(`die-detection/`)** 에서 실행합니다.

```bash
eval "$(/opt/miniconda3/bin/conda shell.zsh hook)" && conda activate keti
```

| 패키지 | 용도 |
|---|---|
| `torch` | 모델 학습·추론 (macOS는 MPS, Linux/Windows는 CUDA, 없으면 CPU 자동 선택) |
| `opencv-python-headless`, `numpy`, `pillow` | 이미지 읽기·전처리 |
| `matplotlib` | 학습 그래프 저장 |

새 PC에서는 `pip install torch matplotlib`을 추가로 설치합니다.

> **경로는 영문으로 두세요.** `cv2.imread`는 Windows에서 한글 경로를 읽지 못합니다. 그래서 학습 데이터 폴더 이름을 `학습추론` → `infer`로 바꿨습니다.

---

## 데이터 파일명 규칙

라벨과 촬영 방향을 **파일명과 폴더 이름에서 읽기** 때문에, 형식이 맞지 않으면 샘플이 빠지거나 집계가 틀어집니다.

### 기본 형식

```
{라벨}_{방향}_{나머지}.{확장자}

NG_Back_20261006_130531_990_0001.tiff
│  │    └───────────────────────── 나머지: 자유 (날짜_시각_번호 권장, 샘플마다 달라야 함)
│  └── 방향: Back | Front | Left | Right
└───── 라벨: OK | NG
```

| 자리 | 규칙 |
|---|---|
| 라벨 | `OK` 또는 `NG` |
| 방향 | `Back`, `Front`, `Left`, `Right` 중 하나. 알파벳만, 첫 글자 대문자 권장 |
| 구분자 | 밑줄 `_`. 라벨과 방향 뒤에 **반드시** 밑줄이 있어야 함 |
| 나머지 | 같은 이름이 겹치지 않게. 확장자를 뺀 파일명 전체가 `sample_id`가 됨 |

### ① 원본 TIFF (`make_data.py` 입력)

- 확장자: `.tif` 또는 `.tiff` (대소문자 무관). `.`으로 시작하는 숨김 파일은 무시합니다.
- A/B **2페이지** TIFF여야 합니다. 1페이지면 크롭 실패(`failed.csv`)로 기록됩니다.
- **라벨**을 정하는 순서:
  1. 상위 폴더 중 이름이 `OK`/`NG`인 폴더가 있으면 그것을 씁니다 (우선).
  2. 없으면 파일명 앞부분 `OK_` / `NG_`를 씁니다 (대소문자 무관).
  3. 둘 다 없으면 "OK/NG 라벨을 알 수 없습니다"로 실패 처리됩니다.
- **방향**은 파일명 두 번째 토큰을 첫 글자만 대문자로 바꿔 씁니다 (`ng_back_...` → `Back`).
- 다른 폴더에 같은 파일명이 있으면 두 번째부터 `__2`, `__3`이 붙습니다.

| 파일명 | 라벨 | 방향 | 결과 |
|---|---|---|---|
| `NG_Back_20261006_130531_990_0001.tiff` | NG | Back | ✅ |
| `ok_left_20260902_155240_607_0020.tif` | OK | Left | ✅ 대소문자 무관 |
| `OK/left/OK_Left_20260902_..._0016.tiff` | OK (폴더) | Left | ✅ |
| `OK_side_left_20260714_094431_131.tiff` | OK | **Side** | ⚠️ 방향이 `Side`로 잡힘 → 방향별 집계·분할에서 별도 그룹 |
| `20261006_0001.tiff` (OK/NG 폴더 밖) | – | – | ❌ 라벨을 알 수 없음 → 실패 |

> `dataset/테스트/Side`의 `OK_side_left_...` 형식 파일을 쓰려면 `OK_Left_...`로 이름을 바꾸거나, 방향 추출 규칙을 고쳐야 합니다.

### ② 학습·평가용 PNG (`train_classifier.py`, `eval_classifier.py` 입력)

`make_data.py`가 만든 파일은 이미 규칙에 맞습니다. 직접 만든 폴더를 평가할 때만 신경 쓰면 됩니다.

```
평가 폴더/
├── OK/OK_Left_20260902_155240_607_0020.png
└── NG/NG_Back_20261006_130531_990_0001.png
```

- **라벨은 폴더 이름(`OK/`, `NG/`)으로만** 정합니다. 파일명 앞부분은 보지 않습니다.
  폴더 이름은 정확히 대문자 `OK`, `NG` (Linux는 대소문자를 구분합니다).
- 확장자는 소문자 `.png`만 읽습니다 (Linux에서 `.PNG`는 빠짐).
- 방향은 파일명 두 번째 토큰을 **그대로** 씁니다. 평가 결과의 방향별 집계는 정확히 `Back`/`Front`/`Left`/`Right`일 때만 셉니다.
- 파일명에 밑줄 `_`이 **최소 1개** 있어야 합니다. 없으면 방향을 꺼내다가 `IndexError`가 납니다.
- 그레이스케일 8bit(uint8) 이미지여야 합니다. 크기는 자동으로 32×176으로 맞춰집니다.

| 파일 | 결과 |
|---|---|
| `NG/NG_Back_20261006_130531_990_0001.png` | ✅ 라벨 NG, 방향 Back |
| `OK/OK_left_0001.png` | ⚠️ 학습·평가는 되지만 방향 `left`는 집계에서 빠짐 |
| `OK/sample0001.png` | ❌ 밑줄이 없어 `IndexError` |
| `ok/OK_Back_0001.png` | ❌ Linux에서 `ok/` 폴더를 못 찾음 |

---

## 1. 데이터셋 만들기 — `tools/make_data.py`

### 하는 일

1. 입력 폴더 아래의 `*.tif`, `*.tiff`를 하위 폴더까지 모두 찾습니다.
2. 파일마다 **검사 프로그램과 같은 함수**(`load_ab_tiff_pages` → `extract_surface_tiles`)로 표면을 잘라 A/B 타일을 만듭니다.
   학습 데이터와 실제 검사가 같은 크롭을 쓰도록 하기 위해서입니다.
3. 라벨(OK/NG)과 방향(Back/Front/Left/Right)을 읽습니다.
   - 라벨: 상위 폴더 이름이 `OK`/`NG`이면 그것을, 아니면 파일명 앞부분(`OK_...`, `NG_...`)을 씁니다.
   - 방향: 파일명 두 번째 토큰 (`NG_Back_...` → `Back`)
4. 라벨별로 **비율**에 맞춰 train/val/test 개수를 정하고, 방향 비율을 유지하면서 무작위 분할합니다 (`--seed`가 같으면 분할도 같음).
   - 크롭에 성공한 실제 샘플 수에 비율을 곱하므로, 샘플 수가 딱 떨어지지 않아도 됩니다 (예: 3997장 × 0.7/0.15/0.15 → 2798/600/599).
   - 소수점 때문에 남는 1~2장은 소수점이 큰 split부터 채워서, 세 split을 더하면 항상 전체 개수가 됩니다.
   - 비율 합이 1보다 작으면 남는 샘플은 어느 split에도 넣지 않습니다 (`dataset_info.json`의 `unused`).
5. 그레이스케일 PNG(무손실)로 저장합니다.

### 현재 데이터셋을 다시 만드는 명령

지금 쓰는 `dataset/infer/side_surface_v1`은 OK/NG 각각 3000/500/500장, seed 7로 만들어졌습니다. 비율로 바꾸면 0.75/0.125/0.125이고, 아래 명령으로 다시 만들면 8000장 모두 같은 split에 들어갑니다.

```bash
python tools/make_data.py dataset/원본/Side_Demo_1006 --output dataset/infer/side_surface_v1 --train 0.75 --val 0.125 --test 0.125 --seed 7 --overwrite
```

> 기본값(`--output dataset/side_surface`, 0.7/0.15/0.15, seed 42)과 다르니, 같은 데이터셋이 필요하면 위 옵션을 그대로 쓰세요.
> v1을 만들 때는 개수로 지정하던 이전 버전이라, v1의 `dataset_info.json`에는 `ratios` 대신 `per_label`(개수)이 적혀 있습니다.

### 테스트(평가) 전용 데이터 만들기

다른 날짜 데이터처럼 **평가에만 쓸 데이터**는 train/val을 0으로 두면 전부 test로 들어갑니다.

```bash
python tools/make_data.py dataset/Test_1013 --output dataset/infer/side_surface_test_1013 --train 0 --val 0 --test 1.0
```

결과의 `B/test` 폴더를 그대로 `eval_classifier.py --data`에 넣으면 됩니다 (`train/`, `val/` 폴더는 비어 있음).

### 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `INPUT_DIR ...` | `dataset/원본/Side_Demo_1006` | TIFF가 있는 폴더. 여러 개 지정 가능 |
| `--output` | `dataset/side_surface` | 출력 폴더 |
| `--train` / `--val` / `--test` | 0.7 / 0.15 / 0.15 | **OK, NG 각각**에서 가져갈 비율. 0 이상, 합은 1 이하 |
| `--seed` | 42 | 분할 셔플 seed |
| `--overwrite` | 꺼짐 | 출력 폴더가 있으면 지우고 다시 생성 |

비율이 잘못되면(예: `--train 0.7 --val 0.3 --test 0.3`, 합 1.3) 크롭을 시작하기 전에 바로 멈춥니다.

### 출력 구조

```
dataset/infer/side_surface_v1/
├── A/{train,val,test}/{OK,NG}/{sample_id}.png   원본 표면 질감 (TIFF 1페이지)
├── B/{train,val,test}/{OK,NG}/{sample_id}.png   배경 평탄화, 결함만 어둡게 (TIFF 2페이지) ← 학습에 사용
├── samples.csv        샘플별 split·label·view·크기·크롭 좌표
├── failed.csv         크롭 실패 목록과 사유
└── dataset_info.json  비율(ratios)·라벨별 실제 개수(counts_by_label)·seed 요약
```

현재 데이터: 8000장 = 4방향 × OK/NG × 1000장, train 6000 / val 1000 / test 1000, 크롭 실패 0장.
타일 크기는 높이 30~40px(대부분 32~34), 너비 174~176px입니다.

---

## 2. 학습 — `tools/train_classifier.py`

### 실행

```bash
python tools/train_classifier.py
```

```bash
python tools/train_classifier.py --device cpu
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--device` | `auto` | `auto`(CUDA → MPS → CPU) / `cpu` / `cuda` / `mps` |

데이터 경로·하이퍼파라미터는 파일 상단 상수로 바꿉니다.

| 상수 | 값 | 의미 |
|---|---|---|
| `DEFAULT_DATA` | `dataset/infer/side_surface_v1` | 학습 데이터셋 폴더 |
| `BATCH_SIZE` | 64 | 한 번에 넣는 타일 수 |
| `EPOCHS` | 30 | 최대 학습 횟수 |
| `PATIENCE` | 5 | val loss가 이 횟수만큼 연속으로 나아지지 않으면 조기 종료 |
| `LR`, `WEIGHT_DECAY` | 1e-3, 1e-4 | AdamW 학습률·가중치 감쇠 |

### 처리 흐름

```
B/{train,val,test} PNG 읽기 (load_split → load_folder)
  → preprocess: 32×176로 resize, 반전(배경 0 / 결함 양수), 0~1 정규화
  → DataLoader (train만 셔플)
  → epoch 반복: 학습(train) → 평가(val)
       · val loss가 가장 낮은 시점의 가중치를 보관
       · PATIENCE 동안 개선 없으면 조기 종료
  → 최적 가중치 저장 (.pt) + 학습 그래프 저장 (.png)
```

### 출력 파일

```
models/
├── side_surface_cnn_101310.pt    가중치 (CPU 텐서로 저장 → 어느 OS에서나 로드 가능)
└── side_surface_cnn_101310.png   Training/Validation Acc·Loss 그래프 + best epoch 표시
```

- 파일명은 `side_surface_cnn_{월일시}` 입니다 (예: `101310` = 10월 13일 10시).
- 같은 시간대에 다시 학습하면 `_01`, `_02`가 붙어 덮어쓰지 않습니다.

### 실행 예시 출력

```
[train] 6000장  x=(6000, 1, 32, 176)  NG 비율 0.50
[val] 1000장  x=(1000, 1, 32, 176)  NG 비율 0.50
[test] 1000장  x=(1000, 1, 32, 176)  NG 비율 0.50
[device] mps
[ 1] 학습세트 loss 0.2733 정확도 0.9688 | 검증세트 loss 0.1524 정확도 0.9900
...
[BEST] 최고 검증 epoch 11 : loss 0.0066
저장: .../models/side_surface_cnn_101310.pt (+ .png)
```

MPS 기준 epoch당 약 1.4초, CPU는 약 8~9초입니다.

---

## 3. 평가 — `tools/eval_classifier.py`

학습을 다시 하지 않고, 저장된 모델로 `OK/`·`NG/` 폴더를 평가합니다.

### 실행

```bash
python tools/eval_classifier.py --model models/side_surface_cnn_101310.pt
```

```bash
python tools/eval_classifier.py --model models/side_surface_cnn_101310.pt --data dataset/infer/side_surface_v1/B/val --threshold 0.08
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model` | **필수** | 평가할 `.pt` 파일 |
| `--data` | `dataset/infer/side_surface_v1/B/test` | `OK/`, `NG/` 하위 폴더가 있는 평가 폴더 |
| `--threshold` | 0.5 | NG 확률이 이 값 이상이면 NG로 판정 |
| `--device` | `auto` | 계산 장치 |
| `--output` | 모델 옆 `{모델명}_wrong_{데이터 폴더명}.csv` | 오분류 CSV 저장 경로 |

### 오분류 CSV

터미널에는 요약만 출력하고, 틀린 샘플은 CSV로 저장합니다. 오분류가 0장이어도 헤더만 있는 파일이 만들어집니다.

```
models/side_surface_cnn_101310_wrong_test.csv    ← --data .../B/test 평가 결과
models/side_surface_cnn_101310_wrong_val.csv     ← --data .../B/val 평가 결과
```

| 컬럼 | 내용 |
|---|---|
| `error` | `FN`(NG를 놓침) 또는 `FP`(OK를 NG로 판정) |
| `sample_id` | 확장자를 뺀 파일명 |
| `label` / `pred` | 실제 라벨 / 모델 판정 |
| `cnn_prob` | NG 확률(점수), 소수점 4자리 |
| `threshold` | 평가에 쓴 판정 기준 |
| `view` | 촬영 방향 |
| `path` | 이미지 경로 (프로젝트 안이면 상대 경로) |

- 정렬: **FN 먼저**, 같은 종류 안에서는 threshold에서 **멀리 벗어난(확실히 틀린) 샘플부터**. 위에서부터 이미지를 열어 보면 됩니다.
- `utf-8-sig`로 저장해서 Windows 엑셀에서도 한글 경로가 깨지지 않습니다.
- 같은 모델·같은 폴더 이름으로 다시 평가하면 덮어씁니다. threshold별로 남기려면 `--output`을 지정하세요.

### 결과 읽는 법

| 항목 | 뜻 |
|---|---|
| **TP** | NG를 NG로 맞힘 |
| **FN** | NG를 OK로 놓침 → 불량 유출, **가장 위험** |
| **FP** | OK를 NG로 잘못 판정 → 재검사 비용 |
| **TN** | OK를 OK로 맞힘 |
| 방향별 오류 | Back/Front/Left/Right 중 특정 방향만 약한지 확인 |
| 오분류 CSV | 틀린 샘플의 경로와 NG 확률. 이미지를 직접 열어 원인 확인 |

### 주의

- 모델은 **B 타일로만** 학습했습니다. `--data`에 A 폴더를 넣으면 결과가 의미 없습니다.
- 방향은 파일명 두 번째 토큰으로 구분합니다. 파일명이 `라벨_방향_...` 형식이 아니면 방향별 집계에서 빠집니다.

---

## 4. 모델 구조 — `side/classifier.py`

| 구성 | 내용 |
|---|---|
| `preprocess(image)` | 그레이스케일 → `(32, 176)` resize(INTER_AREA) → `1 - x/255` 반전 → `(1, 32, 176)` float32 |
| `SurfaceCNN` | Conv 블록 4개 + GAP + Linear(64→1), 파라미터 약 6만 개 (약 250KB) |
| `pick_device(name)` | `auto`면 CUDA → MPS → CPU |

```
입력 (1, 32, 176)
 → Conv3×3+BN+ReLU+MaxPool  (16, 16, 88)
 → Conv3×3+BN+ReLU+MaxPool  (32,  8, 44)
 → Conv3×3+BN+ReLU+MaxPool  (64,  4, 22)
 → Conv3×3+BN+ReLU          (64,  4, 22)   ← 위치별 특징 지도 (heatmap 재료)
 → GAP (88칸 평균)           (64,)
 → Linear(64→1)              logit → sigmoid = NG 확률(cnn_prob)
```

- 라벨: `LABELS = ("OK", "NG")` → OK = 0, NG = 1. 출력 확률은 **NG일 확률**입니다.
- GAP + Linear 구조라서, 마지막 특징 지도에 Linear 가중치를 곱하면 위치별 기여도(CAM heatmap)를 바로 구할 수 있습니다 (6단계 예정).
- CPU에서 타일 1장 추론에 약 0.7ms가 걸립니다.

---

## 5. 현재 성능과 알려진 한계

| 평가 데이터 | 결과 (`side_surface_cnn_101310`, threshold 0.5) |
|---|---|
| test split (10/06 세션, 1000장) | 정확도 100%, 오류 0장 |
| val split (10/06 세션, 1000장) | 오류 1장 (`NG_Back_..._0319`, 확률 0.083) |
| **다른 날짜 Side TIFF** (06/15·06/16·07/14·09/02, 322장) | **OK 177장 중 47장 오검출(27%), NG 145장 중 9장 놓침(6%)** |

- train/val/test가 모두 **같은 날(10/06) 같은 세션**에서 나눠졌기 때문에, test 100%는 "같은 조건"에서의 성능입니다.
- 클래스마다 촬영 세션이 하나씩이라, 모델이 결함 대신 **세션 특유의 밝기·크롭 테두리**를 배웠을 가능성이 있습니다.
  07/14 OK 중 오검출된 타일은 테두리에 어두운 띠가 있었습니다.
- `NG_Back_..._0319`처럼 다이 단위로는 NG지만 **표면에는 결함이 보이지 않는** 샘플이 있을 수 있습니다 (라벨 검토 필요).

### 개선 방향

1. 다른 날짜 데이터를 **날짜 단위**로 외부 평가셋으로 만들고, 개선은 그 기준으로 비교
2. 여러 날짜 세션을 학습에 포함
3. 크롭 테두리 문제 확인, 라벨 검토
4. 증강(평행이동·밝기 변화) 추가
5. 그 뒤 threshold를 외부 val 기준으로 결정

---

## 6. 앞으로 할 일 (예정)

| 단계 | 내용 | 파일 |
|---|---|---|
| 5단계 | val로 threshold 결정, test 평가, 오분류 CSV, 설정 `.json` 저장 (모델과 같은 이름) | `tools/train_classifier.py` |
| 6단계 | `load_model()`, `predict()` — 샘플마다 아래 dict 반환 | `side/classifier.py` |

```python
{"cnn_prob": float, "cnn_defect": bool, "heatmap": np.ndarray(float32, 입력 타일 H×W, 0~1), "threshold": float}
```

---

## 7. 자주 나는 문제

| 증상 | 원인 / 해결 |
|---|---|
| `이미지가 없습니다: ... (OK/, NG/ 하위 폴더 필요)` | 경로가 틀렸거나 폴더 구조가 다름. `DEFAULT_DATA` 또는 `--data` 확인 |
| `ModuleNotFoundError: No module named 'tools'` / `'side'` | 프로젝트 루트가 아닌 곳에서 실행했거나, `sys.path.insert(0, str(ROOT))`보다 앞에서 import함 |
| `the following arguments are required: --model` | `eval_classifier.py`는 `--model`이 필수 |
| `비율이 잘못되었습니다: ... (합 1.3)` | `make_data.py`의 `--train/--val/--test`는 개수가 아니라 비율. 합을 1 이하로 (예: 0.7/0.15/0.15, 테스트만: 0/0/1.0) |
| `KeyError: 'sample_id'` (CSV 읽을 때) | BOM 때문. `encoding="utf-8-sig"`로 열기 |
| Windows에서 이미지가 `None` | 한글 경로 문제. 경로를 영문으로 바꾸기 |
| 학습 그래프의 한글이 □로 깨짐 | matplotlib 기본 폰트에 한글 없음. 라벨은 영문 유지 |
| MPS 결과가 이상함 | `--device cpu`로 비교 |
