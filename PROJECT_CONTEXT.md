# 豆瓣電影評論情緒分析專案交接說明

這份文件是給其他 Codex 或協作者快速理解本專案用的。專案目標是使用豆瓣電影評論資料做中文情緒分類，並比較傳統機器學習模型與深度學習模型的效果。

## 專案目標

- 資料來源：豆瓣電影評論資料 `DMSC.csv`
- 任務：中文電影評論情緒分類
- 輸入：一段中文電影評論
- 輸出：
  - `negative`：負面評論
  - `positive`：正面評論
- 期末報告重點：
  - 資料前處理
  - TF-IDF + Logistic Regression baseline
  - Chinese BERT 深度學習模型
  - 兩種模型結果比較

## 環境

使用 conda 環境：

```powershell
conda activate project
```

若要在未 activate 的情況下執行，可用：

```powershell
conda --no-plugins run -n project python <script_name.py>
```

目前已確認：

- PyTorch 可使用 CUDA
- GPU 顯示為 `NVIDIA GeForce RTX 5070`
- `transformers` 已安裝
- `bert-base-chinese` 已下載並快取過

## 主要檔案

```text
C:\Users\Hastur\Desktop\codex
├── DMSC.csv
├── dashboard_app.py
├── preprocess_douban.py
├── predict.py
├── outputs/manual-20260608-douban/presentations/douban-sentiment-deck/output/douban-sentiment-analysis.pptx
├── train_tfidf_model.py
├── train_bert_model.py
├── PROJECT_CONTEXT.md
├── data
│   └── processed
│       ├── douban_movie_reviews_clean.csv
│       └── preprocess_summary.json
└── models
    ├── tfidf_logreg
    ├── tfidf_logreg_full
    ├── bert_chinese_200k
    ├── bert_chinese_full
    └── bert_chinese_smoke
```

## 資料前處理

前處理程式：

```text
preprocess_douban.py
```

執行指令：

```powershell
python preprocess_douban.py --input DMSC.csv
```

輸出：

```text
data/processed/douban_movie_reviews_clean.csv
data/processed/preprocess_summary.json
```

前處理邏輯：

- 讀取原始 `DMSC.csv`
- 清理評論文字：
  - HTML unescape
  - Unicode normalize
  - 移除 HTML tag
  - 移除 URL
  - 移除控制字元
  - 合併多餘空白
- 移除空評論
- 移除過短評論
- 移除重複評論
- 將星等轉成情緒標籤：
  - 1-2 星：`negative`
  - 4-5 星：`positive`
  - 3 星：預設移除，作為中立資料不進入二分類

前處理結果：

| 項目 | 數量 |
|---|---:|
| 原始資料列數 | 2,125,056 |
| 清洗後資料列數 | 1,589,717 |
| negative | 358,829 |
| positive | 1,230,888 |
| 移除 3 星中立評論 | 473,473 |
| 移除重複評論 | 53,829 |
| 移除空評論 | 161 |
| 移除太短評論 | 7,876 |

注意：目前後續訓練實際讀到的 rows used 是 `1,589,714`，比前處理 summary 少 3 筆，可能是 CSV 重新讀取時有少數空值或格式細節被 drop 掉。這不影響整體訓練。

## TF-IDF + Logistic Regression baseline

訓練程式：

```text
train_tfidf_model.py
```

模型設定：

- 特徵：TF-IDF
- analyzer：`char`
- ngram：2 到 4
- max features：300,000
- classifier：Logistic Regression
- `class_weight="balanced"`

### 平衡抽樣版

指令：

```powershell
python train_tfidf_model.py
```

預設參數：

- negative 最多 200,000
- positive 最多 200,000
- 總資料 400,000
- test size 0.2

輸出資料夾：

```text
models/tfidf_logreg
```

結果：

| 指標 | 數值 |
|---|---:|
| Accuracy | 0.8821 |
| Macro F1 | 0.8821 |
| Weighted F1 | 0.8821 |

### 全資料版

指令：

```powershell
python train_tfidf_model.py --max-per-class 0 --test-size 0.2 --output-dir models/tfidf_logreg_full
```

輸出資料夾：

```text
models/tfidf_logreg_full
```

結果：

| 指標 | 數值 |
|---|---:|
| Rows used | 1,589,714 |
| Train rows | 1,271,771 |
| Test rows | 317,943 |
| Accuracy | 0.9006 |
| Macro Precision | 0.8481 |
| Macro Recall | 0.8920 |
| Macro F1 | 0.8666 |
| Weighted F1 | 0.9035 |

分類報告重點：

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.7345 | 0.8763 | 0.7992 | 71,766 |
| positive | 0.9618 | 0.9077 | 0.9339 | 246,177 |

解讀：

- TF-IDF full 的整體 accuracy 約 0.90。
- 因為資料不平衡，weighted F1 較高，macro F1 較低。
- negative recall 高，代表能抓到大部分負面評論，但 negative precision 較低。

## Chinese BERT 深度學習模型

訓練程式：

```text
train_bert_model.py
```

模型設定：

- 預訓練模型：`bert-base-chinese`
- 任務：Sequence Classification
- label：
  - `negative` -> 0
  - `positive` -> 1
- max length：128
- optimizer：AdamW
- learning rate：2e-5
- epochs：2
- batch size：32
- eval batch size：64
- warmup ratio：0.06
- weight decay：0.01
- GPU：CUDA
- 使用 weighted CrossEntropyLoss，處理不平衡資料

### Smoke test

這只用來確認流程可跑，不代表模型效果。

```powershell
python train_bert_model.py --max-per-class 32 --test-size 0.25 --epochs 1 --batch-size 8 --eval-batch-size 16 --max-length 64 --output-dir models/bert_chinese_smoke
```

輸出資料夾：

```text
models/bert_chinese_smoke
```

### 平衡 200k 版

指令：

```powershell
python train_bert_model.py --max-per-class 100000 --test-size 0.1 --epochs 2 --batch-size 32 --eval-batch-size 64 --max-length 128 --output-dir models/bert_chinese_200k
```

輸出資料夾：

```text
models/bert_chinese_200k
```

結果：

| 指標 | 數值 |
|---|---:|
| Rows used | 200,000 |
| Train rows | 180,000 |
| Test rows | 20,000 |
| Accuracy | 0.9031 |
| Macro Precision | 0.9033 |
| Macro Recall | 0.9030 |
| Macro F1 | 0.9030 |
| Weighted F1 | 0.9030 |
| Training time | 965.77 sec |

分類報告：

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.9136 | 0.8903 | 0.9018 | 10,000 |
| positive | 0.8930 | 0.9158 | 0.9043 | 10,000 |

解讀：

- 這是平衡資料實驗。
- positive 和 negative 的表現非常接近。
- 適合用來展示 BERT 在平衡情境下的穩定分類能力。

### 全資料版

指令：

```powershell
python train_bert_model.py --max-per-class 0 --test-size 0.1 --epochs 2 --batch-size 32 --eval-batch-size 64 --max-length 128 --output-dir models/bert_chinese_full
```

輸出資料夾：

```text
models/bert_chinese_full
```

結果：

| 指標 | 數值 |
|---|---:|
| Rows used | 1,589,714 |
| Train rows | 1,430,742 |
| Test rows | 158,972 |
| Accuracy | 0.9359 |
| Macro Precision | 0.9032 |
| Macro Recall | 0.9172 |
| Macro F1 | 0.9099 |
| Weighted F1 | 0.9365 |
| Training time | 7,791.57 sec |

分類報告：

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.8409 | 0.8831 | 0.8615 | 35,883 |
| positive | 0.9654 | 0.9513 | 0.9583 | 123,089 |

每個 epoch 的測試集表現：

| Epoch | Train Loss | Test Accuracy | Test Macro F1 |
|---:|---:|---:|---:|
| 1 | 0.2530 | 0.9291 | 0.9013 |
| 2 | 0.1835 | 0.9359 | 0.9099 |

解讀：

- 這是目前最強的模型成果。
- BERT full 明顯優於 TF-IDF full。
- negative F1 從 TF-IDF 的 0.7992 提升到 0.8615。
- macro F1 從 TF-IDF 的 0.8666 提升到 0.9099。
- weighted F1 從 TF-IDF 的 0.9035 提升到 0.9365。

## 最適合放進報告的模型比較

| 模型 | 資料設定 | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 全資料，不平衡 | 0.9006 | 0.8666 | 0.9035 |
| Chinese BERT | 全資料，不平衡 | 0.9359 | 0.9099 | 0.9365 |

建議報告文字：

> 本專案先以 TF-IDF + Logistic Regression 作為傳統機器學習 baseline，再使用 `bert-base-chinese` 進行深度學習 fine-tuning。實驗結果顯示，BERT 在全資料測試集上取得 93.59% accuracy 與 90.99% macro F1，優於 TF-IDF baseline 的 90.06% accuracy 與 86.66% macro F1。這表示 BERT 對中文評論中的語意與情緒表達具有更好的辨識能力。

## Test Accuracy 與 Training Accuracy 說明

目前程式輸出的 classification report 與 accuracy 都是 **test accuracy**。

原因：

- 訓練資料切分成 train/test
- 模型訓練完後，程式呼叫 `evaluate(model, test_loader, device)`
- classification report 是根據 test set 預測結果計算

所以報告中應寫：

```text
Test Accuracy = 93.59%
Test Macro F1 = 90.99%
```

目前 BERT 程式有記錄每個 epoch 的：

- train loss
- test accuracy
- test macro F1

目前沒有額外計算 training accuracy。若需要判斷 overfitting，可以加入對 `train_loader` 的 evaluate，但全資料會花很多時間，報告中通常用 train loss + test metrics 即可。

## 常用指令

### 前處理

```powershell
python preprocess_douban.py --input DMSC.csv
```

### TF-IDF baseline，全資料

```powershell
python train_tfidf_model.py --max-per-class 0 --test-size 0.2 --output-dir models/tfidf_logreg_full
```

### BERT，平衡 200k

```powershell
python train_bert_model.py --max-per-class 100000 --test-size 0.1 --epochs 2 --batch-size 32 --eval-batch-size 64 --max-length 128 --output-dir models/bert_chinese_200k
```

### BERT，全資料

```powershell
python train_bert_model.py --max-per-class 0 --test-size 0.1 --epochs 2 --batch-size 32 --eval-batch-size 64 --max-length 128 --output-dir models/bert_chinese_full
```

### 單句評論預測

預設使用 BERT full 的最佳模型：

```powershell
python predict.py --text "這部電影很好看，劇情感人，演員表現也很棒"
```

互動模式：

```powershell
python predict.py --interactive
```

使用 TF-IDF full 模型：

```powershell
python predict.py --model-type tfidf --text "劇情很無聊，節奏拖沓，看到一半就想離場"
```

### Streamlit 網頁儀表板

啟動儀表板：

```powershell
streamlit run dashboard_app.py
```

或明確使用 project 環境：

```powershell
python -m streamlit run dashboard_app.py --server.port 8501
```

本機網址：

```text
http://localhost:8501
```

儀表板內容：

- 專案資料總覽
- TF-IDF 與 BERT 整體結果比較
- precision / recall / F1 多圖表比較
- confusion matrix 圖片展示
- 不同電影 1-5 星統計
- 依平均星等、推薦率、Bayesian 星等判斷推薦 / 可考慮 / 不推薦
- 單句中文評論 positive / negative 測試

### 如果 GPU 記憶體不足

把 batch size 從 32 改成 16：

```powershell
python train_bert_model.py --max-per-class 0 --test-size 0.1 --epochs 2 --batch-size 16 --eval-batch-size 32 --max-length 128 --output-dir models/bert_chinese_full_bs16
```

## 模型輸出內容

TF-IDF 輸出：

```text
models/tfidf_logreg_full
├── tfidf_logreg_model.joblib
├── metrics.json
├── classification_report.txt
└── confusion_matrix.png
```

BERT 輸出：

```text
models/bert_chinese_full
├── best_model
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── vocab.txt
├── final_model
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── vocab.txt
├── labels.joblib
├── metrics.json
├── classification_report.txt
└── confusion_matrix.png
```

## 建議下一步

1. 建立 `predict.py`
   - 輸入一句中文評論
   - 輸出 positive / negative
   - 可選擇載入 TF-IDF 或 BERT 模型

2. 整理投影片
   - 問題定義
   - 資料集介紹
   - 前處理流程
   - TF-IDF baseline
   - BERT model
   - 結果比較表
   - confusion matrix
   - conclusion

3. README 整理
   - 將本文件濃縮成正式 `README.md`
   - 補上執行環境、指令、結果表格

4. 簡報
   - 已產出 PPTX：
     `outputs/manual-20260608-douban/presentations/douban-sentiment-deck/output/douban-sentiment-analysis.pptx`
   - 生成腳本：
     `outputs/manual-20260608-douban/presentations/douban-sentiment-deck/slides/build_deck.mjs`
   - 內容包含資料前處理、TF-IDF baseline、BERT full、confusion matrix、電影推薦統計、單句 demo 與反諷錯誤分析

## 注意事項

- `DMSC.csv` 與模型檔都很大，若要上傳 GitHub，建議不要直接 commit 大檔案。
- 可在 GitHub 上保留程式碼、README、結果表格與圖片。
- 大型資料與模型可用雲端硬碟連結，或在 README 說明需自行下載。
- `models/bert_chinese_full` 內每個 BERT 模型約 409MB，`best_model` 和 `final_model` 都會各存一份。
