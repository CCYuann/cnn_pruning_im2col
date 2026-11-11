# 交接：計畫 V（原因拆解的重現）與計畫 W（原論文流程的公平 LR）

寫給接手的 AI 代理。本文件可單獨閱讀。
環境、佇列、刪檔規則與先前相同，細節見 `research_notes/HANDOFF_2026-11-09.md` 第 3、8 節。

## 1. 背景（一段話）

CNN Cross-Filter Structured Pruning 與 mask-aware 加速器（V3）的研究。
- 每 8 個 filter 共用一個輸入位置 mask。
- 我們的方法是 gradual+floor：前 2/3 epochs 逐步剪，每個 cluster 至少留 1 個位置。
- 主線是 CIFAR-100、90% 剪枝率、8 個模型（計畫 S）；95% 是延伸（計畫 T）。

計畫 U（`runs/other_datasets/PLAN_U.md`）把「gradual+floor 對 rewind」的差拆成三步：

| 步驟 | 對照 | seeds 0/1/2 的結果 |
|---|---|---|
| 配額分配 | A1 − A0 | 7 組中 6 組為負 |
| 位置選擇 | A2 − A1 | 7 組中 6 組為正 |
| 逐步過程 | A3 − A2 | 7 組全部為正 |

這兩個計畫要補兩個缺口：
- **計畫 V：** 計畫 U 的結論要在新 seeds 重現。
- **計畫 W：** 原論文流程（ft_const）的 LR 0.001 是我們的假設。要用 validation 幫它選一個公平的 LR，避免審稿人質疑我們只贏了一個調得很差的對手。

## 2. 你的任務

1. 讀 `runs/other_datasets/PLAN_V.md` 和 `PLAN_W.md`。兩份都已凍結，**不要改門檻、LR 候選或選 LR 的規則**。
2. 確認佇列 daemon 還活著：`ps -p 254618`。它會在 2026-11-11 01:25 自動結束，重啟指令見 `HANDOFF_2026-11-09.md` 第 3 節。
3. 一次排入兩個計畫：
   ```
   cd /home/yenmh3/11457019/cnn_pruning_im2col
   /usr/local/anaconda3/bin/python3 runs/breakthrough_r0p95/enqueue_vw.py --with-w
   ```
   - 應輸出 `84 queued`：V 有 12 條（前綴 `v_`），W 有 72 條（前綴 `w_`）。
   - 重跑是安全的，已排入的不會重複。
4. 查看進度：
   ```
   echo "V $(ls runs/breakthrough_r0p95/logs/v_*.rc 2>/dev/null | wc -l)/12  W $(ls runs/breakthrough_r0p95/logs/w_*.rc 2>/dev/null | wc -l)/72"
   grep -L "^0$" runs/breakthrough_r0p95/logs/[vw]_*.rc    # 失敗的工作
   ```
   - 預估時間：V 約 30–40 分鐘；W 約 3–4 小時，DenseNet121、ResNet50 最慢，已排在最前面。
   - 失敗時先看 log。若是暫時性錯誤（OOM、磁碟滿），刪掉該 `.rc` 讓 daemon 重跑；不要改任何訓練參數。
5. 判定。兩個指令的輸出分別另存成 `runs/other_datasets/PLAN_V_DECISION.txt`、`PLAN_W_DECISION.txt`：
   ```
   /usr/local/anaconda3/bin/python3 scripts/decide_plan_v.py
   /usr/local/anaconda3/bin/python3 scripts/decide_plan_w.py
   ```
   V 先跑完時可以先判定 V，不用等 W。
6. 把結果寫進兩份計畫檔檔尾的「## 結果」，並更新統整文件（第 4 節）。
7. 用繁體中文向使用者回報，結論先講。
8. 不要 commit 或 push，除非使用者要求。

## 3. 判定重點

### 計畫 V（ResNet-56、VGG16-BN，95%，seeds 3/4/5）
- 腳本會直接印出三個主張是否重現（REPLICATED / not replicated）：
  - **V1 過程為正：** 兩個模型都要「正」。
  - **V2 位置為正：** 兩個模型都要「正」。
  - **V3 配額單獨為負：** ResNet-56 要「負」。VGG16-BN 在計畫 U 是「不確定」，只報告，不判定。
- 沒有重現的主張，要在統整文件把它從結論降級為「只在 seeds 0/1/2 觀察到」。
- A0（rewind）和 A3（gradual+floor）的 seeds 3/4/5 結果之前已讀過，只有 A1、A2 是新的。

### 計畫 W（8 個模型，90%，seeds 0/1/2）
- **選 LR：** 腳本只用 validation 選 LR，候選為 0.001、0.003、0.01、0.03，平手取較小的。選定後不能因為 test 結果改選。
- **W1（主判定）：** gradual+floor 對「選定 LR 的 ft_const」。門檻：配對平均 >= +0.3pp 且 3/3 為正。
  - W1 會取代計畫 S 的 S1，成為論文中「對原論文流程」的主數字。
  - S1（LR 0.001）保留，作為對照。
- **W2（只描述）：** rewind 對選定 LR 的 ft_const，看高 LR 重訓還有多少優勢。
- **上界：** 若選中 0.03，腳本會標 `UPPER BOUND`。報告時要註明「最佳 LR 可能更高」，**不要追加 LR**。
- **附表：** 其他 LR 的 test 精度只列在附表，不能用來改選。

## 4. 要更新的文件

1. **`PLAN_V.md`、`PLAN_W.md`：** 檔尾加「## 結果」，格式比照 `PLAN_U.md`、`PLAN_T.md`。
   - W 的主表欄位：模型、各 LR 的 validation、選定 LR、選定 LR 的 ft_const test、gradual+floor test、W1 配對差與判定、W2。
2. **`research_notes/EXPERIMENT_SUMMARY_2026-11-07.md`：**
   - 第 1 節第 3 條（計畫 U）補上 V 的重現結果。
   - 第 1 節第 1 條（90% 主線）加上 W1 的結果。
   - 新增 5.8（計畫 V）、5.9（計畫 W）。
   - 第 12 節檔案索引加入 `PLAN_V.md`、`PLAN_W.md` 和各自的 RESULTS、DECISION 檔。
3. **`research_notes/RESEARCH_FINDINGS_MASTER_2026-11-05.md`：** 在 #33 之後新增 #34（V）、#35（W）。

## 5. 一定要寫進報告的注意事項

- **只有浮點精度：** worker 在 CUDA 用 AMP float16 評估，新模型沒有 Q15 流程。
- **週期都是 V3 公式預測：** 每個 cluster 31 + Hout·ceil(Wout/8)·max(q,15) 拍，50 MHz，只算卷積層，不是 RTL 實測。
- **W 的 LR 只在 4 個值中選，** 不代表原論文作者實際用的設定。原論文沒有給 LR。
- **V 的拆解順序是固定的**（先換配額、再換位置、再換過程），交互作用沒有單獨估計。A2 從 Dense 重新訓練，A3 接續逐步剪枝後的權重，所以「過程」包含起點差異。
- **門檻不是統計顯著性檢定。** 每組 n=3。
- 負面結果照實報告，事後的想法要標明。

## 6. 本次新增或修改、尚未 commit 的檔案

- **`scripts/dataset_pruning_worker.py`：** 新增三個參數：
  - `--ft-lr`：只用於 ft_const，預設 0.001；不是 0.001 時，目錄名加 `_lr<值>`。
  - `--quota-from`、`--mask-from`：計畫 U 加的。
  - 舊指令的行為與目錄名都不變。`--ft-lr` 已用 2 epochs 的小測試確認 LR 有正確套用。
- **`scripts/summarize_datasets.py`：** 這三種新組別不列入舊的彙整表。
- **新檔：**
  - `runs/other_datasets/PLAN_U.md`、`PLAN_V.md`、`PLAN_W.md`
  - `runs/breakthrough_r0p95/enqueue_u.py`、`enqueue_vw.py`
  - `scripts/decide_plan_u.py`、`decide_plan_v.py`、`decide_plan_w.py`
  - 本交接文件
- 兩個判定腳本都已用現有資料試跑過，新組別顯示 pending，運作正常。
