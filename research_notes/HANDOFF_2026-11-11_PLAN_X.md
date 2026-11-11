# 交接：計畫 X（封閉原論文流程 LR 網格的下界）

寫給接手的 AI 代理。本文件可單獨閱讀。
環境、刪檔規則和工作規則都和以前相同，細節見 `research_notes/HANDOFF_2026-11-09.md` 第 3、8 節。

## 1. 背景

計畫 W（`runs/other_datasets/PLAN_W.md`）用 validation 幫原論文流程（ft_const：原論文的 mask 規則、一次剪完、用固定 LR 微調）選 LR，候選是 0.001、0.003、0.01、0.03。

結果是 8 個模型裡有 7 個選到最小的 0.001，而且 validation 隨 LR 變大一路下降。所以網格的下界還沒封閉，更小的 LR 有可能更好。

計畫 X 補測 LR 0.0003，共 24 條。計畫已凍結在 `runs/other_datasets/PLAN_X.md`，**不要改門檻、候選或選 LR 的規則**。

## 2. 你的任務

1. 讀 `runs/other_datasets/PLAN_X.md`。
2. **重啟佇列 daemon。** 舊的 daemon 已在 2026-11-11 01:25 到期結束，所以要先重啟：
   ```
   cd /home/yenmh3/11457019/cnn_pruning_im2col
   nohup /usr/local/anaconda3/bin/python3 scripts/gpu_queue.py --queue runs/breakthrough_r0p95/queue.txt --logs runs/breakthrough_r0p95/logs --per-gpu 4 --hours 40 --cwd . > runs/breakthrough_r0p95/queue6.out 2>&1 &
   ```
   已經有 `.rc` 的舊工作不會重跑。重啟後，用 `nvidia-smi` 確認只有新工作在跑。
3. 排入工作：
   ```
   /usr/local/anaconda3/bin/python3 runs/breakthrough_r0p95/enqueue_x.py
   ```
   應該輸出 `24 queued`。
4. 等 24 條跑完。預計 1.5–2 小時，DenseNet121、ResNet50 最慢，已排在最前面。查看進度：
   ```
   echo "$(ls runs/breakthrough_r0p95/logs/x_*.rc 2>/dev/null | wc -l)/24"
   grep -L "^0$" runs/breakthrough_r0p95/logs/x_*.rc    # 失敗的工作
   ```
   失敗時先看 log。如果是暫時性錯誤（例如 OOM、磁碟滿），刪掉該 `.rc` 讓它重跑；不要改任何參數。
5. **先選 LR，再讀 test。** 兩步要分開跑，各自的輸出也要另外存檔：
   ```
   /usr/local/anaconda3/bin/python3 scripts/decide_plan_x.py --select > runs/other_datasets/PLAN_X_SELECTION.txt
   /usr/local/anaconda3/bin/python3 scripts/decide_plan_x.py > runs/other_datasets/PLAN_X_DECISION.txt
   ```
   - `--select` 只讀 validation，會寫出 `PLAN_X_LR_SELECTION.json`。這個檔一旦存在就不能覆寫，選擇也就凍結了。
   - 第二步讀 test，判定 X1 和 X2。
6. 把結果寫進 `PLAN_X.md` 檔尾的「## 結果」，並更新統整文件（第 4 節）。
7. 用繁體中文向使用者回報，結論先講。
8. 不要 commit 或 push，除非使用者要求。

## 3. 判定重點

- **X1（主判定）：** gradual+floor 對「選定 LR 的 ft_const」。門檻是配對平均 >= +0.3pp，而且 3 個 seeds 都為正。X1 會取代 W1，成為論文中「對原論文流程」的主數字。
- **沿用 W1 的模型：** 選定 LR 和計畫 W 相同的模型，腳本會標 `[same as W, reused]`。這些 test 之前已讀過，要寫明是沿用 W1，不是新結果。
- **改選 0.0003 的模型：** 腳本會標 `[NEW]` 和 `(LOWER BOUND)`。報告要註明「下界仍未封閉」，**不要再追加更小的 LR**。
- **X2（只描述）：** rewind 對選定 LR 的 ft_const。
- **0.0003 的 test：** 沒選中 0.0003 的模型，它的 test 只列在附表，不能拿來改選 LR。

## 4. 要更新的文件

1. **`runs/other_datasets/PLAN_X.md`：** 在檔尾加「## 結果」，格式比照 `PLAN_W.md`。
   - 主表欄位：模型、5 個 LR 的 validation、選定 LR、是否沿用 W、選定 ft test、floor test、X1 配對差與判定、X2。
   - 附表：5 個 LR 的 test。
2. **`research_notes/EXPERIMENT_SUMMARY_2026-11-07.md`：**
   - 5.9 節（計畫 W）的「下界未封閉」段落後，補一句 X 的結論。
   - 新增 5.10「計畫 X」。
   - 第 1 節第 1 條如果引用 W1，改為 X1；X1 和 W1 相同時，寫「W1 經下界補測後不變」。
   - 第 12 節檔案索引加入 `PLAN_X.md` 和它的 SELECTION、RESULTS、DECISION 檔。
3. **`research_notes/RESEARCH_FINDINGS_MASTER_2026-11-05.md`：** 在最後一條之後新增一條「計畫 X」。

## 5. 報告注意事項

- 只有浮點精度：worker 在 CUDA 用 AMP float16 評估。新模型沒有 Q15 流程。
- 原論文沒有給 LR。選出的 LR 只在 0.0003–0.03 範圍內成立，不代表作者實際用的配方。
- 這是事後補的網格點：它是看了計畫 W 的 validation 之後才決定要補的，報告要寫明。不過選擇規則沿用 W，而且是在讀 test 之前就凍結。
- 每組 n=3。門檻不是統計顯著性檢定。負面結果照實報告。

## 6. 本次新增、尚未 commit 的檔案

- `runs/other_datasets/PLAN_X.md`
- `runs/breakthrough_r0p95/enqueue_x.py`
- `scripts/decide_plan_x.py`：已用現有資料試跑，顯示 pending，運作正常。
- 本交接文件

worker 不需要修改：計畫 W 加的 `--ft-lr` 已經支援 0.0003，目錄名會是 `ft_const_r90_lr0.0003`。
