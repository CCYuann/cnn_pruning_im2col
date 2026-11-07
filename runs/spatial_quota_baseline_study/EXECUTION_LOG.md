# 基準粒度審核：工作與問題處理

## 2026-11-02 00:47：原文核對

實際閱讀Figure4/5與對應文字。確認cluster共享mask，但空間subcluster的配額、threshold及整數處理未唯一明確。不能把現有flatten-K判為錯誤，也不能把新增spatial解讀直接當作者原始實作。有限搜尋未取得作者程式，未據此推斷沒有公開程式。

## 00:52：第一階段啟動

- 新增獨立spatial_quota_pruning.py及控制器，未修改主訓練核心或以前的研究腳本。
- 每(r,s) ideal quota取global reference剩餘權重數/cluster filters；largest remainder使空間quota總和精確匹配原flat cluster kept_count。空間整數修正是此實驗自己的公平比較定義。
- 22項tests通過；新增6項涵蓋配額、1×1相同、ties、partial cluster、不同kernel、空／滿mask與無效輸入。
- 首批8個seed0分支，兩模型、兩criteria、兩粒度，全重新跑10epochs。GPU1/0各一個工作，啟動時空閒。
- 在mask產生時與現有核心逐元素比對flat版本；兩版1×1必須相同。記錄mask變動數及各空間配額。
- 預先鎖定：任一criterion差距>=0.3pp時，同模型所有對照都補seeds1/2；不是只補較高者。若三seed仍有大差異，優先報基準敏感性，暫緩分類梯度pilot；否則另凍結梯度方法再啟動。

## 執行狀態

即時進度見status.json、execution.jsonl及logs/。此處僅記錄經確認的進展，不提前宣稱精度改善。只評估validation，不用test調參。

## 00:59：基準階段完成

| Model | Criterion | Flat val (%) | Spatial val (%) | 差異 (pp) |
|---|---|---:|---:|---:|
| ResNet34 | l1_sum | 94.50 | 94.64 | +0.14 |
| ResNet34 | top2 | 94.76 | 94.72 | -0.04 |
| ResNet50 | l1_sum | 93.46 | 93.40 | -0.06 |
| ResNet50 | top2 | 93.42 | 93.68 | +0.26 |

全部8個新訓練工作完成。配額與1×1一致性檢查通過，兩個工作各自生成的flat/spatial masks逐元素一致。ResNet50 flat l1也重現上一輪93.46%的validation控制。沒有任何criterion達到預先0.3pp擴大門檻，未補seeds1/2；這不是等效或顯著性結論。

## 01:05：接續分類梯度pilot

依預先決策，另在`runs/adapted_gradient_mask_study/`啟動ResNet50 seed0的2+8epochs共享gate梯度候選實驗。該階段另凍結計畫、程式及輸出，不混入本表。主方法須高於本輪四個ResNet50控制中最高的93.68%至少0.1pp，且不低於無門檻／隨機消融，才值得跨seed確認。
