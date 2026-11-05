# 共享Mask補償：便宜控制與長微調的研究判斷

## 本輪完成範圍

2026-11-05完成階段0/1的三-seed最小證據鏈，不是整個後續研究計畫全部完成。ResNet50/CIFAR-10、Tm8、三個獨立Dense100 checkpoints、同一45k train/5k val split。CIFAR stem為3x3 stride1、無初始maxpool。所有53個一般卷積含首層、1x1/downsample皆剪；FC/BN不剪。

A/BN/scalar/B/C每seed使用完全相同flat l1/global_cluster共享mask，實際卷積權重稀疏度約89.9994/90.0007/90.0016%。不是之前分組實驗的layer-uniform q。Global U只同總權重；quota U每filter q及逐層MAC亦匹配，允許每filter選不同位置。兩者是經驗參考，非分類精度數學上界。

共24條軌跡：18個主要方法/seed組合連續到FT30，6個非共享參考到FT10。A/B/C九個既有epoch10完整state重用並續訓；沒有重新初始化optimizer/scaler/RNG。FT0/1/3/10/30預先指定，不挑最高epoch，不調lambda追高。

所有數字為validation。原test早期已使用，本輪沒有新test forward，不能稱從未接觸test。校正fit512/check512來自train且互不重疊，不混入val/test。FT RNG=seed+100000，每Dense seed內配對一致。

## 完整比較

下表為三個獨立Dense seeds的validation平均（%）：

| 方法 | FT0 | FT1 | FT3 | FT10 | FT30 |
|---|---:|---:|---:|---:|---:|
| A 原保留權重 | 10.707 | 92.707 | 93.587 | 93.633 | 94.020 |
| BN-only | 75.453 | 92.707 | 93.587 | 93.633 | 94.020 |
| Per-filter scalar | 39.713 | 92.607 | 93.300 | 93.740 | 94.040 |
| B Dense-input ridge | 81.540 | 93.560 | 93.900 | 94.133 | 94.253 |
| C student-prefix ridge | 91.467 | 93.633 | 93.907 | 94.160 | 94.080 |
| Global unstructured | 60.507 | 94.480 | 94.560 | 94.600 | 未跑 |
| Quota-matched unstructured | 15.867 | 94.073 | 94.307 | 94.580 | 未跑 |
| Dense extra | 94.833 | 94.793 | 94.907 | 94.900 | 94.980 |

sample std及全部逐seed數據見mean_results.csv與validation_long.csv；paired std不是各模型std相減。

## 1. B的收益超出便宜控制，但長微調後縮小

| 配對差值 | FT1 mean +/- paired std (pp) | FT10 | FT30 |
|---|---:|---:|---:|
| B-A（亦等於B-BN） | +0.853 +/- 0.242 | +0.500 +/- 0.399 | +0.233 +/- 0.351 |
| B-scalar | +0.953 +/- 0.253 | +0.393 +/- 0.300 | +0.213 +/- 0.300 |

B-scalar在FT10三seed差值為+0.38/+0.70/+0.10pp，FT30為+0.56/+0.04/+0.04pp。B-A在FT30為+0.60/+0.20/-0.10pp，存在明確反例，不寫成每seed都改善。

支持的主張：固定共享mask及相同微調預算下，B對原權重、BN-only及scalar有初步平均收益；收益在低/中預算較突出，FT30縮小，並非單調隨epochs變化。沒有收斂停止準則，不能稱FT30完全收斂，也不能宣稱最終精度普遍提升。

尚未證明達到固定目標精度的總時間更短。校正、求解、模型/資料載入與診斷成本都要計入，不能直接由少epochs換算speedup。

## 2. BN不是FT10收益的替代解釋

BN預校正大幅改善FT0，但在本設定的正常training BN下，A與BN從FT1起validation精度完全相同。FT10/30三seeds的所有parameters、optimizer、scaler及RNG逐bit相同。部分zero-output通道的BN running buffers仍相差最多5.605e-45，因此不宣稱整個state_dict bit-exact；num_batches_tracked不同亦不作推論差異。

A/BN是回答不同機制問題的控制，不是兩份獨立統計證據。原始FT0的10.7到91.5大差距不能全包裝成權重補償：共同BN校正副本的舊結果A約75.45、B90.12、C90.45，也應一併閱讀。BN副本未帶回B/C主FT。

## 3. C不是較好的最終主線

C-B配對平均在FT0為+9.927pp、FT1為+0.073pp、FT3為+0.007pp、FT10為+0.027pp；FT30反為-0.173 +/- 0.133pp，三個seeds均低於B。

因此較簡單B維持主線。C的可能價值在zero/極少微調場景，但FT0差距受BN一致性影響，且尚未經更多seeds或獨立test確認；不因FT0較高就稱全面優於B。

## 4. 拆解剪枝與共享位置的經驗差距（只限FT10）

- Dense extra 94.900%，global U 94.600%，前者減後者為0.300pp。
- Quota U 94.580%，A 93.633%，同逐filter q/逐層MAC參考差距為0.947pp。
- B 94.133%，對quota U仍差0.447pp，補回約52.8%的上述平均差距（ratio of mean gaps，不是平均逐seed百分比）。

這是目前L1共享mask實作與獨立top-q之間的經驗差距，並非共享結構不可避免的最低損失，也不是準確率提升52.8%或錯誤率降低52.8%。兩個非共享參考沒跑FT30，不把不同FT預算混算成FT30共享代價。

## 5. 成本與數值稽核

本輪scalar離線平均6.64秒，BN約0.137秒；B/C沿用原紀錄約25.66/35.30秒。舊B/C與新scalar測量日期/共享負載不同，不能用時間比值宣稱效率倍數。FT30訓練路徑約380至397秒/分支，仍不含完整準備/診斷/排隊；詳見costs.csv。

63項tests通過。訓練前核對來源/資料分離、fresh shared mask、global threshold、quota top-q、scalar alpha與check fallback、BN只改buffers。訓練後42個FT10/30 checkpoints、2226個Conv層狀態稽核：禁止位置非零為0，保留位置恰為0亦為0，所以本批數值稀疏度與結構mask稀疏度一致。一般情況兩者仍須分開，保留位置為0不自動節省kept-K排程週期。

## 問題與處理

1. 原收尾equal_nested沒有處理NumPy RNG array，會拋出ambiguous truth ValueError。提前發現後保持凍結worker/controller與所有訓練不變，另建report-only finalizer以np.array_equal覆核，附兩個回歸tests及來源SHA。沒有因此重算模型。
2. 報表程序NumPy先import造成MKL_THREADING_LAYER=INTEL與libgomp衝突。保留UNIT_TESTS_mkl_failure.txt，使用GNU threading layer重跑測試/報表，63項通過。訓練環境及結果未改。
3. 完整BN state不逐bit相同：定位為5.6e-45的running-buffer殘值；分開記錄parameters及functional buffers，不把數值近似當bit-exact。見BN_A_PARAMETER_AUDIT.json。

重建：`env MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/finalize_compensation_controls.py`，之後以相同環境執行`scripts/audit_compensation_control_outputs.py`。報表重建不等於新訓練。

## 下一輪決策與尚未做的部分

本輪達預先5seed投入資格：FT10及FT30中B對BN/scalar皆平均>=0.2pp，至少2/3seeds同時對兩者為正。這是投入門檻，不是顯著性檢定；n=3與paired std仍不足以宣稱穩定普遍優勢。

建議下一輪新增兩個獨立Dense訓練擴到5seeds，先不改方法；再以有限第二模型確認。尚未執行：5seeds、其他模型、80/95%、512/2048校正量與lambda消融、獨立test確認、INT8及補償感知mask選擇。分組搜尋與activation加權不再投入新實驗。

補償感知mask仍為延伸。未來需對齊regularized reconstruction目標、驗證單步與批次刪除代價並處理Gram奇異性；共享分解次數減少不等於總時間8倍，也不是已確認新穎性。本輪只建立主線的最小可信證據，不宣稱以上延伸已完成。
