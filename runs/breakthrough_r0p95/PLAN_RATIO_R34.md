# 更高剪枝率與ResNet34（凍結計畫）

凍結日期：2026-11-06，於本批結果產生之前寫下。5000張validation最後一epoch，不評估test，lambda固定3e-5不調參。

## A. 更高剪枝率（ResNet50，seed1/2）
97%、98%各跑rewind、group、group_gradual（共12條）。
回答：group類方法相對rewind的增益是否隨剪枝率變大；以「rewind@95%的94.15%」與「90%的94.98%」為參考線。
判定：增益平均 >= +0.5pp 且2/2為正，才列為「高剪枝率有明顯救回」。

## B. ResNet34/80%（硬體評估主體，seed0/1/2）
rewind、group、group_gradual（共9條）。
判定沿用H2：對rewind平均 >= +0.3pp 且3/3為正。未達成則ResNet34維持rewind，不把ResNet50結論外推。
必報：實際sparsity、MAC、q=0、全q=0層；ResNet34週期以同一張RTL校準表估算。

## C. DPF組合（探索，seed1/2，於結果產生前寫下）
dpf_group、dpf+mac beta0.25、dpf+mac beta0.5。動機：DPF位置可回復，也許能減輕MAC配額的精度損失。
參考線：ggrad_mac0.25 94.08%/47.35ms、ggrad_mac0.5 93.78%/40.71ms。同週期下平均高至少0.3pp才值得進下一輪；不算入確認組。

## D. 98%增益的來源與延伸（探索，seed1/2，2026-11-06）
寫下時已知：seed3/4/5的98% validation（group +0.48pp、group_gradual +1.17pp），test尚未彙整（log已印出seed3及seed4第一筆）。本節全部只用開發seed1/2的validation判定。
參考線：group_gradual_r98 dev 93.49%、rewind_r98 dev 92.62%。
- D1 拆解：gradual_r98、dpf_r98。回答增益來自漸進遮罩還是group懲罰。
- D2 99%：rewind_r99、group_gradual_r99。回答增益是否繼續隨剪枝率變大。
- D3 lambda：group_gradual_r98，lambda 1e-5、1e-4（原3e-5是95%選的）。
- D4 預算：group_gradual_r98，100 epochs。
- D5 ResNet34/90%：rewind_r34r90、group_gradual_r34r90。回答ResNet34在更高剪枝率是否出現同樣增益。
判定：D3、D4相對93.49%平均 >= +0.3pp 且2/2為正才列候選；D1、D2、D5只描述，增益 >= +0.5pp且2/2為正才值得新seeds確認。候選須用新Dense seeds重跑，test不參與本節任何選擇。

## D結果（seed1/2 validation，已判定）
| 組別 | 平均% | 對照差 | ms | MAC減少 | 判定 |
|---|---:|---:|---:|---:|---|
| gradual_r98 | 93.46 | 對rewind +0.84（+0.64/+1.04） | 31.49 | 95.53% | 值得確認 |
| dpf_r98 | 92.95 | 對rewind +0.33（+0.04/+0.62） | 30.26 | 95.15% | 不確認 |
| group_gradual_r99 | 92.39 | 對rewind_r99(90.84) +1.55（+1.38/+1.72） | 22.71 | 97.59% | 值得確認 |
| ggrad_r98 lam1e-5 | 93.76 | 對ggrad +0.27（+0.64/-0.10） | 31.58 | 95.54% | 未過 |
| ggrad_r98 lam1e-4 | 93.80 | 對ggrad +0.31（+0.52/+0.10） | 33.19 | 95.26% | 候選（剛過門檻） |
| ggrad_r98 e100 | 92.81 | 對ggrad -0.68 | 31.75 | 96.28% | 未過，全q=0層10 |
| ResNet34/90% ggrad | 94.49 | 對rewind(94.87) -0.38（0.00/-0.76） | 70.66對83.87 | 83.91%對80.14% | 無增益 |
解讀：gradual不加懲罰就得到幾乎全部增益，group懲罰在98%沒有可見貢獻。lambda 0/1e-5/3e-5/1e-4為93.46/93.76/93.49/93.80，非單調，看起來是雜訊。

## E. 確認與延伸（凍結日期2026-11-06，於本節任何結果產生之前寫下）
E1 新seeds確認（seed3/4/5，ResNet50，lambda預設3e-5）：
- 組別：gradual_r98、group_gradual_r98_lam1e-4、rewind_r99、group_gradual_r99、gradual_r99、unstructured_r99（僅參考線）。gradual_r99未在開發seed跑過，由gradual_r98的結果類推加入，同時在E2補seed1/2。
- 通過條件（validation，對同剪枝率rewind，n=3）：平均 >= +0.5pp 且3/3為正。
- group懲罰貢獻：group_gradual對gradual，平均 >= +0.3pp 且3/3才算有貢獻；否則論文主方法改用較簡單的gradual。
- lambda：1e-4對3e-5，平均 >= +0.3pp 且3/3才採用；否則維持3e-5，不再調lambda。
- 不論通過與否，模型各評估一次test，不再依test調整。
E2 探索（seed1/2 validation，test不參與）：
- gradual_r99；gradual_r98 e100（確認e100變差是否由懲罰累積造成）；gradual_r98 ramp 0.5與0.83（--ramp-frac）；gradual_r98+mac beta0.25。
- ResNet34：gradual於80%、90%；rewind與gradual於95%、98%。
- 判定：r98相關組相對gradual_r98(93.46)平均 >= +0.3pp 且2/2為正；ResNet34對同剪枝率rewind平均 >= +0.5pp 且2/2為正才值得確認。

## E1 validation判定（seed3/4/5，對同剪枝率rewind）
- gradual_r98 +0.993（+1.68/+0.56/+0.74）通過；group_gradual對gradual +0.18（3/3但未達+0.3），懲罰不算有貢獻，論文主方法改用gradual。
- lambda 1e-4對3e-5 -0.133，未過，維持3e-5，不再調lambda。
- gradual_r99 +2.027（+3.26/+1.08/+1.74）、group_gradual_r99 +1.767（+2.66/+0.84/+1.80）皆通過。

## F. q下限（探索，凍結日期2026-11-06，於E1 test結果被讀取之前寫下）
動機（事後）：gradual在seed4的98%、99%出現layer1全q=0層，而seed4增益最小；RTL週期模型把q=0當q=1計，全q=0層又依賴bias-only路徑與residual寬位元相加。
方法：--q-floor，每cluster至少留1個位置，總預算不變，名額取自全網最弱的其他已保留位置。
組別：gradual_r98_floor、gradual_r99_floor，seed1/2（開發）與seed3/4/5（新seeds，共10條）。
判定（seed3/4/5 validation，對同剪枝率gradual，n=3）：
- 非劣：平均 >= -0.1pp，且全q=0層數為0、q=0 cluster為0、週期增加 <= 3%，則硬體版主方法改用floor。
- 精度增益：平均 >= +0.3pp 且3/3為正，才可宣稱floor提高精度。
不論結果，seed3/4/5模型各評估一次test，不再依test調整。seed1/2只作開發紀錄，不參與判定。

## E2部分結果（ResNet34，seed1/2 validation，已判定）
| 剪枝率 | rewind %/ms | gradual %/ms |
|---|---:|---:|
| 80% | 95.33/（seed1/2未單獨列）| 95.06/122.98 |
| 90% | 94.87/83.87 | 94.45/69.70 |
| 95% | 94.40/49.75 | 94.15/38.24 |
| 98% | 94.02/25.50 | 93.04/18.34 |
ResNet34的gradual沒有對rewind的精度增益，判定未過（需>=+0.5pp）；同剪枝率下gradual週期少17至28%。ResNet34 rewind 98%（94.02%/25.5 ms）在開發seed上同時優於ResNet50 gradual 98%（新seed 93.30%/27.86 ms），但跨模型、跨seed，不是配對比較。

## G. ResNet34高剪枝率確認（凍結日期2026-11-06，seed3/4/5尚無任何ResNet34結果）
G0 新Dense：ResNet34 seed3/4/5，指令與ResNet50新seeds相同（100 epochs、split-seed 0、val 5000）。
G1 開發（seed1/2）：rewind_r34r99、gradual_r34r99，只描述。
G2 確認（seed3/4/5，ResNet34）：rewind 80%、95%、98%，gradual 98%，unstructured 98%（僅參考線）。
- 主問題：rewind 98%的test（FP32與Q15）與週期，相對ResNet50 gradual 98%（同為seed3/4/5，非配對）。判定：Q15 test較高且週期較低才可說ResNet34在此任務較好。
- 次問題：rewind 98%對rewind 80%的精度代價與週期收益（配對描述）。gradual 98%對rewind 98%（配對，週期對精度取捨）。
- 全部模型各評估一次test與Q15，不再依test調整。Q15校準方式依validation診斷定案後才評估G2。
