# 固定共享Mask的權重補償：執行前規劃草案

狀態：2026-11-02已獲授權執行，實作與驗證中；首次完整GPU實驗前凍結本計畫及來源。這是待驗證方向，不是已成立的改良，也不是OBC/SparseGPT的完整重現。

## 1. 要回答的問題

在分組、mask、每cluster kept_count與FT預算完全相同時，只改變保留權重的初始化值，是否改善分類accuracy或降低所需FT預算？先前換位置的負結果沒有直接回答此問題；但過去FT本來就會更新權重，不能說之前完全沒動權重。

權重補償已有先例；[OBC, NeurIPS2022](https://papers.neurips.cc/paper_files/paper/2022/hash/1caf09c9f4e6b0150b06a07e77f2710c-Abstract-Conference.html)提供post-training pruning/quantization脈絡。本草案採固定support的regularized reconstruction作比較，不把一般重建或ridge求解本身稱為原創。

## 2. 不變條件

- ResNet50、CIFAR-10、seed0先做pilot；32×32、原固定45000/5000/10000 split。
- 使用既有Dense100epoch checkpoint、Tm8、連續分組、flat-K l1 mask，actual conv-weight sparsity 89.9994336%。
- mask逐bit一致，不換kept_k、不重新分配、不加保護或cap，不修改bias、BN或分類器參數作離線補償。
- FT全分支10epochs，SGD lr.001、momentum.9、weight_decay.0005、batch128、AMP float16，optimizer/scaler/RNG起點相同；只評估validation。
- 目前此mask有43個q=0 clusters，不能恢復其連接；最大q=871。q=0保持零，q=K保留原值。這些情況獨立統計。

## 3. 三個分支，一次隔離兩個問題

| 分支 | 權重初始化 | 用途 |
|---|---|---|
| A original | Dense保留位置原值 | 完全相同mask的主要控制 |
| B dense-input compensation | 每層用Dense input patches重建Dense convolution輸出 | 檢查純固定support補償是否有效 |
| C sequential compensation（主方法） | 按forward順序，用目前已剪枝且已補償的student prefix輸入，重建同圖片／同座標的Dense teacher convolution輸出 | 檢查上游輸入偏移是否影響補償 |

C不能重用Dense input當student input；teacher/student的圖片與輸出座標必須對齊。離線補償全程eval模式，BN buffers保持原Dense值，排除同時改BN的混淆。C在每層完成後更新student，後層重新取得真實prefix輸入。

## 4. 固定support求解

每cluster的保留索引S固定，X_S為[q,P]，Y為[實際filter數,P]的Dense teacher線性卷積輸出，W0為Dense在S的原權重。所有ResNet50卷積bias本次不調整；若未來有bias需先扣除既有bias再建target。

```text
min_U (1/P)*||U X_S - Y||_F^2 + lambda*||U-W0||_F^2
G = X_S X_S^T / P
lambda = max(1e-8, 1e-3 * mean(diag(G)))
U = W0 + solve(G + lambda I, X_S (Y-W0 X_S)^T / P)^T
```

正則中心用W0，避免把保留權重任意推向零。使用linear solve／Cholesky，不顯式求inverse；小型CPU FP64 reference交叉驗證。lambda係數在看accuracy前固定，不反覆用validation挑。

執行細節：FP64求解，先cast回實際FP32儲存格式才判斷check MSE。Cholesky若數值失敗最多依序使用lambda×1/10/100，記錄attempt，不以分類accuracy選倍率。校正seed20261107，fit/check各512張、每張最多8個不重複座標；同一teacher/student使用完全相同位置。

每個cluster只有S可寫入，其他位置強制0；非有限解直接報錯。可預定數值jitter恢復，但必須記錄，不當成accuracy調參。若check重建誤差上升，相同規則回退該cluster原值；B/C都採此規則並回報回退數。

## 5. 校正資料

train內固定512張fit＋另外512張check，互斥並排除validation；不讀test。每層每split至多4096個patches，儘可能每張取不同spatial座標；實際樣本數與rank風險必須回報。

4096比現有1024-patch cache更有餘裕，不能直接假裝舊cache符合新設定。先試最大q及幾個不同kernel/stride層，確認記憶體、耗時與數值穩定後才跑完整補償。

check若用來回退，它就是開發資料，不是獨立泛化證據。不能由check MSE改善直接宣稱分類有效。

## 6. 執行順序

1. 不訓練：從既有控制紀錄整理FT0/2/10與完整epoch曲線；目前A為10.08%／93.22%／93.46% validation，epoch4曾93.66%，也需留意後段過擬合／波動，不只用FT抹平解釋。
2. 工程驗證：solve reference、patch/K順序、teacher/student座標、q=0/q=K、mask零位不變、BN不變、非有限值與回退規則。使用含padding/stride2/1×1/3×3的小型cases。
3. 校正pilot：生成B/C權重，報fit/check MSE、權重變化量、範圍與回退率、離線時間／峰值記憶體。
4. 三分支FT：A/B/C相同10epochs，保留0/2/10及每epoch validation。A重新跑以核對最終模型重現；不是直接混用另一流程的舊結果。
5. 若主方法有足夠訊號，再鎖定方法補seeds1/2；同時確認B與A，避免只確認較有利分支。無明確訊號就保留負結果，不擴模型或調更多lambda。
6. 浮點結果成立後才進INT8；cycle-aware配額分析暫列下一條獨立主線，不混進本次權重實驗。

## 7. BN診斷與評估公平性

FT0除了原始直接accuracy，另可在模型clone上對A/B/C都做相同校正資料、相同次數的BN校正，單獨報診斷數字。clone不帶回主FT，避免把BN初始化差異算成權重補償貢獻。

同epoch不代表相同總成本：重建是額外離線工作，必須分開報告。FT2或FT0有效但FT10差異消失，只是低FT預算方向的線索，不事後改稱主目標成功。

## 8. 預先成功與停止條件

- 主方法固定C。C的FT10須比A高至少0.3pp，且不低於相同預算下已觀察到的較強控制，才值得投入三seed確認；目前同seed參考最高為spatial top2的93.68%。
- 0.3pp是pilot投入門檻，不是統計顯著性。三seed需完整呈現配對差異，不能只報最佳seed或最佳epoch。
- 若只B較高，保留消融結果，不把C改名為B來宣稱主方法成功。
- 若只有局部MSE改善，或主方法分類未過門檻，停止本版，不反覆調lambda、patch數或回退門檻直到validation有利。
- test等方法及評估規則鎖定後才做一次正式確認；本pilot不讀test。

## 9. INT8後續檢查

先查明硬體相容Q15參考程式的位置與模型覆蓋。現有pruned_int8_simulation.py使用FX oneDNN且含test評估，不能直接拿來執行本pilot或聲稱hardware bit-exact。

後續須同一INT8/Q15規約、相同校正樣本與設定，報FP32/AMP與INT8 accuracy差、權重動態範圍、clipping、scale／multiplier／right_shift範圍及overflow。格式可保持相同，但不能先保證補償後仍能以相同bitwidth保持精度。此階段不寫RTL。

## 10. 交付與資源安排

交付：方法與來源hash、data indices、mask一致性證據、補償前後權重摘要、每層重建診斷、0/2/10曲線、配對CSV、圖表、失敗及恢復紀錄。

執行時再檢查GPU共享狀態；預計兩卡各一個工作，不以顯存尚有空間為理由盲目加並行。第一批只有3個seed0分支，時間先由校正benchmark加近期約2～3分鐘/ResNet50 FT10的量測估算，不先承諾完整補償或量化耗時。
