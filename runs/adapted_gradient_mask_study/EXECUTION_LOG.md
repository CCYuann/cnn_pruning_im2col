# 適應後共享Gate梯度：工作與問題紀錄

## 2026-11-02 01:05：啟動

- 由配額粒度基準階段的預先門檻決定接續，並非先看新方法結果才改研究方向。
- 新候選直接來自適應2epochs後完整剪枝模型的signed共享gate導數；每cluster最多一換一。主方法按層以search/check真實CE接受，另有全部交換及同cluster同交換數的隨機控制。
- 全部分支2+8epochs，保留SGD optimizer、momentum、AMP scaler與訓練RNG。移除／新增位置momentum清零，其他不變。
- 先通過29項測試，包括顯式gate autograd對照、權重及momentum恢復、RNG恢復、配額與隨機交換數。
- 搜尋train資料使用新seed20261104，512張search、512張check、1024張audit，互斥並排除validation。audit是未參與mask選擇的train，不假稱模型從未見過。
- GPU1先執行continuous_control，GPU0執行gradient_all；完成後接gradient_gate、matched_random。

## 01:07：報表程式修訂

- 程式審閱發現baseline_rows以顯式seed加上已有seed的JSON展開，會造成重複keyword TypeError。問題只在控制結果合併，不涉及worker、訓練或mask算法。
- 已修成mapping合併，另加兩個測試（重複seed與擴大條件）。原凍結controller保留，加入report-only恢復入口：所有工作完成且worker/helper/evaluator hash未改，才允許記錄controller revision並重建報告。
- 初始controller已載入舊函式，仍可能在四個worker完成後於報表階段報錯；屆時用已凍結worker結果恢復，不重跑或改動模型數據。此預期錯誤及恢復都保留於execution.jsonl。

## 初步完成的分支

continuous_control為93.46%，gradient_all為93.52%。目前只確認scalar控制重現；完整state_dict bit-exact檢查待全部分支完成後統一驗證。這兩個數字均未超過本輪最佳控制93.68%，尚無改良成功結論。

## 01:10：全部完成與恢復

- gradient_gate 93.54%、matched_random 93.58%；主方法低於最佳控制93.68%，亦低於隨機控制，未通過預先門檻，不擴三seed或改挑消融。
- 全部新分支10epochs，沒有使用test。gradient_all實際套用3261對交換；gate與random各144對，落在完全相同clusters。search共接受8層；不同分支生成的完整plans逐元素相同。
- Gate未參與選擇的train audit CE由0.04024升至0.04288；此有限樣本觀察不直接證明過擬合，但不能宣稱search/check改善已泛化。
- 初始controller果然於報表合併拋出TypeError；四個worker已全完成。01:10使用--resume-report核對worker/helper/evaluator SHA未改、保留initial controller於source_snapshot/revisions，記錄新舊SHA並恢復報表。沒有任何模型重跑，沒有變更實驗方法。
- 完整state_dict驗證：continuous_control最終模型與第一階段flat l1逐bit相同；各分支post-adaptation模型相同。配額、資料隔離、隨機交換數均通過。
- 後續ADDITIONAL_AUDIT.json再核對post-adaptation的optimizer、AMP scaler、Python/NumPy/Torch/CUDA/loader RNG及實際套用交換數。大型模型與resume檔保留本機，ZIP只含小型mask、數據、logs及快照。

## 研究判斷

本次單一設定沒有支持「適應2epochs後，signed gate一階交換＋雙CE門檻」優於已有基準。只能針對這個有界pilot下結論，不推廣成所有分類梯度或訓練期mask方法皆無效。這輪不再增加交換比例、改adaptation時點或重複用validation挑設定。
