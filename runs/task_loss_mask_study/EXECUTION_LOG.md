# 工作與問題處理紀錄

## 2026-11-02：目標與實作

- 問題：上一輪降低局部重建誤差，但ResNet50及DenseNet121的seed0分類pilot未改善。不能再把局部誤差當作分類效果。
- 本輪改動：保持原始每cluster配額，候選仍用前輪固定的mean/worst repair；改由目前完整剪枝網路的有標籤分類CE決定是否換某層mask。
- 資料：train內512張search、512張check、1024張額外audit互斥，排除固定validation5000張。dual使用check參與決策，因此不稱check獨立泛化測試。
- 公平性：相同checkpoint、配額及10epoch fine-tune；只移交mask，搜尋使用的BN狀態不移交訓練。
- 工程檢查：14項unit tests通過，包含逐cluster配額、被刪權重恢復、BN rollback、非有限loss及接受條件；兩個新程式通過py_compile。
- 啟動：00:06於GPU1執行frozen/dual，GPU0執行frozen/search_only，各一個工作。啟動時兩卡均空閒。
- 自動流程：先fine-tune兩分支；若主方法dual未達比l1高0.1pp且不低於無條件worst的門檻，再做一次per-candidate BN重新校正的兩分支。只在通過時擴大seeds1/2，不以test挑設定。
- 恢復：各層完成後原子儲存search_state.pt；控制器跳過已完成工作。來源hash已凍結。若遇到錯誤，保留status與完整logs，不隱藏失敗。

## 解讀限制

尚未有分類改善結論。不能保證單seed的0.1pp差異顯著；此門檻僅控制是否值得投入三seed確認。若兩版失敗，交付可重現負結果，並由證據決定下一輪方向，不無限搜尋直到出現有利數字。

逐層結果見各分支journal.jsonl；排程、分支調整與錯誤見execution.jsonl；最終結果由FINAL_REPORT.md與CSV整理。

## 00:09：第一版結果與調整

- 搜尋約15秒完成；dual接受26/53層，search_only接受32/53層。dual在未參與選擇的1024張audit CE由2.61899降至2.38222，約下降9.04%。這說明分類proxy有改善，但不是fine-tune後改善證據。
- 相同10epochs後，dual validation為93.24%，原l1為93.46%，差-0.22pp，未通過預設擴大門檻。因此未拿search_only改當主方法。
- 自動啟動預先規劃的BN校正版本。每候選只使用search資料校正，拒絕時完整復原buffers；最後只轉交mask給fine-tune，以排除帶入不同BN初始化的混淆。
- 另加兩個整合測試：分類改善時接受且可恢復；check不接受時原mask及真實權重都復原。全部16項測試通過。程式及資料目前無執行錯誤。

## 00:12：第二版完成與研究判斷

- BN proxy dual接受26層，audit CE 0.77365→0.57543（下降25.62%）；相同10epochs後validation為93.20%，相對l1為-0.26pp。
- 兩個search-only消融分別為93.48%（frozen）、93.54%（BN校正），相對l1僅+0.02／+0.08pp。未達0.1pp擴大門檻，且不是預先指定的主方法；不因此另挑一個當成功結果。
- 實際剪枝率全部89.9994336%，逐cluster kept_count一致。四組資料隔離、逐層單調接受條件與原l1 mask equality全部通過。
- 新增四次完整10epoch訓練，重用兩個來源及mask一致的控制；沒有改epoch、學習率或資料切分來補救新方法，沒有使用test或擴大seeds。
- 未發生程式例外、OOM或程序被kill。兩GPU每張單工作約2.9GB，訓練完成後釋放。
- 判斷：在現有固定局部候選庫與逐層貪婪策略下，全模型CE門檻不足以帶來可靠的fine-tune後改善。不能把原因確定歸咎於BN，因為校正proxy後仍未改善主方法；也不能僅稱校正過擬合，因audit CE同樣改善。

## 下一輪需要先分離的問題，不是已成立的新方法

1. 候選庫仍由Dense輸入的局部重建產生，可能缺少能改善訓練後accuracy的mask；本輪只測「從既有候選挑選」，未測直接由分類梯度產生固定配額交換。
2. 剪枝後、fine-tune前的CE是否能預測可訓練性尚未證實。可以先在train內做短暫且相同預算的適應，再產生候選；但必須讓l1控制也得到相同訓練步數，不增加新方法總epochs。
3. 下一輪應重新預先鎖定假說、資料與停止條件，不繼續用同一validation無限制挑版本。跨seed及其他模型仍是必要限制；現有負結果保持可追溯。
