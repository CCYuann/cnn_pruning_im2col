# 原文與本實作：已確認及待確認事項

## 原文來源

Pham et al., Cross-Filter Structured Pruning for Efficient Sparse CNN Acceleration, IEEE Access 2025, DOI10.1109/ACCESS.2025.3587027。

實際核對本機PDF第5、6頁，印刷頁129465、129466；不是依摘要推論。Figure4顯示cluster內每個input channel的spatial mask跨filters一致。Figure5與後續公式對G[c,r,s]使用共同threshold tau；但129465左下又描述k²個1×1 subclusters，未進一步唯一指定它們的獨立配額或threshold。

## 可以確認

- importance為相同座標跨filters的絕對值加總；公式影像確認含絕對值。
- 先global unstructured得到稀疏比例參考，再產生cross-filter共享mask，最後retrain。
- 3×3不是直接刪掉整個output filter；每個(c,r,s)可共同刪除。
- 1×1可視為channel位置的共同刪除。

## 不能由原文唯一確定

- spatial subcluster是否各用自己的global-reference sparsity與threshold，還是僅描述共享mask的座標分解。
- 每filter-cluster或spatial位置如何取整、同分如何處理。
- 完整可重跑的Dense初始化、CIFAR小圖stem、資料切分、每個模型訓練與FT全部設定。
- 本次有限title/DOI查詢未取得作者程式；這不是「作者沒有程式」的證據。

## 本輪兩個操作性定義

flat與spatial都符合共享mask圖樣；差別在是否固定每(r,s)配額。spatial用global-reference權重存活數/實際cluster寬度作ideal，largest remainder精確匹配flat的總kept_count。這個取整定義由我們為公平比較加入，不是原文已聲明的算法。

不論結果哪個較高，都只能說「配額解讀在本設定的影響」，不能因此證明作者實際用了哪一個。也不能把原文accuracy較低的原因直接歸給這項差異。

## 本專案已確認設定

- CIFAR-10輸入32×32；ResNet34/50入口改3×3 stride1、最初maxpool移除。
- Dense100epochs、SGD初始lr0.1、cosine到0.0001；checkpoint training_config逐分支保留。
- train45000、validation5000、test10000，split ID42fbb7440cc27c71。
- 本輪Tm8、連續filters、conv權重reference .9、無保護/cap；所有分支FT10epochs。
- 只使用validation，不把先前三seed test表當成這輪重新執行的控制。

## 本輪判讀邊界

不同K位置即使權重數與MAC完全相同，也可能有不同SRAM bank／資料重用特性。本輪是軟體精度與配額審核，沒有量測硬體能耗或latency。單seed的小差異不代表統計顯著，較小差異也不能當成兩算法等效證明。
