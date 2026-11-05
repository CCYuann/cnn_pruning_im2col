# Cross-filter配額粒度：原文審閱與實驗計畫

## 原文能確認及不能確認的地方

已實際讀取本機PDF第5頁（印刷頁129465）Figure4/5與第6頁129466的步驟。Figure4要求cluster內各filter同一channel使用相同spatial pattern；Figure5及後文使用三維importance G[c,r,s]、一個threshold tau。129465左欄又描述k×k可拆成k²個1×1 subclusters並在subcluster層級剪枝。

以上支持共享mask核心，但沒有唯一明定每個(r,s)是否獨立決定reference ratio、整數配額與threshold。共享圖案本身不代表必須分空間配額。此次有限標題／DOI搜尋未取得作者程式，不能據此斷言作者未釋出。

因此目前程式不能被判定為錯誤，也不能宣稱完整數據已重現。以下對照是兩種明確的演算法解讀，不把spatial版命名為paper_original。

## 固定設定與配額

ResNet34/ResNet50、CIFAR-10、Tm8連續分組、同一Dense100epoch checkpoint、相同45000/5000/10000切分，無sensitivity保護或cap。參考仍由所有eligible conv權重的global magnitude pruning .9產生。

- flat：G[c,r,s]展平成K，整體cluster配額q=K-round(reference_zero_ratio*K)，刪最小importance，與現有核心逐元素比對。
- spatial：每(r,s)的實數保留配額a[r,s]=該位置global reference非零權重數/cluster實際filter數，先floor，再依小數餘數由大到小分配至總q；同餘數用raster順序。各位置保留自己importance最大的q[r,s]個input-channel位置。
- spatial配額總和精確等於flat的q，故每cluster權重數、逐層MAC及總MAC相同。這個largest-remainder規則是本次公平實驗定義，不是原文已明定的細節。
- 記錄若各位置獨立round時的總數差，避免將整數誤差誤認成演算法提升。1×1兩版必須mask完全相同。
- l1_sum/top2_mean_abs均比較。共2模型×2criteria×2解讀=8個seed0分支，全部重新跑10epochs，不混用舊test數字當validation控制。

## 訓練與評估

沿用已凍結pilot helper：SGD lr.001、momentum.9、weight_decay.0005、batch128、AMP float16、同一seed+100000重新設定training RNG；只使用validation。被剪位置持續mask，不再生。記錄來源hash、實際sparsity、mask差異與全部logs。

## 預先決定的分支規則

若某模型任一criterion的spatial-flat validation差距絕對值>=0.3pp，該模型的兩criteria與兩解讀都補seeds1/2，避免只確認較有利的版本。若補完後任一criterion三seed平均差距仍>=0.3pp，優先交付基準敏感性結論，暫緩新mask演算法，避免在未穩定的基準上疊方法。

若未觀察到上述大幅基準差距，才值得做下一個有界ResNet50 seed0分類梯度候選pilot：先適應2epochs、在train資料上選固定配額一換一候選、再訓練剩餘8epochs，對照l1與top2，總訓練仍10epochs。此階段會另行凍結數值算法和測試後再啟動，不把梯度剪枝或配額分攤本身稱為新穎性。

無論哪個版本較高，保留所有結果；不藉由更多epochs或test反覆挑設定。0.3pp只控制實驗投入，不是顯著性門檻，也不證明較小差異不存在。
