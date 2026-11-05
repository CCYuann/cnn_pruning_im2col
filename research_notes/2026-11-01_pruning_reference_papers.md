# 剪枝參考論文與可借鑑的技術

整理日期：2026-11-01。用途是related work、比較方法及研究設計參考。這些方法屬於原作者；引用、實作或改成cross-filter版本，都不能直接宣稱其原理是本研究首創。

本紀錄依論文原文、作者頁面或官方會議頁面整理，不代表已重現各篇實驗，也不是完整的系統性文獻回顧。搜尋未找到完全相同方法，不能作為新穎性證明。

## 保留的結構限制

- 固定連續filter clusters與Tm，維持cluster內共用K-position mask。
- 優先固定每cluster的kept_count，不因新方法降低實際剪枝率。
- 軟體可改變kept_k及保留權重數值；硬體仍接收kept_count、kept_k與packed weights。
- 一般卷積與depthwise的輸入對應不同，不能混用activation廣播假設。

## 核心參考

### R0. Cross-Filter Structured Pruning for Efficient Sparse CNN Acceleration

- 出版：IEEE Access，2025；DOI：10.1109/ACCESS.2025.3587027。
- [作者所屬機構的論文頁面](https://pure.korea.ac.kr/en/publications/cross-filter-structured-pruning-for-efficient-sparse-cnn-accelera/)。
- 本專案的結構基底，主要關注cluster共享稀疏模式與加速器資料流。
- 已實作流程的精確範圍以主程式及正式結果文件為準；不能據此宣稱全部原論文數據已重現。

### R1. A Simple and Effective Pruning Approach for Large Language Models（Wanda）

- 出版：ICLR 2024；arXiv：2306.11695。
- [論文PDF](https://arxiv.org/pdf/2306.11695)、[作者專案頁](https://eric-mingjie.github.io/wanda/home.html)。
- 可借鑑：權重大小結合對應input activation norm的importance。
- 與本專案差異：原研究對象為LLM；將其轉成CNN的共享K-position分數，需要另外定義cluster聚合與卷積patch統計。
- 建議用途：activation-aware比較基準，檢查只使用weight magnitude的不足。
- 不可直接主張：Wanda的效果已證明我們的cross-filter CNN也有效，或乘上activation就是我們的原創。

### R2. Importance Estimation for Neural Network Pruning

- 出版：CVPR 2019；arXiv：1906.10771。
- [官方論文頁](https://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html)、[PDF](https://arxiv.org/pdf/1906.10771)。
- 可借鑑：以Taylor近似與梯度估計移除filter對loss的影響。
- 與本專案差異：我們的移除單位是cluster內相同K位置，而非整個output filter。若改用group gate，需評估整組共同移除的效應。
- 建議用途：task-loss-aware比較基準，與只看權重或activation的分數比較。
- 計算代價：需要backward；校正樣本與梯度聚合方式必須固定。

### R3. Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning

- 出版：NeurIPS 2022；arXiv：2208.11580。
- [官方論文頁](https://papers.neurips.cc/paper_files/paper/2022/hash/1caf09c9f4e6b0150b06a07e77f2710c-Abstract-Conference.html)、[PDF](https://arxiv.org/pdf/2208.11580)。
- 可借鑑：layer-wise reconstruction與二階誤差補償，調整剩餘權重以補償移除造成的損失。
- 與本專案差異：須強制保留cross-filter shared mask；不能讓各filter自行使用不同support。
- 建議用途：固定mask後的reconstruction基準，隔離「選位置」與「調權重」的效果。
- 注意：矩陣計算、校正記憶體與離線成本需要另報告；不增加推論硬體運算不等於軟體成本為零。

### R4. SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot

- 出版：ICML 2023；arXiv：2301.00774。
- [官方論文PDF](https://proceedings.mlr.press/v202/frantar23a/frantar23a.pdf)。
- 可借鑑：利用layer input所形成的二階資訊，結合mask選擇與保留權重補償。
- 與本專案差異：其高效求解與支援的稀疏模式不能直接視為本專案的共享K約束；原研究主要是大型語言模型。
- 值得注意：論文討論不同output rows採用不同mask帶來的重建計算問題。相同mask在計算上有可共用的結構，但不保證模型精度較高。

### R5. To prune, or not to prune: exploring the efficacy of pruning for model compression

- 年份：2017；arXiv：1710.01878。
- [作者機構頁](https://research.google/pubs/to-prune-or-not-to-prune-exploring-the-efficacy-of-pruning-for-model-compression/)、[PDF](https://arxiv.org/pdf/1710.01878)。
- 可借鑑：隨訓練逐步提高sparsity，讓模型有恢復時間。
- 與本專案差異：需把mask更新單位改成整個共享K position。
- 比較要求：控制總訓練步數、learning-rate schedule與最終實際sparsity。
- 相關觀察：其MobileNet實驗未剪最初standard convolution與depthwise層；這是特定實驗設定，不能直接推廣成MobileNetV2的最佳策略。

### R6. NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications

- 出版：ECCV 2018；arXiv：1804.03230。
- [官方論文頁](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Tien-Ju_Yang_NetAdapt_Platform-Aware_Neural_ECCV_2018_paper.php)、[作者專案頁](https://web.mit.edu/netadapt/)。
- 可借鑑：逐步建立、評估候選網路並推進資源預算，而不是一次性相信初始敏感度。
- 與本專案差異：我們希望固定網路channel形狀及共享K結構，只更新mask或各cluster配額。
- 建議用途：動態配置搜尋的related work；引用其逐步評估思路，不把它當成已實作的cross-filter算法。
- 注意：相同weight sparsity或MAC並不等於相同量測latency，這點也與目前資料解讀有關。

### R7. ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression

- 出版：ICCV 2017；arXiv：1707.06342。
- [論文](https://arxiv.org/abs/1707.06342)、[作者專案頁](https://www.lamda.nju.edu.cn/luojh/project/ThiNet_ICCV17/ThiNet_ICCV17.html)。
- 可借鑑：利用資料與後續層的重建需求判斷移除哪些結構，避免只用當層weight magnitude。
- 與本專案差異：ThiNet移除整個filter/channel；我們保留channel數量，移除cluster共享的K位置。
- 建議用途：說明reconstruction-based pruning已有文獻脈絡，不能把重建本身宣稱為新概念。

## 新提案需特別檢查的相關工作

### R8. Layer-adaptive Sparsity for the Magnitude-based Pruning（LAMP）

- 出版：ICLR 2021；arXiv：2010.07611。
- [官方論文PDF](https://openreview.net/pdf?id=H6ATjJ0TKdf)、[論文摘要](https://arxiv.org/abs/2010.07611)。
- 核心關聯：對magnitude重新縮放並考慮剪枝distortion，產生layer-adaptive sparsity。
- 對我們的提醒：「做正規化」「用相對重要性」本身已有先例，不能單靠這些字眼主張創新。
- 若發展固定cluster配額的mask修補，需清楚說明它與global/layer sparsity allocation的差異。

### R9. Structured Pruning Via Multi-Observation Iterative Hard Thresholding

- 出版：ICASSP 2026；DOI：10.1109/ICASSP55912.2026.11460689。
- [IEEE官方頁面](https://ieeexplore.ieee.org/abstract/document/11460689)。
- 目前查到官方摘要：loss-aware ratio allocation、多觀測下共同support選擇與iterative hard thresholding。
- 對我們的提醒：共同support與多輸出需求的聯合最佳化已有相關研究，必須取得並仔細核對全文才能判斷具體重疊。
- 狀態：作為待深入閱讀的related work；不因摘要就判定與我們提案相同或不同。

## 與我們已有成果的關係

此前75個新增任務證明，改importance與保護／重新分配的效果依模型不同；它們沒有證明上述文獻的改編方法有效。

下一個自己的研究提案應明確指出：觀察到的問題、要最佳化的目標、與現有方法的差異、固定的硬體與稀疏度約束，以及可反駁該提案的實驗。

詳見 [共享mask修補研究提案](2026-11-01_shared_mask_repair_proposal.md)。目前尚未啟動這些新方法的訓練。
