# 固定剪枝量的 Filter 重新分組實驗計畫

## 凍結問題與範圍

比較相同共享mask位置數下，filter分組是否降低L1代理損失並改善精度。ResNet50/CIFAR-10、Tm=8、連續flat K、l1_sum；不加保護/cap、補償、activation-aware、蒸餾或量化，不改RTL。

沿用固定Dense100epochs checkpoint及split_id=42fbb7440cc27c71。每seed先以global magnitude unstructured 90%取得每層reference，令q_l=K-round(layer_ratio*K)。所有方法在同一層每組恰保留q_l個位置。取整數造成實際sparsity不一定恰90%，必須逐層及全模型報告。這與舊global_cluster分配不同，不能直接和舊結果當成同一基準。

## 方法與搜尋上限

- contiguous：連續分組。
- random0/1/2：固定獨立分組RNG，3個初始化，分別保留結果，不選最好者。
- norm_sorted：filter絕對值總和升序，相鄰8個一組。
- jaccard：原程式確定性貪婪，用同一global reference二值mask。
- jaccard_swap：從jaccard開始，4輪交換；每輪min(4096,max(128,2*Cout))個隨機跨組交換候選，固定RNG，分批64個。以float64 abs weights計算兩組刪除L1質量，按下降量排序，每輪只接受互不共用cluster且嚴格下降的交換。無下降則提早停止；q=0/K直接跳過。只看代理代價，不看validation選組或延長搜尋。

本pilot只支援groups=1且Cout可被8整除，遇到其他層報錯，不把depthwise廣播問題藏進一般卷積。filter ID始終維持原channel意義，full mask必須依group indices散佈回去，不能repeat_interleave假定連續。

L1代理損失為sum_g sum_{k不在Top_q(sum_i abs(W_ik))} sum_i abs(W_ik)。下界為每filter獨立top_q的同一損失；缺口閉合率=(contiguous-candidate)/(contiguous-lower_bound)。分母接近0回傳null，不裁切負值。這不是分類loss或硬體cycles。

## 評估與公平性

所有7分支從同一seed Dense出發，同樣SGD lr=.001,momentum=.9,wd=.0005,batch128,float16 AMP、10epochs、workers2，重置seed+100000的訓練RNG。各epoch固定full validation，報FT0、FT1、FT2、FT10，不以best epoch選模型。另用512張固定train子集對副本做共同BN校正，絕不將校正副本帶入主FT。test不做forward或用於選擇。

先完整跑seed0七分支。jaccard_swap相較contiguous FT10提升>=0.3pp，且不低於jaccard與norm_sorted，才完整擴展seeds1/2七分支。此為投入門檻，不是顯著性檢定；失敗則如實停止並報完整seed0結果，不調參追高。隨機基準另報平均及初始化標準差；若擴3seeds，先在同Dense seed內平均random0/1/2，避免當9個獨立Dense seeds。

## 驗證與產出

驗證filter partition、每組q、每層及總weight/MAC budget一致、mask散佈、無禁止位置regrowth、非conv參數初始化未變、swap代理代價不增、來源及checkpoint SHA；每epoch可resume並保存RNG/optimizer/scaler。單元測試包括非連續filter ID、下界/zero gap、決定性、交換有效性及已知toy。

輸出原始groups/masks、逐層代價與cluster kept_count、FT曲線、配對CSV、accuracy/代理代價圖、中文README、執行及問題紀錄、來源snapshot與交接zip。現有paper_cycle_estimate不是8x8 OS cycle-accurate結果，不報真實speedup。

GPU最多一個新worker/卡且需>=12GiB free；不終止或改動既有工作。GPU忙碌造成時間拉長如實紀錄。
