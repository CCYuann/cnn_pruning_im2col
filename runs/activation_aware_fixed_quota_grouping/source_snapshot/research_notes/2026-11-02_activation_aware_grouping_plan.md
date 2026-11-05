# Activation-aware固定配額分組pilot：凍結計畫

## 範圍

僅ResNet50/CIFAR-10 seed0、Tm8、一般卷積含1x1；逐層q完全沿用fixed_quota_filter_regrouping/prepared/seed0/contiguous.pt。實際卷積權重稀疏度89.9987171194%，保留2,345,024 weights及233,504,768 MAC/image。不改reference、cap、補償、量化或RTL。不碰test、不用validation選組或調搜尋參數。

## 分數與統計

用同一Dense checkpoint eval、FP32 forward、TF32關閉，在上一輪相同512張train BN校正影像上收集每層Conv2d輸入。按該層kernel/stride/padding/dilation unfold成[N,K,L]，逐K以FP64累計平方和，count=N*L，RMS_k=sqrt(sum(X_k^2)/count)。padding零也計入分母；每個(c,r,s)獨立統計，不用整個channel的單一近似。只支援groups=1；檢查K及L符合實際conv output。

S_ik=abs(W_ik)*RMS_k。RMS=0合法；不加epsilon，不跨層重分配q。此分數不是分類loss，也不完整建模輸入相關性。512張資料來自train，不是獨立test；與BN診斷共用資料明確揭露。

## 四格設計

- L1 contiguous：重用已完成93.20%分支。
- L1 optimized：重用已完成Jaccard+4輪交換93.38%分支。
- Activation contiguous：只換mask排序，連續groups不變。
- Activation optimized：從上一輪相同Jaccard初始groups開始，以S作共享剪枝與交換代價，仍4輪、min(4096,max(128,2*Cout)) proposals/輪、batch64、相同layer RNG，嚴格下降且每輪不重用cluster。

後兩個是唯一新增的FT10分支。optimized欄的groups依指標改變，故該欄兩列差異是整個score-aware pipeline差異，不能說只是固定groups換mask。contiguous兩列才隔離importance排序效果。

## 訓練及評估

直接重用未改動的fixed_quota_grouping_worker.train：同Dense、完整full mask按原filter IDs散佈、SGD lr=.001/momentum=.9/wd=.0005、batch128、10epochs、float16 AMP、workers2、seed+100000 training RNG、每epochvalidation與resume。新wrapper另凍結自己的方法/source fingerprint，舊train只負責完全相同的訓練路徑。512張BN副本校正不得帶入主FT。

報raw FT0、獨立BN副本、FT1/2/10、固定第10輪結果，不能挑最高epoch。本輪不自動擴seeds或改參數追高。提供score在contiguous下的差、各score下grouping差與interaction，全部是單seed探索，不做顯著性宣稱。

## 驗證與產出

單元測試unfold/獨立padding-stride計算、batch累計、zero RMS、groups拒絕與已知score排序；在真實層以獨立手工patch抽樣重算RMS並檢查Dense state未變。驗證來源checksum、校正indices與validation無交集、每層q/partition/scatter、masked weight無regrowth、全模型weight/MAC budget及source unchanged。

四格均計算同一套L1與activation-weighted proxy，避免直接比較不同單位的loss值。保留每層RMS、sum/count、groups/masks、搜尋歷程、原始結果/曲線、中文README、圖表、執行/問題紀錄與小型ZIP；大型checkpoint與BN影像保留本機，不放ZIP。

GPU先準備於GPU1，再兩卡各一個新training worker，需>=12GiB free，不干涉既有程序。時間是共享GPU下測量，不是獨占推論speedup。
