# 共享mask補償：階段0/1最小證據鏈（凍結計畫）

## 本輪邊界

主線為共享mask下的權重補償；補償感知mask為後續延伸。本輪不做新分組搜尋、activation加權、量化、RTL或OBS mask實作。先完成ResNet50/CIFAR-10、Tm8、約90%卷積權重剪枝、既有三個獨立Dense seeds的控制及長微調；5seeds、跨模型、80/95%與lambda/校正量掃描只依本輪證據決策，另立下一輪計畫，不在本輪無界全展開。

## 評估規則

- 50000個原train samples以split_seed0排列，前5000為validation，其餘45000為Dense訓練。原10000個test歷史上曾用於早期實驗，不能稱未接觸test；本輪禁止任何新test forward或以test選擇。
- Dense seeds0/1/2是三個獨立100epoch訓練，非同一Dense多個FT seed。每個Dense seed內所有方法共用FT RNG=seed+100000、相同資料/順序/增強與optimizer/scaler設定。
- 校正fit512與check512為互不重疊train子集，calibration_seed20261107；每層最多4096個對齊patches。不是validation或test。check只作既有重建安全fallback，不作最終獨立證據。
- 不調lambda或選最好epoch。B/C沿用既有prior-centered ridge，lambda=max(1e-8,.001*mean(diag(H_SS)))，僅既有數值失敗jitter規則。全部報固定FT0/1/3/10/30。
- 剪所有53個groups=1的Conv2d，包含首層、1x1及downsample；不剪FC、BN affine或bias。Dense CIFAR stem為3x3 stride1、無初始maxpool。各seed沿用原flat l1/global_cluster共享mask；禁止保護/cap及跨cluster重新分配。
- 稀疏度分母是所有納入的卷積權重，不是全模型parameters。分開報structural mask density及數值nonzeros。逐層/cluster q、實際weight/MAC預算完整保存。

## 分支

| ID | 定義 | 目標epochs |
|---|---|---:|
| A | 原保留權重，原BN | 30 |
| BN | A的相同權重，只用fit512重校BN running statistics，再正常FT | 30 |
| scalar | 每filter一個least-squares純量縮放，僅作用於kept weights | 30 |
| B | 既有Dense-input ridge補償 | 30 |
| C | 既有student-prefix逐層ridge | 30 |
| global_unstructured | 全域abs magnitude，精確匹配該seed A總kept weights | 10 |
| quota_unstructured | 每filter可選不同位置，但保留數等於其原cluster q | 10 |
| dense_extra | 同Dense、完全不剪，追加相同SGD訓練 | 30 |

共24條軌跡。A/B/C既有FT0..10原始紀錄及epoch10完整state重用，僅續跑11..30，保留optimizer、AMP scaler、所有RNG與loader generator，不重啟。BN/scalar/dense_extra從0連續跑30，兩個unstructured從0跑10。不把10與30各自重訓再混成曲線。

scalar：對每層、每cluster的原Dense-input aligned fit patches，z_i=W_i,S X_S，alpha_i=(z_i dot y_i)/(z_i dot z_i)。den<=1e-12取alpha=1；不調clip、允許負alpha。q=0保持全零，q=K保持原權重。alpha轉成實際FP32權重後，以同check patches計算整cluster MSE；若高於原kept weights則整cluster fallback，與B的保守gate一致。記錄所有alpha、accepted/fallback、fit/check MSE及時間。不增加bias、不恢復連接、不校正主模型BN。

BN只改running_mean/var/num_batches_tracked，不改權重/affine。所有方法另報相同fit512的獨立BN校正副本FT0，不帶回主FT。正常training BN可能使BN分支與A的weights及functional buffers收斂相同，須檢查而非假設；差異若消失如實報告。

global_unstructured只匹配總weights，不保證逐層MAC相同；不能稱數學精度上界。quota_unstructured精確匹配每filter/cluster q及逐層MAC，是分離共享位置約束代價的較直接控制；也不是分類精度嚴格上界。

## 訓練及驗證

SGD lr=.001,momentum=.9,wd=.0005,batch128,AMP float16,workers2，constant LR與既有FT10完全一致，deterministic CUDA、TF32 off。每epoch validation。只用最後第10/30輪作該budget結果，不挑最好epoch。每epoch可resume，額外保留epoch10 state供BN/A與舊控制核對。

訓練前核對原來源/模型/cache SHA、資料隔離、三seed mask、首層/downsample/FC分類、scalar解與實際FP32 fallback、global總額與quota逐filter額。訓練後檢查禁止位置為0、A/B/C epoch10重用狀態、optimizer/RNG連續、BN/A差異及每個sample split。一切測試/稽核失敗先停止並記錄，不改方法追val。

## 報告及下一階段決策

報各Dense seed與mean/sample std，以及B/C對A、BN、scalar的配對差值（不以不同Dense seed當非配對樣本）。FT0主結果與共同BN副本分開。報與Dense_extra相同budget的accuracy gap、兩種unstructured gap，以及準備/求解/FT時間和峰值GPU記憶體；不將epoch直接當總時間speedup。

下一輪5seed確認資格預先定為：B在FT10或FT30的平均paired gain同時對BN及scalar >=0.2pp，且該budget至少2/3 Dense seeds各自對兩者皆為正。這只控制投入，不是顯著性檢定。未達時檢查FT0/1/3是否存在少FT價值並報告，不在本輪增加參數掃描。達標時提出下一輪5seed/跨模型計畫，沒有確認前不宣稱已完成。C不必高於B；若差異很小，B保留主線。

完整輸出中文README、曲線、paired CSV、層/cluster配額、來源與資料manifest、問題紀錄與小型ZIP。大模型、teacher cache及影像保留本機不重複打包。每卡一個新worker，空閒顯存>=12GiB、可用RAM>=16GiB、磁碟>=15GiB；不停止外部程序。
