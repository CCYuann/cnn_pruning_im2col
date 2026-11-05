# 固定Mask權重補償：工作與問題處理

## 2026-11-02：開始執行

- 使用者授權依固定mask權重補償草案執行。未修改主訓練核心或既有研究結果；工作樹內其他聊天工具與.gitignore變更保留、不介入。
- 新增獨立prior-centered ridge solver、teacher/student對齊cache與可恢復補償/FT worker，控制器只按預設規則擴seeds。
- 主方法固定sequential；dense_input是輸入假設消融，original是相同mask的控制。不換位置、不增加FT epochs、不使用test。
- 校正seed20261107，train512 fit+512 check，排除validation；每圖每層最多8個不重複空間座標，最大4096 patches。
- 39項tests通過，含augmented least-squares reference、rank-deficient輸入、空/滿mask、誤差回退、K/座標順序、跨batch student prefix與BN不變。

## 01:46：凍結並啟動GPU

- 01:46:14啟動teacher cache與數值benchmark；首次collector在fit/check皆完成，約6秒（不含前置checkpoint載入與設定）建立cache及benchmark。
- 確認最大q=871、q=0的clusters為43。首層、1×1、stride2及最大q層皆通過Dense Conv2d與patch投影對齊檢查。
- GPU1/0各單一工作，校正完成後自動跑original與dense_input；空出的GPU再接sequential。來源、計畫與校正檔案SHA已凍結。
- 每層補償後儲存resume狀態；FT每epoch儲存model/optimizer/scaler/RNG。check回退基於實際FP32儲存權重的MSE，不以FP64理想值掩蓋差異。

## 判斷規則

C在FT10比A高至少0.3pp且不低於既有較強seed0控制，才對稱補A/B/C的seeds1/2。0/2epochs僅作低FT預算線索，不事後改主目標；不因B較高就更名主方法。

本輪首先確認浮點效果。oneDNN不是硬體Q15參考；若主方法成立，另查硬體相容量化流程，不寫RTL、不擅自讀test。

## 01:51：seed0完成，依預設門檻擴大

- original：FT0 10.08%、BN clone 74.86%、FT2 93.22%、FT10 93.46%。
- dense_input：FT0 83.12%、BN clone 90.22%、FT2 93.44%、FT10 93.86%；離線補償約25.0秒，3277個非空clusters全通過check回退門檻。
- sequential主方法：FT0 91.90%、BN clone 91.06%、FT2 93.50%、FT10 93.94%；離線補償約34.8秒，3276 accepted、1 fallback、43 empty。沒有改動BN/bias/分類器。
- 主方法C比A高0.48pp且超過既有較強控制93.68%，通過原先0.3pp投入門檻；不是因B較高而改選方法。
- 01:51:44自動準備seeds1/2，兩者cache各約2.42GiB，最大q分別859/1095，benchmark均通過。01:52起對稱補A/B/C，不挑較有利分支。

## 量化參考核對

在另一個本機硬體專案找到quantization.py、analyze_resnet34_quantization.py及evaluate_bitexact_resnet34_backbone.py。前者提供Q15 scalar primitives；已檢查的BN配對及forward_backbone僅支援conv1/conv2兩卷積block，未覆蓋ResNet50 Bottleneck的conv3。

因此不能直接把這條完整模型路徑當ResNet50 bit-exact，也不以oneDNN替代。此次未修改硬體專案或RTL。先完成三seed浮點結論，將模型參考覆蓋列入後續量化必要條件。

## 02:00完成訓練；09:10續接最終稽核

- 9個分支全部於02:00完成，後續沒有繼續佔用GPU。使用者09:10要求繼續時，核對既有結果並執行最終稽核與封裝，不重跑模型。
- 三seed FT10 validation平均：A 93.633%、B 94.133%、C 94.160%。B-A逐seed+0.40/+0.94/+0.16pp；C-A逐seed+0.48/+0.84/+0.26pp。
- C-A平均+0.527pp，配對sample std0.293pp；C-B平均僅+0.027pp，逐seed+0.08/-0.10/+0.10pp。因此支持補償整體的初步效果，不支持C複雜度已明確優於B。
- FT2平均A 93.300%、B 93.747%、C 93.907%；C的FT2高於A的FT10是預先觀察點，但必須計入離線補償成本，不能直接宣稱訓練加速。
- 最終39項tests通過；3seed teacher cache checksum、取樣座標、fit/check隔離、mask逐bit一致、禁止位置為零、BN/bias/classifier離線不變均通過。seed0 A完整最終state_dict逐bit重現舊10epoch控制。
- 沒有程式錯誤、OOM或kill；未變更lambda、patch數或回退門檻挑結果。全部負例、回退及來源hash保留。模型權重與約7.26GiB teacher cache仍在本機，不放ZIP。
- 此輪不評估test或INT8，不改RTL。下一步量化須先有經驗證的ResNet50三卷積block參考；目前只核對到BasicBlock路徑，不虛報相容性。
