# 執行與問題處理紀錄

## 2026-11-01：實作及啟動

- 使用者授權自主實作、調整與執行，要求完整保存問題及解法。
- 建立持續工作目標；完成的定義是有證據的可行性／pilot結論，不限定一定提高accuracy。
- 檢查既有checkpoint、CIFAR split、GPU與磁碟；兩張GPU可用，保留所有既有實驗與未提交修改。
- 新增固定K配額的交換器，不修改原剪枝核心。以原l1_sum cluster配額作基準。
- 先處理patch與權重K順序、stride/padding/dilation及grouped/depthwise輸入差異，避免將不同channel誤當共享activation。
- 以直接投影重新計算檢查快速殘差交換公式。5個測試已通過，包含已知衝突的合成例子。
- ResNet50的8+8張、16patch小型整合smoke測試成功；其結果只驗證流程，樣本太少，不當作正式研究結果。
- 正式v1使用512+512張及最多1024patch；增加Tm16純診斷對照，修補與訓練仍限定Tm8。
- 輸出原始誤差、最壞誤差、近全損filters數、排除filters數、實際交換及獨立檢查指標，避免只呈現有利摘要。
- 事先限制v2調整方式與訓練gate，保留負結果；不以test選設定。

詳細時間及每次調整由execution.jsonl與各版本journal.jsonl自動續記。後續人工分析寫在本檔下方。

## 首個正式診斷：DenseNet121 seed0

- v1使用512張搜尋與512張獨立檢查資料，非smoke結果。
- 最壞誤差修補共接受8752次cluster內一換一，全部保持原cluster的kept_count。
- 獨立檢查平均cluster最壞誤差由0.645805降至0.411693，改善36.251%；有效filters平均相對誤差由0.421130降至0.349626，改善16.979%。
- 只最小化平均誤差的修補也有效：平均誤差下降22.887%，最壞誤差下降30.713%。因此尚不能把可行性當成minimax目標獨有優勢，必須保留此消融。
- 同次數隨機交換讓獨立檢查平均誤差增加20.758%，支持交換位置的選擇有影響，但不是分類精度證據。
- 通過預先設定的5% / 1%診斷gate；後續排程將依既定規則做validation-only fine-tune，不使用test調參。

## 五模型v1完成，進入分類驗證

| Model | 獨立檢查最壞誤差改善 | 平均相對誤差變化 |
|---|---:|---:|
| ResNet34 | 12.839% | −4.711% |
| ResNet50 | 23.494% | −9.792% |
| GoogLeNet | 21.64%（完整數值見CSV） | −8.13%（完整數值見CSV） |
| DenseNet121 | 36.251% | −16.979% |
| MobileNetV2 | 10.81%（完整數值見CSV） | −3.80%（完整數值見CSV） |

- 五模型均通過預先定義gate。ResNet50與DenseNet121已通過，所以沒有必要為了調參而啟動v2。
- 開始兩模型各5組validation-only fine-tune，固定seed0、10 epochs、相同SGD與資料順序。
- 第一個負向觀察：ResNet50正規化靜態分數雖改善重建誤差，其10輪validation accuracy為92.28%，低於同次重跑l1的93.46%。這直接提醒重建surrogate不等於分類accuracy；必須等待mean/worst修補消融，不以診斷通過宣稱成功。

## 問題與自我調整：平均誤差保護可能過鬆

- ResNet50 seed0的mean_repair及worst_repair均為93.40% validation，l1為93.46%，沒有分類優勢。
- 機制分析：worst版本的搜尋平均誤差上限是原l1，允許在改善尾端時放棄部分原本可達的平均誤差改善。其獨立檢查平均誤差0.484172高於mean版本0.463677。
- 決定測試v3_mean_safe：先做8輪平均誤差修補，再從其mask出發做8輪最壞誤差修補，使用更嚴格的「第一階段後平均誤差」上限。
- 為控制搜尋量，v3的mean對照使用最多16輪平均誤差修補；不是拿16輪新方法和8輪mean相比。
- 此調整是依validation及校正誤差作出的開發決策，不使用test。保留所有v1負結果。最多對原兩個pilot模型中未證明優勢者執行一次此調整，再決定是否值得三seed確認。
- v3的worst_repair標籤表示two-pass mean-safe版本，完整定義與source hash另存method_definition.json，不能混同v1的單階段minimax。

## v1十個訓練分支全部完成

| Model | l1 | normalized | mean_repair | worst_repair | random |
|---|---:|---:|---:|---:|---:|
| ResNet50 seed0 | 93.46% | 92.28% | 93.40% | 93.40% | 93.28% |
| DenseNet121 seed0 | 93.84% | 93.90% | 93.64% | 93.74% | 94.04% |

- 以上均為validation accuracy，沒有使用test。不是三seed統計。
- v1的尾端修補未勝過l1，不宣稱新方法提升分類精度。
- DenseNet121的random控制組反而高於l1，說明單seed小幅差距可能來自mask擾動及fine-tune變異；不能把重建改善與accuracy差異直接建立因果。
- 兩個模型都符合啟動一次mean-safe調整的條件。v3診斷先行，完成後才決定訓練及是否補三seed，完整保留v1資料。

## v3調整的早期結果與仍未解決的問題

- v3兩模型診斷均通過；DenseNet121的獨立檢查平均誤差下降24.401%、最壞誤差下降36.664%，比v1保留了更多平均誤差改善。
- ResNet50 v3的16輪mean對照為93.06% validation，mean-safe為93.02%，仍低於原l1的93.46%。更多搜尋並未改善分類。
- 已確認的問題是「這些局部重建surrogates的改善未穩定轉化為分類改善」，不是程式配額錯誤；patch、公式、共享mask、配額與資料隔離均另有檢查。
- 不能只歸咎於搜尋資料overfit：獨立檢查重建誤差同樣改善。尚待驗證的解釋包括：Dense層輸入與完整pruned network實際輸入的差異、BN後的任務效用，以及平均／最壞相對誤差對所有filters的權重不符合分類loss。
- 這些是後續機制假說，不宣稱已找到確切原因。保持相同10輪訓練，不追加只有新方法才有的epochs或learning-rate優惠。

## 2026-11-01 23:51：本輪完成與停止決策

| Model | 原始l1 | v1尾端修補 | v3 mean-safe | v3 mean16對照 | v3隨機對照 |
|---|---:|---:|---:|---:|---:|
| ResNet50 seed0 | 93.46% | 93.40% | 93.02% | 93.06% | 93.56% |
| DenseNet121 seed0 | 93.84% | 93.74% | 93.52% | 93.88% | 94.08% |

- 全部為validation accuracy。test未用於調整，且本輪未觸發test確認。
- 新增訓練共16個分支：v1十個、v3六個；另有四個逐元素mask相同的l1/normalized控制結果復用。
- 共7個診斷model/version：五模型v1，加ResNet50及DenseNet121的v3。v2分支因初始診斷已通過而未啟動。
- 候選均未達「對l1至少提高0.1 pp且不低於mean對照」門檻，不擴大三seed，也不啟動原先可能的60個全面訓練分支。
- 此決策保存負結果，不以繼續改seed、加epochs或查看test來尋找成功。這不是斷言方法普遍無效；目前只是未得到支持的seed0 pilot。
- 目前可確認的成果：固定配額的共享mask修補在局部重建指標上有效，所有cluster配額與資料隔離均通過檢查；但重建指標不是充分的分類精度surrogate。
- 精度問題尚未解決。可能的下一輪研究應先檢查pruned-prefix輸入偏移、BN/任務效用與梯度導向的交換接受條件，而不是無根據擴大相同重建目標搜尋；這些假說尚未驗證。
- 最終數據、圖表、source snapshots、mask與稽核由summarize_shared_mask_study.py產生，所有大型patch與完整訓練權重保留本機，未刪除。
