# 固定配額共享Mask修補：自主實驗

開始日期：2026-11-01。研究目標是判斷共享mask的最壞filter誤差是否值得修補；成功或失敗都必須留下證據，不以無限制調參追求正結果。

**完成狀態：** 2026-11-01已完成7個model/version診斷與16個新增fine-tune分支。v1與mean-safe調整均未在兩模型seed0 validation勝過l1，未擴大三seed，也未使用test。[最終結果](FINAL_REPORT.md)、[問題與解法](EXECUTION_LOG.md)。下文保留本輪設定與預先定義的流程，方便重現。

## 執行順序

1. 已通過patch順序、stride/padding/dilation、一般/grouped/depthwise輸入對應、殘差交換公式、固定配額及零能量測試。
2. 五模型seed0，1024張固定train圖片：512搜尋、512獨立檢查；每層每組最多1024個patches。
3. Tm8各cluster保留數固定為正式l1_sum約90%配置，同時加入Tm16原始mask誤差診斷，Tm16不是同配額repair對照。
4. 比較l1、正規化靜態分數、平均誤差交換、最壞誤差受平均約束的交換、同次數隨機交換。
5. 最多8輪接受交換，每輪8個加入與8個移除候選；誤差計算為FP32，關閉TF32，排除bias，在BN/ReLU之前定義。
6. Gate：獨立檢查集的平均cluster最壞誤差降低至少5%，且有效filters平均相對誤差增加不超過1%。
7. 若ResNet50/DenseNet121未通過，各只允許一次預先定義的v2調整：若檢查平均誤差回歸，擴大為1024+1024張及每組2048patch並換校正seed；否則擴大為16候選、16輪。
8. 依固定優先順序選最多兩個通過模型，做5組validation-only、同10 epochs的fine-tune。沒有通過就輸出負結果，不假裝已有accuracy改良。

## 依第一輪結果啟動的有限調整

v1兩個pilot模型未勝過l1，因此另跑`v3_mean_safe`：先8輪mean修補，再8輪受更嚴格平均誤差上限約束的tail修補；mean對照改為16輪。v2是原先預留的診斷失敗調整分支，因五模型均通過診斷而未執行。

v3只有在診斷仍通過時才fine-tune。l1與normalized的mask若和v1逐元素完全相同，就沿用相同設定的已訓練控制組，記錄復用來源；random依新方法交換數重新訓練。

若validation相對l1至少提高0.1 pp，且不低於相同搜尋預算的mean對照，才對該模型鎖定版本、補seed1/2及完整5組消融，再做一次test確認。未達門檻則停止本輪，保存負結果，不繼續在test找好看的設定。這是實驗資源門檻，不是統計顯著性判定。

優先順序為ResNet50、DenseNet121、MobileNetV2、ResNet34、GoogLeNet；依診斷gate選模型，非依test accuracy。v2如重用檢查資料，該資料成為開發選擇的一部分，不能再視為完全盲測；分類結論另看validation，test未用於任何調整。

近零能量門檻為每層正輸出能量中位數乘1e-6；低於門檻的filter不參與該組最壞誤差與平均誤差目標，但仍維持原始配額，並記錄排除數量。

## 資料與重現

- `status.json`：目前階段、active PID與失敗狀態。
- `execution.jsonl`：控制器的啟動、完成、調整與錯誤。
- `EXECUTION_LOG.md`：人工判讀的問題、解決與研究決策。
- `v*/journal.jsonl`：每層進度與各worker事件。
- `v*/<model>_seed0/`：參數、train資料索引、逐cluster指標、交換紀錄、summary與mask。
- `diagnostic_summary.csv`、`diagnostic_gates.png`、`RESULTS.md`：跨模型比較。
- `pilot_validation.csv`：小規模訓練結果，不能冒充三seed test結果。
- `study_result.json`：本輪可行性與pilot的明確結論。
- `adjustment_result.json`：有限自我調整及是否補三seed的最終結果。
- `FINAL_REPORT.md`、`VALIDATION.json`、`MANIFEST.json`：最終人工可讀報告、資料／配額稽核及hash。

大型patch與mask快照以.pt保存於本機，依專案.gitignore不提交Git。此處未改動原剪枝核心與既有正式結果。

恢復：從專案根目錄執行 `python scripts/run_shared_mask_study.py`。控制器會跳過完成項，並以lock避免同時啟動兩份。若某項報錯，先檢查log及設定一致性；不能把不同實作結果混到同一版本。

自動排程在本機Python程序存活時持續運作，不消耗模型輪詢tokens。若機器重啟或程序被終止，需從相同命令恢復；不能宣稱API用量重置會自動重啟本機或對話。
