# 固定配額、全模型分類 Loss 篩選 Mask

## 目標與邊界

上一輪局部重建誤差改善但未改善分類。本輪不更動 cluster 成員、Tm8、逐 cluster kept_count、權重或 fine-tune 預算，只改共享 K 位置。先做 ResNet50 seed0、實際約90%剪枝。

方法尚未驗證有效，也未確認學術新穎性。不能把整網路 loss 搜尋或 BatchNorm 校正本身宣稱為首創。

## 第一版

以 l1 mask 建立完整剪枝模型，依 forward 層順序，逐層比較前輪生成的 mean_repair / worst_repair mask。其他層維持目前已接受的 mask，使用 Dense 原權重恢復需要重新保留的位置，不能在已歸零的權重上直接乘新 mask。

每層先挑 search CE 最低者；至少改善1e-5才考慮接受。dual版本還要求 check CE 至少改善1e-5，否則原狀復原。search_only為消融，不用check做決策。每層只處理一次，不反覆調到有利結果。

## 資料隔離

- search：train 512張，與上一輪局部候選產生資料一致。
- check：train另外512張；dual會用它接受／否決，故它是開發資料，不是獨立泛化證據。
- audit：train額外1024張，與前兩者互斥；只在搜尋開始與結束評估，不影響逐層決策。
- validation：原固定5000張，用於10epoch fine-tune後的pilot與是否擴大seed之決策。
- test：本輪pilot不評估。通過三seed確認後，鎖定方法才做最終確認。

## 有界調整

第一版 frozen BN 的 dual / search_only 均訓練。若 dual 比l1提升不到0.1pp，才做第二版：每個候選用同一search資料重新校正BN，再量測分類CE；拒絕時恢復原BN buffers。這是在修正嚴重剪枝後 Dense BN 不匹配的 proxy，並非先認定它有效。

搜尋的 BN 狀態不帶入 fine-tune；所有分支從同一 Dense checkpoint 的 BN 開始，只移交 masks，確保比較的是選出的mask。第二版同樣跑dual/search_only，不無限增加超參數。

## 公平對照與擴大規則

重用上一輪逐元素相同l1與無條件worst_repair控制，記錄hash、來源、10epoch訓練設定。主方法只認dual，不因search_only碰巧較高就換名挑贏家。dual須比l1高至少0.1pp且不低於無條件worst_repair，才以同版本進入seeds1/2；優先第一版，否則第二版。保留所有負結果。

若兩版均未達條件，交付負結果與下一步判斷，不把無限試參數等同持續優化。若通過單seed，三seed結果仍必須完整呈現，不能只報平均有利者。共享mask配額相同只保證理論權重/MAC數相同，不保證實測硬體能耗相同。

## 紀錄與恢復

輸出位於 `runs/task_loss_mask_study/`。逐層journal、可恢復search_state、來源hash、互斥資料indices、搜尋前後CE、fine-tune logs、驗證圖與報告均保留。不更改舊研究快照或原始核心訓練程式。
