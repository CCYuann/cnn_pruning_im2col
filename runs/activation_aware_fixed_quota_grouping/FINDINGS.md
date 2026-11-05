# Activation-aware固定配額分組：研究判斷

## 已完成

ResNet50/CIFAR-10 Dense seed0、Tm8、53個一般卷積層（含1x1），新增2個FT10分支並重用上輪2個L1控制。所有四格精確相同逐層q、2,345,024保留卷積權重及233,504,768 MAC/image，實際卷積權重稀疏度89.9987171194%。不碰test、不做INT8或RTL、不混入補償/保護。

新分數為abs(W)*每(c,r,s) patch RMS。Dense eval FP32 forward、512張固定train影像、FP64 moment累計，含padding/stride/dilation。沿用相同Jaccard初始分組及4輪有界交換，分組目標與mask皆用新分數。訓練直接呼叫未修改的舊worker.train，不改SGD/AMP、訓練RNG或10epoch預算。

## 數據

| 方法 | BN副本 FT0 (%) | FT2 (%) | FT10 (%) |
|---|---:|---:|---:|
| L1連續 | 68.02 | 92.78 | 93.20 |
| L1最佳化 | 72.88 | 92.64 | 93.38 |
| Activation連續 | 60.10 | 91.58 | 92.48 |
| Activation最佳化 | 68.30 | 91.86 | 92.90 |

- 固定連續groups，換新score：-0.72pp。
- L1 score下最佳化groups：+0.18pp。
- Activation score下最佳化groups：+0.42pp。
- 兩個最佳化pipeline相比：activation低0.48pp；這個比較同時改score與產生的groups，不是固定groups只換mask。
- 探索性interaction +0.24pp，不是顯著性證據。

共同activation proxy（以Dense weights和同一RMS計算的刪除分數）依序2781.54、2740.83、2697.02、2644.05。新指標確實優化了自己定義的目標，但沒有提高分類accuracy。不能把它稱為實測重建MSE下降或分類loss下降。

## 結論與界線

本pilot不支持改用這個activation RMS score。只報新score內分組+0.42pp會誤導，因為92.90%仍低於原L1連續93.20%，也低於原L1最佳化93.38%。依計畫完成後停止，不擴seed、不改統計或搜尋參數追高。

這是固定layer quota、特定RMS啟發式、特定初始化/有界搜尋與單Dense seed的有限負結果；不是所有activation-aware、Taylor、分群或稀疏訓練方法都無效。validation曾參與其他開發，沒有獨立test證據。RMS只衡量輸入能量，不完整描述分類敏感度或輸入相關性；這些是可能限制，不是本輪已證明的下降原因。

目前不宜再堆更多啟發式搜尋次數。若後續重啟這條線，須另立計畫，明確驗證代理與task loss的關係或新的訓練策略，不在本表追加結果改結論。補償方法原3seed正結果仍為獨立主線；quota配置不同，不能直接拿它的accuracy與本表相減當成方法提升。

## 驗證與成本

53項單元/integration tests通過。真實stem/1x1/stride2層的patch統計與獨立切片相符，Dense state在收集後未變；53層count/RMS/q、非連續mask散佈、score排序、交換目標不增、actual weight/MAC budgets與禁止位置無regrowth均通過。無OOM、數值錯誤或中斷，未改動舊結果或既有程序。

統計1.246秒、完整準備4.099秒；兩個新增FT10訓練路徑約145.37/149.74秒，單worker峰值allocated GPU約2.064GB。這些是當下共享GPU條件，不是推論speedup或完整研究總成本。

原始結果見four_cell_results.csv、effects.json；來源與資料SHA見study_configuration.json。小型ZIP含統計、mask、結果、來源及索引，不含大型checkpoint或BN影像；大型資料保留本機。
