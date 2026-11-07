# ResNet50 95% 固定mask補償：單次判定實驗

凍結日期：2026-11-05，於讀取任何95%補償結果之前寫下。

問題：B能否在95%補回約0.8pp，使保留權重減半時精度接近A@90%？

設定：ResNet50/CIFAR-10、Tm8、flat l1 global_cluster共享mask、target 95%；A(original)與B(dense_input)各三個既有Dense seeds，FT10，5000張validation。lambda、校正資料、FT流程全部沿用90%凍結設定；不評估test。

判定（主要，僅此一條）：B-A於FT10配對平均 >= +0.8pp 且 3/3 seeds 為正。
- 通過：值得做CIFAR-100與FT30確認。
- 未通過：B定位為低成本附加步驟，不再投入精度實驗。

次要、僅描述：B@95% FT10對A@90% FT10（同validation）；q=0數量與fallback數。
限制：n=3；FT10門檻不代表長微調後仍成立；不調參、不挑epoch、不加seed救結果。
