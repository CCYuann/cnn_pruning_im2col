# 適應訓練後的共享Gate梯度候選：有界pilot

## 啟動依據

spatial_quota_baseline_study於2026-11-02完成8組seed0控制，spatial-flat差異為ResNet34 l1 +0.14pp、top2 -0.04pp；ResNet50 l1 -0.06pp、top2 +0.26pp，均未達預先0.3pp門檻。這不是等效證明，也未確認作者原始實作；本pilot明確沿用flat K基準。

## 要驗證的變更

先適應2epochs，再由目前剪枝模型的有標籤CE梯度直接產生共享K候選，不再從Dense輸入的局部重建候選庫挑選。其後訓練剩餘8epochs；同一SGD optimizer、momentum及AMP scaler持續使用，總共10epochs，不重置訓練流程。搜尋前後保存／恢復訓練RNG及loader generator。

## 數值定義

有效位置用adapted權重，被剪位置用原Dense權重作待恢復值。把兩者合成W_tilde；共享gate g_k在kept位置為1，dropped位置為0。一次在search資料上對完整模型平均CE求梯度：

```text
D_k = sum_over_filters(W_tilde[m,k] * dL/dW_eff[m,k])
remove a: 選kept中D最大者
add b: 選dropped中D最小者
predicted_delta_CE = -D_a + D_b
```

每cluster最多交換一對；只有預測delta<-1e-8才提議。保留數逐cluster不變，0/full沒有交換。這是signed一階gate導數，不是把abs梯度稱為同一公式。算梯度時，模型parameter的dropped數值是0但仍可取得dL/dW；不可先把該梯度mask成0。隱式公式已有顯式gate autograd對照測試。

## 分支

1. continuous_control：2+8epochs無mask交換，應與原flat l1控制重現一致。
2. gradient_all：一次套用所有預測改善的交換，檢驗梯度候選本身。
3. gradient_gate（主方法）：按forward層序在目前網路試套該層候選，search與check CE均下降至少1e-5才接受。初始梯度只算一次，會有stale-gradient限制；實際CE門檻是保護，不假稱精確二階最佳化。
4. matched_random：相同gate接受的clusters、每cluster相同交換數，但隨機挑keep/drop位置。用獨立generator，不使用loss挑隨機mask。

梯度／gate／random三分支各自從完全相同2epoch狀態產生相同候選及gate決策。選擇完成後恢復post-adaptation模型，再套自己的mask。移除位置的momentum清零、新恢復位置使用Dense值，其他權重及momentum不改。

## 資料及公平性

ResNet50 seed0、相同Dense checkpoint、CIFAR-10、Tm8、actual約90%、無保護cap。全分支10epochs、lr.001與SGD規則同原控制。比較本輪所有flat/spatial l1/top2控制，不只挑較弱l1。

search512/check512為train內新calibration_seed20261104；另train audit1024未參與mask決策。三者互斥並排除validation5000張。Dense及適應訓練本來已使用train，因此audit只代表未參與mask選擇的訓練樣本，不是完全未見泛化測試。validation用於pilot判斷，test不評估。

## 預先鎖定判斷

主方法只有gradient_gate，不因其他分支碰巧較高就更名。須比本輪ResNet50四個控制中最高者（seed0為spatial top2 93.68%）高至少0.1pp，且不低於gradient_all與matched_random，才值得跨seed確認。若未達，保留負結果，不繼續試交換比例、epoch位置或更多loss門檻。

本輪先完成seed0 pilot，不以短期validation差異宣稱顯著、原創或普遍有效。Taylor重要性與梯度再生已有文獻，這是共享K約束下的具體改編驗證，不把概念本身視為新穎性。

參考：[Molchanov et al., CVPR2019](https://arxiv.org/abs/1906.10771)、[GraNet, NeurIPS2021](https://proceedings.neurips.cc/paper/2021/file/5227b6aaf294f5f027273aebf16015f2-Paper.pdf)。
