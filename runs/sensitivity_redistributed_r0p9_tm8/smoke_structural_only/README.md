# 結構整合測試，禁止當成 accuracy 實驗

本資料夾使用真實 ResNet34 seed0 checkpoint，於CPU驗證重新分配後的mask數量、精確權重預算、敏感層上限與現有cross-filter剪枝流程相容。

Evaluation與fine-tune均被測試替身取代，CSV中的accuracy=0是測試用值，沒有研究意義。正式排程只讀`../per_run_csv/`，不會將本資料夾納入統計。
