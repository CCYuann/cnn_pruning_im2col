# 執行與問題紀錄

完成7個FT10分支，準備1個Dense seeds。

- 啟動前GPU皆忙碌；限制每卡一新worker，沒有終止既有程序。
- 新增獨立程式與資料夾，未修改核心剪枝或RTL。
- 任意filter分組必須依原filter ID散佈full mask；以toy及真實模型獨立審核，禁止直接按cluster順序展開。
- 固定layer q會改變舊global_cluster配置，因此本輪重新訓練自己的連續控制。
- 若出現錯誤，保留原log與epoch resume狀態，不用validation修改搜尋規則。

- 程式覆核後明確保存controller amendment：擴展前先稽核reference/實際mask budgets，補zero-gap報表及integration tests。未改worker或搜尋/訓練/擴展規則；已完成分支保留，未完成分支從epoch checkpoint續跑。原始與修訂來源均保存。

## 事件

- 2026-11-02T10:05:48+08:00 study_start_or_resume {"plan_sha256": "6490540cab21d12c60d4c00dca41f721f9f471d837ed50717488e154339dddf7"}
- 2026-11-02T10:05:48+08:00 job_start {"job": {"mode": "prepare", "seed": 0}, "gpu": 1, "pid": 1051284}
- 2026-11-02T10:06:48+08:00 job_complete {"job": {"mode": "prepare", "seed": 0}, "gpu": 1}
- 2026-11-02T10:06:48+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "contiguous"}, "gpu": 1, "pid": 1051564}
- 2026-11-02T10:06:48+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "random0"}, "gpu": 0, "pid": 1051567}
- 2026-11-02T10:10:08+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "contiguous"}, "gpu": 1}
- 2026-11-02T10:10:08+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "random1"}, "gpu": 1, "pid": 1052993}
- 2026-11-02T10:10:28+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "random0"}, "gpu": 0}
- 2026-11-02T10:10:28+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "random2"}, "gpu": 0, "pid": 1053106}
- 2026-11-02T10:14:01+08:00 explicit_review_amendment {"changed": ["scripts/run_fixed_quota_regrouping.py", "tests/test_regrouping_integration.py"], "completed_branches_preserved": true}
- 2026-11-02T10:14:04+08:00 study_start_or_resume {"plan_sha256": "6490540cab21d12c60d4c00dca41f721f9f471d837ed50717488e154339dddf7"}
- 2026-11-02T10:14:07+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "random1"}, "gpu": 1, "pid": 1053879}
- 2026-11-02T10:14:07+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "random2"}, "gpu": 0, "pid": 1053882}
- 2026-11-02T10:15:27+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "random1"}, "gpu": 1}
- 2026-11-02T10:15:27+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "norm_sorted"}, "gpu": 1, "pid": 1054259}
- 2026-11-02T10:15:47+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "random2"}, "gpu": 0}
- 2026-11-02T10:15:48+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "jaccard"}, "gpu": 0, "pid": 1054348}
- 2026-11-02T10:18:08+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "norm_sorted"}, "gpu": 1}
- 2026-11-02T10:18:08+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "jaccard_swap"}, "gpu": 1, "pid": 1055238}
- 2026-11-02T10:18:28+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "jaccard"}, "gpu": 0}
- 2026-11-02T10:20:48+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "jaccard_swap"}, "gpu": 1}
- 2026-11-02T10:20:55+08:00 promotion_decision {"expanded": false, "seed0_gain_pp": 0.1799999999999926, "reason": "交換搜尋版未達預先擴展條件，完成seed0可行性檢查後停止，不調參追高。", "test_evaluated": false}
