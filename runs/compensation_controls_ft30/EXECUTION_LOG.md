# 執行與問題紀錄

本輪完成三seed的24條軌跡：9個A/B/C的FT10狀態重用並續訓20epochs，9個BN/scalar/Dense軌跡連續FT30，6個非共享參考FT10。沒有新test forward。

- 啟動時兩張L40S空閒；每卡只新增一個worker，未動外部程序。
- 原始cache/結果/程式只讀，新方法全部獨立資料夾與來源snapshot。
- 舊epoch10狀態續訓前重新驗證validation並還原probe消耗的RNG；所有續訓保留optimizer/scaler/RNG。
- 若發生錯誤保留log與checkpoint，不能修改lambda或epoch追高。

## 事件

- 2026-11-05T15:32:58+08:00 study_start_or_resume {"stage": "three-seed minimum evidence chain"}
- 2026-11-05T15:32:59+08:00 job_start {"job": {"mode": "prepare", "seed": 0}, "gpu": 1, "pid": 2619766}
- 2026-11-05T15:32:59+08:00 job_start {"job": {"mode": "prepare", "seed": 1}, "gpu": 0, "pid": 2619769}
- 2026-11-05T15:33:19+08:00 job_complete {"job": {"mode": "prepare", "seed": 0}, "gpu": 1}
- 2026-11-05T15:33:19+08:00 job_complete {"job": {"mode": "prepare", "seed": 1}, "gpu": 0}
- 2026-11-05T15:33:19+08:00 job_start {"job": {"mode": "prepare", "seed": 2}, "gpu": 1, "pid": 2619809}
- 2026-11-05T15:33:39+08:00 job_complete {"job": {"mode": "prepare", "seed": 2}, "gpu": 1}
- 2026-11-05T15:33:52+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "BN", "epochs": 10}, "gpu": 1, "pid": 2619861}
- 2026-11-05T15:33:52+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "scalar", "epochs": 10}, "gpu": 0, "pid": 2619864}
- 2026-11-05T15:36:12+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "BN", "epochs": 10}, "gpu": 1}
- 2026-11-05T15:36:12+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "scalar", "epochs": 10}, "gpu": 0}
- 2026-11-05T15:36:12+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "global_unstructured", "epochs": 10}, "gpu": 1, "pid": 2621236}
- 2026-11-05T15:36:12+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "quota_unstructured", "epochs": 10}, "gpu": 0, "pid": 2621239}
- 2026-11-05T15:38:32+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "global_unstructured", "epochs": 10}, "gpu": 1}
- 2026-11-05T15:38:32+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "quota_unstructured", "epochs": 10}, "gpu": 0}
- 2026-11-05T15:38:32+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "dense_extra", "epochs": 10}, "gpu": 1, "pid": 2626274}
- 2026-11-05T15:38:32+08:00 job_start {"job": {"mode": "train", "seed": 1, "variant": "BN", "epochs": 10}, "gpu": 0, "pid": 2626277}
- 2026-11-05T15:40:52+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "dense_extra", "epochs": 10}, "gpu": 1}
- 2026-11-05T15:40:52+08:00 job_complete {"job": {"mode": "train", "seed": 1, "variant": "BN", "epochs": 10}, "gpu": 0}
- 2026-11-05T15:40:52+08:00 job_start {"job": {"mode": "train", "seed": 1, "variant": "scalar", "epochs": 10}, "gpu": 1, "pid": 2627147}
- 2026-11-05T15:40:52+08:00 job_start {"job": {"mode": "train", "seed": 1, "variant": "global_unstructured", "epochs": 10}, "gpu": 0, "pid": 2627150}
- 2026-11-05T15:43:12+08:00 job_complete {"job": {"mode": "train", "seed": 1, "variant": "scalar", "epochs": 10}, "gpu": 1}
- 2026-11-05T15:43:12+08:00 job_complete {"job": {"mode": "train", "seed": 1, "variant": "global_unstructured", "epochs": 10}, "gpu": 0}
- 2026-11-05T15:43:12+08:00 job_start {"job": {"mode": "train", "seed": 1, "variant": "quota_unstructured", "epochs": 10}, "gpu": 1, "pid": 2628096}
- 2026-11-05T15:43:12+08:00 job_start {"job": {"mode": "train", "seed": 1, "variant": "dense_extra", "epochs": 10}, "gpu": 0, "pid": 2628099}
- 2026-11-05T15:45:32+08:00 job_complete {"job": {"mode": "train", "seed": 1, "variant": "quota_unstructured", "epochs": 10}, "gpu": 1}
- 2026-11-05T15:45:32+08:00 job_complete {"job": {"mode": "train", "seed": 1, "variant": "dense_extra", "epochs": 10}, "gpu": 0}
- 2026-11-05T15:45:32+08:00 job_start {"job": {"mode": "train", "seed": 2, "variant": "BN", "epochs": 10}, "gpu": 1, "pid": 2628958}
- 2026-11-05T15:45:32+08:00 job_start {"job": {"mode": "train", "seed": 2, "variant": "scalar", "epochs": 10}, "gpu": 0, "pid": 2628961}
- 2026-11-05T15:47:53+08:00 job_complete {"job": {"mode": "train", "seed": 2, "variant": "BN", "epochs": 10}, "gpu": 1}
- 2026-11-05T15:47:53+08:00 job_complete {"job": {"mode": "train", "seed": 2, "variant": "scalar", "epochs": 10}, "gpu": 0}
- 2026-11-05T15:47:53+08:00 job_start {"job": {"mode": "train", "seed": 2, "variant": "global_unstructured", "epochs": 10}, "gpu": 1, "pid": 2629825}
- 2026-11-05T15:47:53+08:00 job_start {"job": {"mode": "train", "seed": 2, "variant": "quota_unstructured", "epochs": 10}, "gpu": 0, "pid": 2629828}
- 2026-11-05T15:50:13+08:00 job_complete {"job": {"mode": "train", "seed": 2, "variant": "global_unstructured", "epochs": 10}, "gpu": 1}
- 2026-11-05T15:50:13+08:00 job_complete {"job": {"mode": "train", "seed": 2, "variant": "quota_unstructured", "epochs": 10}, "gpu": 0}
- 2026-11-05T15:50:13+08:00 job_start {"job": {"mode": "train", "seed": 2, "variant": "dense_extra", "epochs": 10}, "gpu": 1, "pid": 2630693}
- 2026-11-05T15:52:33+08:00 job_complete {"job": {"mode": "train", "seed": 2, "variant": "dense_extra", "epochs": 10}, "gpu": 1}
- 2026-11-05T15:52:33+08:00 ft10_controls_complete {}
- 2026-11-05T15:52:33+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "A", "epochs": 30}, "gpu": 1, "pid": 2631173}
- 2026-11-05T15:52:33+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "BN", "epochs": 30}, "gpu": 0, "pid": 2631176}
- 2026-11-05T15:57:13+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "A", "epochs": 30}, "gpu": 1}
- 2026-11-05T15:57:13+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "BN", "epochs": 30}, "gpu": 0}
- 2026-11-05T15:57:13+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "scalar", "epochs": 30}, "gpu": 1, "pid": 2632765}
- 2026-11-05T15:57:13+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "B", "epochs": 30}, "gpu": 0, "pid": 2632768}
- 2026-11-05T16:01:53+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "scalar", "epochs": 30}, "gpu": 1}
- 2026-11-05T16:01:53+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "B", "epochs": 30}, "gpu": 0}
- 2026-11-05T16:01:53+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "C", "epochs": 30}, "gpu": 1, "pid": 2634346}
- 2026-11-05T16:01:53+08:00 job_start {"job": {"mode": "train", "seed": 0, "variant": "dense_extra", "epochs": 30}, "gpu": 0, "pid": 2634349}
- 2026-11-05T16:06:13+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "dense_extra", "epochs": 30}, "gpu": 0}
- 2026-11-05T16:06:13+08:00 job_start {"job": {"mode": "train", "seed": 1, "variant": "A", "epochs": 30}, "gpu": 0, "pid": 2635913}
- 2026-11-05T16:06:33+08:00 job_complete {"job": {"mode": "train", "seed": 0, "variant": "C", "epochs": 30}, "gpu": 1}
- 2026-11-05T16:06:33+08:00 job_start {"job": {"mode": "train", "seed": 1, "variant": "BN", "epochs": 30}, "gpu": 1, "pid": 2636018}
- 2026-11-05T16:10:33+08:00 job_complete {"job": {"mode": "train", "seed": 1, "variant": "A", "epochs": 30}, "gpu": 0}
- 2026-11-05T16:10:33+08:00 job_start {"job": {"mode": "train", "seed": 1, "variant": "scalar", "epochs": 30}, "gpu": 0, "pid": 2637719}
- 2026-11-05T16:11:13+08:00 job_complete {"job": {"mode": "train", "seed": 1, "variant": "BN", "epochs": 30}, "gpu": 1}
- 2026-11-05T16:11:13+08:00 job_start {"job": {"mode": "train", "seed": 1, "variant": "B", "epochs": 30}, "gpu": 1, "pid": 2637957}
- 2026-11-05T16:15:33+08:00 job_complete {"job": {"mode": "train", "seed": 1, "variant": "scalar", "epochs": 30}, "gpu": 0}
- 2026-11-05T16:15:33+08:00 job_start {"job": {"mode": "train", "seed": 1, "variant": "C", "epochs": 30}, "gpu": 0, "pid": 2639543}
- 2026-11-05T16:15:53+08:00 job_complete {"job": {"mode": "train", "seed": 1, "variant": "B", "epochs": 30}, "gpu": 1}
- 2026-11-05T16:15:53+08:00 job_start {"job": {"mode": "train", "seed": 1, "variant": "dense_extra", "epochs": 30}, "gpu": 1, "pid": 2639634}
- 2026-11-05T16:20:13+08:00 job_complete {"job": {"mode": "train", "seed": 1, "variant": "dense_extra", "epochs": 30}, "gpu": 1}
- 2026-11-05T16:20:13+08:00 job_start {"job": {"mode": "train", "seed": 2, "variant": "A", "epochs": 30}, "gpu": 1, "pid": 2641203}
- 2026-11-05T16:20:33+08:00 job_complete {"job": {"mode": "train", "seed": 1, "variant": "C", "epochs": 30}, "gpu": 0}
- 2026-11-05T16:20:33+08:00 job_start {"job": {"mode": "train", "seed": 2, "variant": "BN", "epochs": 30}, "gpu": 0, "pid": 2641358}
- 2026-11-05T16:24:33+08:00 job_complete {"job": {"mode": "train", "seed": 2, "variant": "A", "epochs": 30}, "gpu": 1}
- 2026-11-05T16:24:33+08:00 job_start {"job": {"mode": "train", "seed": 2, "variant": "scalar", "epochs": 30}, "gpu": 1, "pid": 2642758}
- 2026-11-05T16:25:33+08:00 job_complete {"job": {"mode": "train", "seed": 2, "variant": "BN", "epochs": 30}, "gpu": 0}
- 2026-11-05T16:25:34+08:00 job_start {"job": {"mode": "train", "seed": 2, "variant": "B", "epochs": 30}, "gpu": 0, "pid": 2643097}
- 2026-11-05T16:28:54+08:00 job_complete {"job": {"mode": "train", "seed": 2, "variant": "scalar", "epochs": 30}, "gpu": 1}
- 2026-11-05T16:28:54+08:00 job_start {"job": {"mode": "train", "seed": 2, "variant": "C", "epochs": 30}, "gpu": 1, "pid": 2644261}
- 2026-11-05T16:30:34+08:00 job_complete {"job": {"mode": "train", "seed": 2, "variant": "B", "epochs": 30}, "gpu": 0}
- 2026-11-05T16:30:34+08:00 job_start {"job": {"mode": "train", "seed": 2, "variant": "dense_extra", "epochs": 30}, "gpu": 0, "pid": 2644909}
- 2026-11-05T16:33:14+08:00 job_complete {"job": {"mode": "train", "seed": 2, "variant": "C", "epochs": 30}, "gpu": 1}
- 2026-11-05T16:35:34+08:00 job_complete {"job": {"mode": "train", "seed": 2, "variant": "dense_extra", "epochs": 30}, "gpu": 0}
- 2026-11-05T16:35:48+08:00 report_only_numpy_audit_fix {"training_recomputed": false}
- 2026-11-05T16:36:34+08:00 report_only_numpy_audit_fix {"training_recomputed": false}
- 2026-11-05T16:36:36+08:00 study_start_or_resume {"stage": "three-seed minimum evidence chain"}
- 2026-11-05T16:36:50+08:00 ft10_controls_complete {}

## 報表問題與修正

原報表equal_nested對NumPy RNG array使用==，會產生ambiguous truth ValueError。訓練方法及所有checkpoint無影響。以獨立report-only finalizer用np.array_equal覆核，原凍結程式不變，新增來源與回歸測試記於REPORTING_FIX.json。沒有因此重算訓練。gap_decomposition.csv另明確拆解global參考與同配額非共享mask差距，均非數學上界。

## 報表環境與額外輸出稽核

第一次report-only finalizer在unit subprocess遇到MKL_THREADING_LAYER=INTEL與libgomp不相容，保留原始輸出為UNIT_TESTS_mkl_failure.txt。以MKL_THREADING_LAYER=GNU及OMP/MKL/OPENBLAS threads=4重跑報表，63項tests通過。這是報表程序環境修正，不改訓練結果，沒有新GPU訓練工作。

另以獨立analysis_audit腳本核對42個FT10/30 checkpoint，共2226個卷積層狀態。禁止位置非零及保留位置恰為零皆為0；C-vs-B直接配對及BN buffer殘值亦分開保存。腳本來源與SHA見DERIVED_AUDIT.json及source_snapshot/analysis_audit。圖表已人工檢視。
