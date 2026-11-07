# 共享Mask補償便宜控制與FT30：索引

2026-11-05完成三個獨立ResNet50 Dense seeds、Tm8、約90%原共享mask的24條軌跡。B對scalar的FT10平均配對提升+0.393pp、FT30+0.213pp；對A的FT30為+0.233pp，僅2/3seeds正值。C在FT30三seeds均低於B，主線保留較簡單B。沒有新test forward、量化或RTL變更。

本輪達下一輪5seed投入資格，但尚未跑5seeds/跨模型/80或95%/lambda或校正量消融。不是階段1全部完成，更沒有實作補償感知mask。

- [完整研究判斷與限制](../runs/compensation_controls_ft30/FINDINGS.md)
- [結果表及圖](../runs/compensation_controls_ft30/README.md)
- [凍結計畫](2026-11-05_compensation_controls_plan.md)
- [執行/問題紀錄](../runs/compensation_controls_ft30/EXECUTION_LOG.md)
- [資料ZIP](../runs/compensation_controls_ft30.zip)
