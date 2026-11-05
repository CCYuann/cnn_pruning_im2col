已額外加入簡要 Python/PyTorch 專案架構，並限定軟體演算法、共享 mask 結構化剪枝與公平驗證；不帶入舊實驗結果、不要求硬體設計。此簡介與論文會在網頁與終端機模式共用。

目前預設只載入使用者指定的 Cross-Filter Structured Pruning for Efficient Sparse CNN Acceleration 論文全文文字（DOI: 10.1109/ACCESS.2025.3587027），不載入舊專案紀錄或結果。文字位於 Git 忽略的 .claude_research/references/cross_filter_paper.txt，由工作區原始 PDF 使用 pdftotext 擷取；圖像與公式排版可能不完整。論文會隨每輪請求傳送並計入輸入用量。以下先前「空白背景」描述已由此設定取代。

# 與 Claude 討論研究方向

## 網頁聊天介面（建議）

```bash
bash scripts/start_claude_web.sh
```

在執行這個指令的電腦上，以瀏覽器開啟 http://127.0.0.1:8765 。
在網頁輸入海大 MYAI168 金鑰即可開始；支援串流回覆、多輪追問、複製與匯出。
金鑰只在頁面記憶體與單次後端請求使用，不存檔；對話存於瀏覽器 localStorage。
網頁模式與終端機模式的歷史各自保存。預設不讀取或傳送任何專案背景檔案。
本機服務僅監聽 127.0.0.1，Ctrl+C 可關閉。若程式跑在遠端 SSH 主機，
可在自己的電腦用 ssh -L 8765:127.0.0.1:8765 使用者@主機 轉送後再開瀏覽器。
可加 --port 8766 換埠號。

此工具預設為探索新想法模式，不自動讀取 README、研究紀錄或實驗報告。
只傳送你輸入的內容和當前主題對話。終端機可用 --file 主動指定參考檔。
不會自行讀取所有程式、CSV、模型權重或執行訓練，也沒有網路文獻搜尋功能。

## 啟動

已預設模型 claude-fable-5-1。Linux/macOS 可直接執行：

```bash
bash scripts/start_claude_research.sh
```

啟動器會在 Git 忽略的 .claude_research/venv 安裝獨立 SDK 環境，
並以隱藏輸入要求 API Key；金鑰不存檔。啟動器預設連到海大 MYAI168：
https://www.myai168.com/ntou/api/anthropic （SDK 會自行加 /v1）。
可用 ANTHROPIC_BASE_URL 覆寫；直接執行 Python 腳本則沿用 SDK 預設或環境變數。

先驗證金鑰與模型權限（只查詢模型，不傳送研究背景）：

```bash
bash scripts/start_claude_research.sh --check
```

平台文件：https://www.myai168.com/ntou/developer/anthropic_claude_api

在專案根目錄執行：

```bash
python -m pip install anthropic
# 安全輸入金鑰，避免寫進 shell history
read -rsp 'Anthropic API key: ' ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY
echo
# 換成你在 Claude Console 可用的模型 ID
export ANTHROPIC_MODEL='你的模型ID'
python scripts/claude_research_chat.py --preview
python scripts/claude_research_chat.py
```

API Key 請在 Claude Console 建立。API 會依用量計費；每輪會重送研究背景與
完整歷史，對話越長費用越高，可用 /reset 重新討論。啟動後會列出傳送檔案。
金鑰只從環境變數取得，不存入對話檔。

可以這樣開始：

> 請根據現有成果與負結果，先整理目前證據支持什麼、不支持什麼。
> 我希望找出能形成論文貢獻的下一步。請比較三個方向，列出最小實驗、
> 公平對照、成功與停止標準，並先問我可用的 GPU 資源和完成期限。

## 加入特定結果或程式碼

```bash
python scripts/claude_research_chat.py \
  --file runs/shared_mask_repair_study/all_pilot_validation.csv
```

--file 可重複指定，路徑相對於專案根目錄。文字檔每份上限 40,000 字元，
背景合計上限 200,000 字元；超過時請先整理摘要。這是字元防護，不保證符合
每個模型的 token 上限。

對話自動儲存在 Git 忽略的 .claude_research/ideas_session.json，下次啟動會接續。
使用 --session .claude_research/另一主題.json 可另開主題。
輸入 /quit 離開，/reset 清除目前主題的本機歷史。
舊版網頁對話不會自動帶入探索模式；新對話仍會保留在瀏覽器供後續追問。

官方 SDK 文件：https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python
