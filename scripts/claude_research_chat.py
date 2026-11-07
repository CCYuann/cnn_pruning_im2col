#!/usr/bin/env python3
"""Discuss this project's research direction with Claude (optional SDK)."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / ".claude_research/references/cross_filter_paper.txt"
ARCHITECTURE = """使用者目前專案的簡要架構（只描述實作，不包含舊實驗結果）：
- Python / PyTorch CNN 剪枝研究程式。主程式 resnet34_pruning_experiments.py
  負責模型建立、資料載入、訓練、剪枝、fine-tuning 與指標輸出。
- 支援 ResNet34/50、GoogLeNet、DenseNet121、MobileNetV2；資料集包含
  CIFAR-10 與 ImageFolder/Animals-10。
- 卷積權重為 [輸出通道, 輸入通道, kernel_h, kernel_w]。Cross-filter 以
  tm 個輸出 filters 為一組，組內共享輸入通道與 kernel 位置上的 mask；
  重要性可由組內權重絕對值總和（l1_sum）等準則取得。
  剪枝後 fine-tuning 會持續套用 mask，維持被剪位置為零。
- 有 dense baseline 與 global unstructured reference，可比較準確率、實際
  sparsity、理論 MAC reduction；scripts/ 提供批次實驗、彙整與繪圖。
- 另有 INT8/Q15 軟體模擬、成本估算與匯出工具，但本次不要求改動它們。

本次研究範圍：只探索軟體部分的創新，維持 cross-filter 組內共享 mask 的
結構化剪枝約束。可討論重要性準則、分組、稀疏預算分配、mask 選擇與學習、
訓練/fine-tuning 策略及公平實驗設計。若改動分組，需說明如何維持共享結構。
不要提出 RTL/Verilog、FPGA、PE/記憶體架構或新硬體加速器設計；
不要把理論 MAC 減少當成實測加速。不要推斷任何舊方法已成功或失敗。
"""
SYSTEM = """你是協助探索 CNN 結構化剪枝新想法的研究討論夥伴，使用繁體中文。
以使用者當前提出的問題與限制為主，開放探索可行方向，不預設其舊實驗結果。
清楚區分推測與已驗證事實，不虛構論文、數據或新穎性。
沒有查閱外部文獻時，不宣稱已確認學術新穎性。
需要時協助設計最小驗證實驗與公平對照，保持互動討論。
"""
SYSTEM += "\n\n" + ARCHITECTURE


def context_files(extra):
    paths = [PAPER] + [ROOT / name for name in extra]
    result = []
    for path in dict.fromkeys(p.resolve() for p in paths):
        if not path.is_relative_to(ROOT):
            raise ValueError(f"只能加入專案內的檔案：{path}")
        if path.suffix.lower() not in {".md", ".txt", ".csv", ".json", ".py"}:
            raise ValueError(f"不支援的文字檔案：{path}")
        if not path.is_file():
            raise ValueError(f"找不到檔案：{path}")
        result.append(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", "claude-fable-5-1"))
    parser.add_argument("--file", action="append", default=[], help="額外檔案，相對於專案根目錄")
    parser.add_argument("--session", default=".claude_research/ideas_session.json", help="本機對話檔，相對於專案根目錄")
    parser.add_argument("--preview", action="store_true", help="只顯示待傳送檔案，不呼叫 API")
    parser.add_argument("--check", action="store_true", help="只查詢模型，驗證金鑰與模型權限")
    args = parser.parse_args()
    paths = context_files(args.file)
    parts = []
    for path in paths:
        content = path.read_text(encoding="utf-8")
        if len(content) > 120000:
            raise ValueError(f"檔案超過 120,000 字元，請先整理摘要：{path}")
        name = path.relative_to(ROOT)
        print(f"研究背景：{name} ({len(content):,} 字元)")
        parts.append(f"<project_file path={json.dumps(str(name))}>\n{content}\n</project_file>")
    context = "\n\n".join(parts)
    if len(context) > 200000:
        raise ValueError("研究背景超過 200,000 字元，請縮減預設背景檔案。")
    if args.preview:
        print(f"合計 {len(context):,} 字元；未呼叫 API。")
        return
    if not args.model:
        raise ValueError("請用 --model 或 ANTHROPIC_MODEL 指定帳戶可用的模型 ID。")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("請先設定 ANTHROPIC_API_KEY 環境變數。")
    try:
        from anthropic import Anthropic, APIError
    except ImportError:
        raise ValueError("請先執行 python -m pip install anthropic") from None
    session = (ROOT / args.session).resolve()
    if not session.is_relative_to(ROOT / ".claude_research"):
        raise ValueError("對話檔必須位於專案 .claude_research/ 目錄。")
    messages = json.loads(session.read_text(encoding="utf-8")) if session.exists() else []
    if not isinstance(messages, list):
        raise ValueError("對話檔格式錯誤。")
    client = Anthropic()
    print(f"API 網址：{client.base_url}；模型：{args.model}")
    if args.check:
        try:
            model = client.models.retrieve(args.model)
        except APIError as exc:
            raise ValueError(f"連線檢查失敗：{exc}") from None
        print(f"金鑰與模型查詢成功：{model.id}；未傳送研究背景或產生模型回覆。")
        return
    print("上列背景與對話會傳送到所列 API 平台，依平台規則計費。")
    print("輸入研究問題開始；/reset 清除對話，/quit 離開。")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question == "/quit":
            break
        if question == "/reset":
            messages = []
            session.unlink(missing_ok=True)
            print("已清除本機對話。")
            continue
        if not question:
            continue
        pending = messages + [{"role": "user", "content": question}]
        try:
            response = client.messages.create(
                model=args.model, max_tokens=4096,
                system=SYSTEM + ("\n\n使用者指定的參考資料：\n" + context if context else ""),
                messages=pending,
            )
        except APIError as exc:
            print(f"API 呼叫失敗（對話未更新）：{exc}", file=sys.stderr)
            continue
        answer = "\n".join(b.text for b in response.content if b.type == "text")
        print(f"\nClaude：\n{answer}")
        if response.stop_reason == "max_tokens":
            print("（回覆達到長度上限，可以請 Claude 繼續。）")
        messages = pending + [{"role": "assistant", "content": answer}]
        session.parent.mkdir(parents=True, exist_ok=True)
        temporary = session.with_suffix(".tmp")
        temporary.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(session)
        print(f"[tokens：輸入 {response.usage.input_tokens}，輸出 {response.usage.output_tokens}]")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        sys.exit(str(exc))
