# qwen-dpo-alignment — Qwen2.5-0.5B 的 DPO 偏好對齊

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tun0000/qwen-dpo-alignment/blob/main/dpo_colab.ipynb)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

用 TRL 的 `DPOTrainer` + LoRA，在**免費的 Google Colab（T4 / L4）**上，讓
[`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
從人類偏好資料 [`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
學會「兩個回答之中，哪一個更好」。

## DPO 是什麼？（給沒做過對齊的人的三段話）

**① SFT 的極限。** 指令微調（SFT）只教模型「模仿好答案長什麼樣」。但很多時候兩個回答都
文法通順、都切題，差別在一個更有幫助、更誠實、更完整——這種「相對好壞」SFT 學不到，
因為它從頭到尾只看過單一個標準答案，沒看過比較。

**② 用偏好資料對直接學比較。** ultrafeedback_binarized 的每筆資料都是同一個 prompt 配上
一好一壞兩個回答（`chosen` / `rejected`）。傳統 RLHF 要先用這種資料訓練一個獨立的
reward model，再用 PPO 這類強化學習演算法去最佳化——流程長、超參數敏感、很吃算力。
DPO（Direct Preference Optimization）的洞見是：這整套可以化簡成一條對比式的損失函數，
讓模型**直接**從成對比較學習，不需要 reward model，也不需要強化學習迴圈。

**③ 隱式獎勵與 beta。** DPO 把「獎勵」定義成訓練中模型（policy）與訓練前模型（reference）
對同一段文字的 log 機率比值：模型愈是比原本更願意說某句話，這句話的隱式獎勵就愈高。
訓練目標就是拉大 chosen 與 rejected 的隱式獎勵差距。超參數 **beta** 控制拉開的力道：
beta 越大越保守（緊貼 reference、變化小），越小越激進（對齊訊號強，但容易偏離原模型、
輸出劣化）。本專案用常見的 `beta = 0.1`。

## 專案結構

```
dpo_colab.ipynb            # 主 notebook：在 Colab 上完成資料處理 → DPO 訓練 → 對照 → 推送 HF
scripts/dpo_formatting.py  # 共用格式化模組：message list → DPOTrainer 標準格式（notebook 與本機共用）
scripts/preview_pairs.py   # 本機驗證：下載 20 對資料、套用相同轉換並列印
requirements.txt           # 本機驗證依賴（Colab 依賴由 notebook 自行安裝）
```

## 在 Colab 執行

1. 點上方 **Open in Colab** 徽章開啟 `dpo_colab.ipynb`。
2. `執行階段 → 變更執行階段類型 → T4 GPU`（或 L4；程式會自動偵測用 fp16 / bf16）。
3. 左側**鑰匙圖示（Secrets）**新增 `HF_TOKEN`：填入 Hugging Face 的 **write 權限** token，
   並開啟「筆記本存取權」。
4. 建議先把參數 cell 的 `SMOKE_TEST = True` 跑一次全部（約 3–5 分鐘）確認流程，
   再改回 `False` 正式訓練（6000 對、T4 約 40–70 分鐘）。
5. `執行階段 → 全部執行`。訓練完成後 notebook 會自動：
   - 畫出 loss / rewards-accuracies / rewards-margins 曲線；
   - 用 5 個 prompt 並排對照「原始模型 vs DPO 後模型」的輸出；
   - 把 **merged 模型**與自動產生的 model card 推送到 Hugging Face Hub。

## 本機驗證資料格式（免 GPU、免 torch）

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\preview_pairs.py
```

會串流下載 20 對偏好資料，印出原始的 message-list 結構與轉換後的
prompt / chosen / rejected 字串，並檢查 Qwen chat template 套用是否正確
（生成標記、EOS 收尾、前綴一致性）。

## 成果（2026-07-11，Colab L4 實測）

- **模型**：[steven0226/qwen2.5-0.5b-dpo-ultrafeedback](https://huggingface.co/steven0226/qwen2.5-0.5b-dpo-ultrafeedback)（merged、bf16，推論不需 PEFT）
- **rewards/accuracies**：最終滾動平均 **0.639**，訓練中後段穩定於 0.65–0.70（隨機基線 0.5）
- **DPO loss**：0.693（ln 2 理論起點）→ ~0.62；**rewards/margins** 持續擴大至 ~0.25，無 reward hacking 跡象
- 訓練規模：6000 對、1 epoch、375 steps，L4 上約 25 分鐘
- Model card 內含：DPO 方法說明、超參數、訓練曲線圖、before/after 對照範例。

## License

[Apache-2.0](LICENSE)
