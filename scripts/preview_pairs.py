"""本機驗證：串流下載 20 筆 ultrafeedback_binarized 偏好資料，
套用與 dpo_colab.ipynb 完全相同的格式化函式（scripts/dpo_formatting.py）並列印，
確認轉換正確。不需 GPU、不需 torch。

用法：
    .venv\\Scripts\\python.exe scripts\\preview_pairs.py
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # 防 cp950 主控台編碼錯誤

from datasets import load_dataset
from transformers import AutoTokenizer

from dpo_formatting import GENERATION_MARKER, to_dpo_format

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"
SPLIT = "train_prefs"
N_PAIRS = 20
N_FULL = 2       # 前幾對印全文
TRUNC_AT = 200   # 其餘截斷長度（字元）


def trunc(s, n=TRUNC_AT):
    return s if len(s) <= n else s[:n] + f" …（截斷，全長 {len(s)} 字元）"


def main():
    print(f"載入 tokenizer：{MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"串流載入資料集：{DATASET_ID}（split={SPLIT}，取 {N_PAIRS} 筆）")
    ds = load_dataset(DATASET_ID, split=SPLIT, streaming=True)
    rows = list(ds.take(N_PAIRS))

    print("\n=== 欄位 ===")
    print(sorted(rows[0].keys()))

    r0 = rows[0]
    print("\n=== 原始樣本（第 1 筆）— chosen / rejected 是 message list ===")
    print(f"prompt（str）：{trunc(r0['prompt'], 150)}")
    print(f"score_chosen={r0['score_chosen']}  score_rejected={r0['score_rejected']}")
    for key in ("chosen", "rejected"):
        print(f"{key}: list[{len(r0[key])}]")
        for m in r0[key]:
            print(f"  [{m['role']}] {trunc(m['content'], 120)}")

    n_dup = 0
    for i, row in enumerate(rows, 1):
        out = to_dpo_format(row, tokenizer)  # 內含前綴一致性檢查（失敗會 raise）
        assert out["prompt"].endswith(GENERATION_MARKER), f"第 {i} 筆 prompt 未以生成標記結尾"
        assert out["chosen"].endswith(tokenizer.eos_token), f"第 {i} 筆 chosen 未恰以 EOS 結尾"
        assert out["rejected"].endswith(tokenizer.eos_token), f"第 {i} 筆 rejected 未恰以 EOS 結尾"
        # 非空檢查要先去掉 EOS，否則 completion 永遠含 <|im_end|> 而永真
        c_body = out["chosen"].removesuffix(tokenizer.eos_token).strip()
        r_body = out["rejected"].removesuffix(tokenizer.eos_token).strip()
        assert c_body and r_body, f"第 {i} 筆有空白回覆"
        if out["chosen"] == out["rejected"]:
            n_dup += 1  # 資料集已知雜訊：這裡只計數，訓練 notebook 會過濾掉
        show = (lambda s: s) if i <= N_FULL else trunc
        print(f"\n--- Pair {i}/{N_PAIRS} ---")
        print("[prompt]  ", show(out["prompt"]))
        print("[chosen]  ", show(out["chosen"]))
        print("[rejected]", show(out["rejected"]))

    print(f"\nOK：{N_PAIRS} 對轉換全數通過檢查（前綴一致 / 生成標記 / 非空 / 恰以 EOS 收尾）")
    print(f"chosen == rejected：{n_dup} 對（資料集已知雜訊，訓練時由 notebook 過濾）")


if __name__ == "__main__":
    main()
