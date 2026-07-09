"""ultrafeedback_binarized → TRL DPOTrainer「標準格式」的共用轉換模組。

HuggingFaceH4/ultrafeedback_binarized 的 chosen / rejected 欄位是 message list
（[{"role": "user", ...}, {"role": "assistant", ...}]），兩者共享同一段對話前綴，
只有最後一則 assistant 回覆不同。

這裡把每筆資料轉成 TRL DPOTrainer 的「標準格式」（explicit prompt）：
prompt / chosen / rejected 三個純字串，其中 chosen / rejected 只含接續文字
（不重複 prompt）。作法鏡射 TRL 的 maybe_apply_chat_template：
先用 add_generation_prompt=True 渲染 prompt，再渲染完整對話後切掉 prompt 前綴。

此檔案同時被 dpo_colab.ipynb（於 Colab 執行時從 GitHub raw URL 下載）與
scripts/preview_pairs.py（本機）import，確保兩邊的格式化邏輯完全一致。
"""

# Qwen chat template 的生成起始標記：格式化後的 prompt 必須以它結尾
GENERATION_MARKER = "<|im_start|>assistant\n"


def to_dpo_format(example, tokenizer):
    """把一筆 ultrafeedback_binarized 資料轉成 prompt/chosen/rejected 純字串。

    completion 結尾若是 eos_token + "\\n"（Qwen chat template 的預設輸出），
    會把換行去掉、讓字串恰好以 eos_token 結尾 —— TRL 的 add_eos 是字串層級的
    endswith(eos_token) 檢查，不修剪的話會被誤判而重複補上第二個 EOS。
    """
    prompt_messages = example["chosen"][:-1]
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    out = {"prompt": prompt}
    for key in ("chosen", "rejected"):
        full = tokenizer.apply_chat_template(example[key], tokenize=False)
        if not full.startswith(prompt):
            raise ValueError(
                f"chat template 輸出與 prompt 前綴不一致（{key}），無法安全切分"
            )
        completion = full[len(prompt):]
        if completion.endswith(tokenizer.eos_token + "\n"):
            completion = completion[:-1]
        out[key] = completion
    return out
