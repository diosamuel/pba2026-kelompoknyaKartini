# Cursor Chat Session

| Field | Value |
|---|---|
| **Project** | (unknown) |
| **Date** | 2026-05-12 19:25:36 |
| **Session ID** | f2a106fd-d66a-48f3-b869-0fb0f0f5f2ac |

---

## Conversation

---

### 👤 You

> `@src/program/app.py` check how you treat LSTM model on how you test with custom text? because it always error, go back to basic and please read `@notebook/LSTM_notebook.ipynb` in section **'Test with custom text'** or i place snippet here  
> `@LSTM_notebook.ipynb (1-19)`  
> adjust my `@src/program/app.py` in LSTM model test like this

---

### 🤖 Cursor Assistant

Reading `app.py` and the notebook's "Test with custom text" section to align LSTM inference.

---

### 🤖 Cursor Assistant

Checking how the project's LSTM checkpoint was trained and which tokenizer was used.

---

### 🤖 Cursor Assistant

Aligning LSTM inference with the notebook: apply `fullClean` then `encode` using whitespace `split()` (not regex on raw text). Removing the unused `re`/`dlTokenize` path.

---

### 🤖 Cursor Assistant

Here's what was wrong and what we changed.

#### Cause

The Streamlit app treated LSTM like `deep_learning/main.py`: **raw text** + **`re.findall(r"[a-z]+", ...)`**.

In **`LSTM_notebook.ipynb`** ("Test with custom text"), the correct flow is:

1. `EmailBodyPreprocessor().fullClean(text)`
2. `encode(cleaned_text)` where **`tokenize` is `text.split()`**
3. `torch.tensor([encoded_text], ...)` → `model.eval()` → `no_grad` → `prediction.item()` → threshold `0.5`

That mismatch (especially tokenizer + no `fullClean`) can break if your saved weights/vocab were built from the **notebook** pipeline.

---

#### Changes in `app.py`

- Added `lstmTokenizeNotebook` → `str(text).split()` (same as the notebook's `tokenize` on cleaned text).
- **`predictLstm`** now: `fullClean` → `encode(..., lstmTokenizeNotebook, vocab, max_len)` → tensor on device → forward → `prediction.item()` and `0.5` threshold (same structure as your snippet).
- Removed `dlTokenize` and the unused `re` import.
- LSTM branch in **`prediksiEmail`** passes `teks` straight through; cleaning happens inside **`predictLstm`**.
- Expander text updated so it matches this notebook behavior.

```python
# src/program/app.py  —  lines 48–72

def lstmTokenizeNotebook(text: str) -> list[str]:
    """Match ``LSTM_notebook``: ``tokenize`` uses whitespace split on ``fullClean`` output."""
    return str(text).split()


def predictLstm(bundle, raw_text: str):
    """Same pipeline as notebook section *Test with custom text*: fullClean → encode → model."""
    preprocessor = EmailBodyPreprocessor()
    cleaned_text = preprocessor.fullClean(raw_text)

    model = bundle["model"]
    vocab = bundle["vocab"]
    max_len = bundle["max_len"]
    device = bundle["device"]

    encoded_text = encode(cleaned_text, lstmTokenizeNotebook, vocab, max_len=max_len)
    input_tensor = torch.tensor([encoded_text], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)

    prob = float(prediction.item())
    label = "spam" if prob > 0.5 else "ham"
    return label, prob
```

> **Important:** `spam_model_lstm.pth` must be a checkpoint whose **`vocab` was built** with this same preprocessing (cleaned text + whitespace tokens). If the file was produced only with `deep_learning/main.py` (raw + regex tokens, no `fullClean`), numbers may look off even if nothing crashes; in that case you'd either re-export the notebook checkpoint or align training with this inference path.

---

*Exported from [Cursor View](https://github.com/saharmor/cursor-view)*
