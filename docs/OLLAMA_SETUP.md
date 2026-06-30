# Running DataPilot AI with Ollama (local LLM)

DataPilot AI's **default** model runs locally through [Ollama](https://ollama.com):
**`qwen2.5-coder:7b`**. This means **no API key, no rate limits, and it works
offline** once the model is downloaded — ideal for sharing the app with others.

This guide covers installing Ollama, getting the model, hardware needs, and how
it's wired into the app.

---

## 1. Install Ollama

Download and install for your OS from <https://ollama.com/download>:

- **Windows / macOS:** run the installer. Ollama starts a background service and
  a local server at `http://localhost:11434`.
- **Linux:** `curl -fsSL https://ollama.com/install.sh | sh`

Verify it's installed:

```bash
ollama --version
```

If the server isn't running for some reason, start it manually with:

```bash
ollama serve
```

---

## 2. Pull the model

Download the default model (one-time, needs internet; ~4.7 GB):

```bash
ollama pull qwen2.5-coder:7b
```

Confirm it's available:

```bash
ollama list          # should list qwen2.5-coder:7b
```

After this download, **no internet is required** to use the app.

---

## 3. Hardware requirements

Ollama serves quantized models. Approximate memory **in use while running**
(system RAM, or GPU VRAM if you have one):

| Model | Download | In-use memory | Comfortable on |
|-------|----------|---------------|----------------|
| `qwen2.5-coder:7b` (default) | ~4.7 GB | ~5–6 GB | 16 GB RAM, or any modern GPU / Apple Silicon |
| `qwen2.5-coder:14b` | ~9 GB | ~9–11 GB | 16 GB+ (tight) / 12 GB+ VRAM |
| `qwen2.5-coder:32b` | ~20 GB | ~20 GB | 24 GB+ VRAM, or 32–64 GB RAM |

Speed notes:
- **GPU or Apple Silicon (M1/M2/M3, 16 GB+):** fast — a phase generates in
  seconds to ~30s.
- **CPU-only:** works but slower (a long generation can take 30s–2 min per
  phase). Fine for recorded demos, sluggish for live ones.
- **Cold start:** the model loads into memory on the first call (a few seconds
  up to ~30s); it stays warm afterwards. For a live demo, do one warm-up run.

Check what's loaded and its memory use:

```bash
ollama ps
```

---

## 4. How it's wired into the app

All model configuration lives in **one file**, [`utils/llm.py`](../utils/llm.py):

- `get_llm()` builds a `ChatOllama` client for the selected model, with
  `num_ctx=8192` (a larger context window, since this app's prompts include
  feature stats and data samples).
- `AVAILABLE_MODELS` lists the choices shown in the sidebar dropdown.
- `DEFAULT_MODEL = "qwen2.5-coder:7b"`.

You don't need to touch any agent or node code — every agent calls `get_llm()`.

### Use a different / larger local model

1. Pull it: `ollama pull qwen2.5-coder:14b`
2. Add it to `AVAILABLE_MODELS` in [`utils/llm.py`](../utils/llm.py):
   ```python
   AVAILABLE_MODELS = {
       "qwen2.5-coder:7b":  "Qwen2.5-Coder 7B (local · Ollama · no key)",
       "qwen2.5-coder:14b": "Qwen2.5-Coder 14B (local · Ollama)",
       "gemini-2.0-flash":  "Gemini 2.0 Flash (Google · needs key)",
   }
   ```
3. Pick it from the sidebar dropdown (or set it as `DEFAULT_MODEL`).

The model id **is** the Ollama model name, so any model you've pulled works by
adding its name here.

---

## 5. Run the app

```bash
uv run streamlit run ui/app.py
```

Make sure **Qwen2.5-Coder 7B (Ollama)** is selected in the sidebar (it's the
default), upload a CSV, and run.

---

## 6. A note on quality

Local models are smaller than hosted frontier models, so they make more coding
mistakes. DataPilot AI is built to absorb that:
- the **self-healing loop** feeds execution errors (and the real library
  signature) back to the model to fix its own code,
- the **review screen** lets you edit the code or add comments and regenerate.

So weaker models degrade gracefully (more retries) rather than breaking. If a
dataset consistently trips the 7B model, step up to `14b`.

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| App shows *"Couldn't reach the local model (Ollama)"* | Ensure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull qwen2.5-coder:7b`). |
| `model "..." not found` | You haven't pulled that model name. Run `ollama pull <name>` or check `ollama list`. |
| Very slow generations | You're likely on CPU. Use a machine with a GPU / Apple Silicon, or a smaller model. |
| Out-of-memory / system freeze | The model is too large for your RAM/VRAM — drop to a smaller variant (e.g. 7b). |
| Want to avoid local setup entirely | Switch the sidebar dropdown to a cloud model — **Gemini** or **Groq** (set the matching API key in `.env`). |

---

## 8. Switching to a cloud model instead

No local setup needed if you'd rather use a hosted model — just pick it in the
sidebar dropdown (no code changes):

- **Gemini 2.0 Flash** — get a key at
  <https://aistudio.google.com/apikey> and set `GOOGLE_API_KEY` in `.env`.
- **Groq (Llama 3.3 / GPT-OSS)** — get a key at
  <https://console.groq.com/keys> and set `GROQ_API_KEY` in `.env`.

See the main [README](../README.md) for details.
