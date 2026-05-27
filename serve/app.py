from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator

from .loader import load


_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "checkpoints")
    model, preprocessor, config, meta = load(checkpoint_dir)
    _state.update(model=model, preprocessor=preprocessor, config=config, meta=meta)
    yield
    _state.clear()


app = FastAPI(
    title="BiLSTM Sentiment API",
    description="Student-feedback sentiment analysis with Bahdanau attention.",
    version="1.0.0",
    lifespan=lifespan,
)



class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class TokenWeight(BaseModel):
    token: str
    weight: float


class PredictResponse(BaseModel):
    label: str
    label_index: int
    confidence: float
    probabilities: dict[str, float]
    attention: Optional[list[TokenWeight]] = None



@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(_state)}


@app.get("/model-info")
def model_info():
    if not _state:
        raise HTTPException(503, "Model not loaded")
    cfg = _state["config"]
    return {
        "embedding_type": cfg.embedding_type,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "use_attention": cfg.use_attention,
        "num_classes": cfg.num_classes,
        "label_names": cfg.label_names,
        **_state["meta"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not _state:
        raise HTTPException(503, "Model not loaded")

    model       = _state["model"]
    preprocessor = _state["preprocessor"]
    config       = _state["config"]

    tokens = preprocessor.get_tokens(req.text)
    if not tokens:
        raise HTTPException(422, "Text is empty after preprocessing")

    padded, length = preprocessor.encode_padded(req.text)
    x       = torch.tensor([padded],  dtype=torch.long)
    lengths = torch.tensor([length],  dtype=torch.long)

    with torch.no_grad():
        logits, alpha = model(x, lengths)

    probs    = torch.softmax(logits[0], dim=-1).cpu().numpy()
    pred_idx = int(probs.argmax())
    names    = config.label_names

    def _name(i: int) -> str:
        return names[i] if i < len(names) else str(i)

    attention = None
    if alpha is not None:
        weights   = alpha[0, :length].cpu().numpy().tolist()
        attention = [TokenWeight(token=t, weight=w) for t, w in zip(tokens, weights)]

    return PredictResponse(
        label=_name(pred_idx),
        label_index=pred_idx,
        confidence=float(probs[pred_idx]),
        probabilities={_name(i): float(p) for i, p in enumerate(probs)},
        attention=attention,
    )


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(requests: list[PredictRequest]) -> list[PredictResponse]:
    return [predict(r) for r in requests]



@app.get("/", response_class=HTMLResponse)
def index():
    return _HTML


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Sentiment Analysis</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: #f8fafc; color: #0f172a;
      min-height: 100vh; display: flex; flex-direction: column; align-items: center;
      padding: 3rem 1.5rem;
    }
    .card {
      background: white; border: 1px solid #e2e8f0; border-radius: 12px;
      padding: 2rem; width: 100%; max-width: 760px;
      box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }
    h1 { font-size: 1.5rem; font-weight: 700; margin-bottom: .25rem; }
    .sub { color: #64748b; font-size: .875rem; margin-bottom: 1.5rem; }
    textarea {
      width: 100%; height: 110px; padding: .75rem 1rem; font-size: 1rem;
      border: 1.5px solid #cbd5e1; border-radius: 8px; resize: vertical;
      font-family: inherit; transition: border-color .15s;
    }
    textarea:focus { outline: none; border-color: #3b82f6; }
    .row { display: flex; gap: .75rem; margin-top: .75rem; align-items: center; }
    button {
      padding: .55rem 1.6rem; font-size: .95rem; font-weight: 600;
      background: #2563eb; color: white; border: none; border-radius: 8px;
      cursor: pointer; transition: background .15s;
    }
    button:hover { background: #1d4ed8; }
    button:disabled { background: #93c5fd; cursor: default; }
    .hint { color: #94a3b8; font-size: .8rem; }
    #result { margin-top: 1.75rem; }
    .verdict { font-size: 1.5rem; font-weight: 700; margin-bottom: .85rem; }
    .positive { color: #16a34a; }
    .negative { color: #dc2626; }
    .bar-row { display: flex; align-items: center; gap: .75rem; margin: .3rem 0; font-size: .875rem; }
    .bar-bg { flex: 1; height: 10px; background: #f1f5f9; border-radius: 99px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 99px; transition: width .4s; }
    .section-title {
      font-size: .8rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: .05em; color: #94a3b8; margin: 1.25rem 0 .6rem;
    }
    .tokens { line-height: 2.8; }
    .tok {
      display: inline-block; padding: 2px 7px; margin: 2px 2px;
      border-radius: 5px; font-size: .9rem; cursor: default;
      transition: transform .1s;
    }
    .tok:hover { transform: scale(1.08); }
    #error { color: #dc2626; margin-top: .75rem; font-size: .9rem; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Student Feedback Sentiment</h1>
    <p class="sub">BiLSTM + Bahdanau Attention &mdash; type below and press <strong>Analyse</strong></p>

    <textarea id="txt"
      placeholder="The lectures were well-structured and the professor explained every concept clearly...">
    </textarea>

    <div class="row">
      <button id="btn" onclick="run()">Analyse</button>
      <span class="hint">or Ctrl + Enter</span>
    </div>
    <div id="error"></div>
    <div id="result"></div>
  </div>

  <script>
    const btn = document.getElementById('btn');
    const txt = document.getElementById('txt');
    txt.addEventListener('keydown', e => { if (e.ctrlKey && e.key === 'Enter') run(); });

    async function run() {
      const text = txt.value.trim();
      if (!text) return;
      btn.disabled = true;
      document.getElementById('error').textContent = '';
      document.getElementById('result').innerHTML = '';
      try {
        const r = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        });
        if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.statusText); }
        render(await r.json());
      } catch (e) {
        document.getElementById('error').textContent = 'Error: ' + e.message;
      } finally {
        btn.disabled = false;
      }
    }

    function render(d) {
      const cls = d.label.toLowerCase();
      let h = '';

      // Verdict
      h += `<div class="verdict ${cls}">${d.label} &mdash; ${(d.confidence * 100).toFixed(1)}%</div>`;

      // Probability bars
      for (const [lbl, p] of Object.entries(d.probabilities)) {
        const pct = (p * 100).toFixed(1);
        const color = lbl.toLowerCase() === 'positive' ? '#16a34a' : '#dc2626';
        h += `<div class="bar-row">
          <span style="width:75px;flex-shrink:0">${lbl}</span>
          <div class="bar-bg">
            <div class="bar-fill" style="width:${pct}%;background:${color}"></div>
          </div>
          <span>${pct}%</span>
        </div>`;
      }

      // Attention heatmap
      if (d.attention && d.attention.length) {
        h += `<div class="section-title">Attention weights
          <span style="font-weight:400;letter-spacing:0;text-transform:none">
            (darker = more attended, hover for exact %)
          </span>
        </div>`;
        h += '<div class="tokens">';
        const maxW = Math.max(...d.attention.map(a => a.weight));
        for (const { token, weight } of d.attention) {
          const norm  = maxW > 0 ? weight / maxW : 0;
          const alpha = (norm * 0.82 + 0.07).toFixed(2);
          const light = norm < 0.5;
          h += `<span class="tok"
            style="background:rgba(37,99,235,${alpha});color:${light ? '#1e293b' : 'white'}"
            title="${(weight * 100).toFixed(2)}%">${token}</span>`;
        }
        h += '</div>';
      }

      document.getElementById('result').innerHTML = h;
    }
  </script>
</body>
</html>"""
