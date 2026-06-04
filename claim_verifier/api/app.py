"""
FastAPI web API and HTML frontend (T10.3).

Endpoints:
    GET  /         → HTML single-page frontend
    GET  /health   → {"status": "ok", "backend": "<type>"}
    POST /verify   → multipart upload; returns JSON verification report

Environment variables:
    BACKEND_TYPE       ollama (default) | anthropic | stub
    ANTHROPIC_API_KEY  required when BACKEND_TYPE=anthropic
    API_KEY            if set, POST /verify requires X-API-Key header

Run locally:
    uvicorn claim_verifier.api.app:app --host 0.0.0.0 --port 8000
    python -m claim_verifier.api
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from claim_verifier.backends import LLMBackend
from claim_verifier.judge import DiagnosisJudge, LLMJudge
from claim_verifier.pipeline import run, run_from_text
from claim_verifier.stages.ingestion import IngestionError, ingest_audio, ingest_document

# ---------------------------------------------------------------------------
# Embedded HTML frontend
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Medical Claim Verifier</title>
<script src="https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"></script>
<style>
:root{--blue:#2563eb;--blue-d:#1d4ed8;--green:#16a34a;--amber:#d97706;--red:#dc2626;--gray:#6b7280;--bg:#f1f5f9;--card:#fff;--border:#e2e8f0;--text:#1e293b;--muted:#64748b}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.6;min-height:100vh}
.hdr{background:var(--blue);color:#fff;padding:1.25rem 2rem}
.hdr h1{font-size:1.3rem;font-weight:700}
.hdr p{font-size:.85rem;opacity:.85;margin-top:.2rem}
.wrap{max-width:760px;margin:2rem auto;padding:0 1rem}
.card{background:var(--card);border-radius:8px;border:1px solid var(--border);padding:1.5rem;margin-bottom:1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.card h2{font-size:1rem;font-weight:600;margin-bottom:1.2rem}
.fld{margin-bottom:1rem}
.fld>label{display:block;font-size:.78rem;font-weight:700;color:var(--muted);margin-bottom:.3rem;text-transform:uppercase;letter-spacing:.04em}
input[type=text],input[type=file]{width:100%;padding:.45rem .7rem;border:1px solid var(--border);border-radius:6px;font-size:.9rem;background:#f8fafc;transition:border-color .15s}
input[type=text]:focus{outline:none;border-color:var(--blue)}
.radio-row{display:flex;gap:1.25rem;margin-bottom:.7rem}
.radio-row label{display:flex;align-items:center;gap:.35rem;font-size:.9rem;cursor:pointer;color:var(--text);font-weight:normal;text-transform:none;letter-spacing:normal}
.btn{display:block;width:100%;padding:.65rem;background:var(--blue);color:#fff;border:none;border-radius:6px;font-size:.95rem;font-weight:600;cursor:pointer;transition:background .15s;margin-top:1.2rem}
.btn:hover{background:var(--blue-d)}
.btn:disabled{background:var(--muted);cursor:not-allowed}
.spin{display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,.35);border-top-color:#fff;border-radius:50%;animation:spin .65s linear infinite;margin-right:.45rem;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
.badge{display:inline-block;padding:.2rem .7rem;border-radius:999px;font-size:.78rem;font-weight:700;letter-spacing:.04em}
.b-LOW{background:#dcfce7;color:var(--green)}
.b-MEDIUM{background:#fef3c7;color:var(--amber)}
.b-HIGH{background:#fee2e2;color:var(--red)}
.b-INSUFFICIENT{background:#f1f5f9;color:var(--gray)}
.res-hdr{display:flex;align-items:center;gap:.9rem;margin-bottom:1rem}
.res-hdr h2{margin:0}
.err-box{background:#fee2e2;border:1px solid #fca5a5;border-radius:6px;padding:.65rem .9rem;color:#991b1b;font-size:.85rem;margin-bottom:.9rem}
.rpt{font-size:.88rem}
.rpt h1{font-size:1.05rem;margin-bottom:.6rem}
.rpt h2{font-size:.88rem;margin:1rem 0 .4rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.rpt table{width:100%;border-collapse:collapse;margin:.4rem 0 .9rem;font-size:.85rem}
.rpt th{background:#f8fafc;text-align:left;padding:.35rem .7rem;border:1px solid var(--border);font-weight:600}
.rpt td{padding:.35rem .7rem;border:1px solid var(--border)}
.rpt blockquote{border-left:3px solid var(--blue);padding:.4rem .9rem;background:#f0f7ff;margin:.4rem 0;color:var(--muted);font-size:.85rem}
.rpt code{background:#f1f5f9;padding:.1rem .3rem;border-radius:3px;font-family:monospace;font-size:.83em}
.rpt p{margin-bottom:.45rem}
.rpt ul,.rpt ol{padding-left:1.4rem;margin-bottom:.45rem}
#result{display:none}
</style>
</head>
<body>
<div class="hdr">
  <h1>Medical Claim Verification Assistant</h1>
  <p>Verify insurance claims against hospital bills — locally, privately.</p>
</div>
<div class="wrap">
  <div class="card">
    <h2>Verify a Claim</h2>
    <form id="frm">
      <div class="fld">
        <label>Claim ID</label>
        <input type="text" id="cid" placeholder="e.g. CLM-2025-001" required>
      </div>
      <div class="fld">
        <label>Transcript Source</label>
        <div class="radio-row">
          <label><input type="radio" name="src" value="txt" checked onchange="toggle(this.value)"> Text file (.txt)</label>
          <label><input type="radio" name="src" value="audio" onchange="toggle(this.value)"> Audio (WAV / MP3)</label>
        </div>
        <div id="txt-input"><input type="file" id="transcript" accept=".txt"></div>
        <div id="audio-input" style="display:none"><input type="file" id="audio" accept=".wav,.mp3,.m4a,.flac,.ogg"></div>
      </div>
      <div class="fld">
        <label>Hospital Bill (PDF)</label>
        <input type="file" id="document" accept=".pdf" required>
      </div>
      <button type="submit" class="btn" id="sbtn">Verify Claim</button>
    </form>
  </div>
  <div id="result" class="card">
    <div class="res-hdr">
      <h2>Verification Result</h2>
      <span id="badge" class="badge"></span>
    </div>
    <div id="errs"></div>
    <div id="rpt" class="rpt"></div>
  </div>
</div>
<script>
function toggle(v){
  document.getElementById('txt-input').style.display=v==='txt'?'block':'none';
  document.getElementById('audio-input').style.display=v==='audio'?'block':'none';
}
document.getElementById('frm').addEventListener('submit',async e=>{
  e.preventDefault();
  const btn=document.getElementById('sbtn');
  btn.disabled=true;
  btn.innerHTML='<span class="spin"></span>Verifying…';
  document.getElementById('result').style.display='none';
  const src=document.querySelector('input[name=src]:checked').value;
  const fd=new FormData();
  fd.append('claim_id',document.getElementById('cid').value);
  const doc=document.getElementById('document').files[0];
  if(!doc){alert('Please select a hospital bill PDF.');btn.disabled=false;btn.textContent='Verify Claim';return;}
  fd.append('document',doc);
  if(src==='txt'){
    const t=document.getElementById('transcript').files[0];
    if(!t){alert('Please select a transcript file.');btn.disabled=false;btn.textContent='Verify Claim';return;}
    fd.append('transcript',t);
  }else{
    const a=document.getElementById('audio').files[0];
    if(!a){alert('Please select an audio file.');btn.disabled=false;btn.textContent='Verify Claim';return;}
    fd.append('audio',a);
  }
  try{
    const r=await fetch('/verify',{method:'POST',body:fd});
    const d=await r.json();
    const rl=(d.risk_level||'INSUFFICIENT_DATA');
    const badge=document.getElementById('badge');
    badge.textContent=rl.replace(/_/g,' ');
    badge.className='badge b-'+rl.split('_')[0];
    const errs=document.getElementById('errs');
    errs.innerHTML=d.errors&&d.errors.length?'<div class="err-box">'+d.errors.join('<br>')+'</div>':'';
    document.getElementById('rpt').innerHTML=marked.parse(d.report||'');
    const res=document.getElementById('result');
    res.style.display='block';
    res.scrollIntoView({behavior:'smooth'});
  }catch(err){
    alert('Request failed: '+err.message);
  }finally{
    btn.disabled=false;
    btn.textContent='Verify Claim';
  }
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def _make_backend() -> tuple[LLMBackend, DiagnosisJudge]:
    backend_type = os.environ.get("BACKEND_TYPE", "ollama").lower()
    if backend_type == "anthropic":
        from claim_verifier.backends.anthropic_backend import AnthropicBackend
        backend: LLMBackend = AnthropicBackend()
    elif backend_type == "stub":
        from claim_verifier.backends import StubBackend
        backend = StubBackend({})
    else:
        from claim_verifier.backends.ollama import OllamaBackend
        backend = OllamaBackend()
    judge = LLMJudge(backend)
    return backend, judge


# ---------------------------------------------------------------------------
# App factory (injectable for tests)
# ---------------------------------------------------------------------------


def create_app(
    backend: LLMBackend | None = None,
    judge: DiagnosisJudge | None = None,
    api_key: str | None = None,
) -> FastAPI:
    """
    Build the FastAPI application.

    backend / judge: injected at test time; None → created lazily from BACKEND_TYPE env var.
    api_key: if provided (or set via API_KEY env var), POST /verify requires
             an X-API-Key header with this value. Empty string or None → open.
    """
    _resolved_key = api_key if api_key is not None else (os.environ.get("API_KEY") or None)

    # Lazily-initialized backend cache so module import doesn't connect to Ollama.
    _cached: list = [backend, judge]

    def _get() -> tuple[LLMBackend, DiagnosisJudge]:
        if _cached[0] is None:
            _cached[0], _cached[1] = _make_backend()
        return _cached[0], _cached[1]

    # ------------------------------------------------------------------ app

    @asynccontextmanager
    async def _lifespan(app: FastAPI):  # noqa: ARG001
        bk, _ = _get()
        if hasattr(bk, "warmup"):
            bk.warmup()
        yield

    api = FastAPI(
        title="Medical Claim Verification Assistant",
        description="Verify insurance claims against hospital bills.",
        version="0.9.0",
        docs_url="/docs",
        lifespan=_lifespan,
    )

    # ------------------------------------------------------------------ auth

    def _check_key(x_api_key: Optional[str] = Header(default=None)) -> None:
        if _resolved_key and x_api_key != _resolved_key:
            raise HTTPException(status_code=403, detail="Invalid or missing API key")

    # ------------------------------------------------------------------ routes

    @api.get("/", response_class=HTMLResponse, include_in_schema=False)
    def root() -> str:
        return _HTML

    @api.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "backend": os.environ.get("BACKEND_TYPE", "ollama"),
        }

    @api.post("/verify", dependencies=[Depends(_check_key)])
    def verify_endpoint(
        claim_id: str = Form(...),
        document: UploadFile = File(...),
        transcript: Optional[UploadFile] = File(default=None),
        audio: Optional[UploadFile] = File(default=None),
    ) -> JSONResponse:
        if transcript is None and audio is None:
            raise HTTPException(
                status_code=400,
                detail="One of 'transcript' or 'audio' is required.",
            )
        if transcript is not None and audio is not None:
            raise HTTPException(
                status_code=400,
                detail="'transcript' and 'audio' are mutually exclusive.",
            )

        bk, jg = _get()
        tmp = Path(tempfile.mkdtemp())
        try:
            doc_suffix = Path(document.filename or "bill.pdf").suffix or ".pdf"
            doc_path = tmp / f"document{doc_suffix}"
            doc_path.write_bytes(document.file.read())

            if transcript is not None:
                t_suffix = Path(transcript.filename or "t.txt").suffix or ".txt"
                t_path = tmp / f"transcript{t_suffix}"
                t_path.write_bytes(transcript.file.read())
                result = run(claim_id, t_path, doc_path, bk, jg)
            else:
                a_suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
                a_path = tmp / f"audio{a_suffix}"
                a_path.write_bytes(audio.file.read())
                try:
                    transcript_text = ingest_audio(a_path)
                    document_text = ingest_document(doc_path)
                except IngestionError as exc:
                    return JSONResponse({
                        "claim_id": claim_id,
                        "report": f"# Ingestion Error\n\n{exc}",
                        "risk_level": "INSUFFICIENT_DATA",
                        "consistency_score": None,
                        "errors": [str(exc)],
                    })
                result = run_from_text(claim_id, transcript_text, document_text, bk, jg)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        vr = result.verification_result
        return JSONResponse({
            "claim_id": result.claim_id,
            "report": result.report,
            "risk_level": vr.risk_level if vr else "INSUFFICIENT_DATA",
            "consistency_score": vr.consistency_score if vr else None,
            "errors": result.errors,
        })

    return api


# ---------------------------------------------------------------------------
# Module-level instance (for uvicorn)
# ---------------------------------------------------------------------------

app = create_app()
