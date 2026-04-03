import os
import sys
import copy
import time
import uuid
import glob
import wave
import struct
import re
import threading
import torch
from flask import Flask, request, jsonify, send_file, render_template_string, Response
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

app = Flask(__name__)

# ── Dirs
VOICES_DIR = os.path.join(os.path.dirname(__file__), "demo", "voices", "streaming_model")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs")
PREVIEW_DIR = os.path.join(OUTPUT_DIR, "previews")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

# ── Per-voice metadata
VOICE_META = {
    "en-carter_man":  {"label":"Carter",   "lang":"🇬🇧 English",   "gender":"Male",   "style":"Warm & Narrative",      "emoji":"🎙️"},
    "en-davis_man":   {"label":"Davis",    "lang":"🇬🇧 English",   "gender":"Male",   "style":"Deep & Authoritative",  "emoji":"🎤"},
    "en-emma_woman":  {"label":"Emma",     "lang":"🇬🇧 English",   "gender":"Female", "style":"Bright & Friendly",     "emoji":"💁‍♀️"},
    "en-frank_man":   {"label":"Frank",    "lang":"🇬🇧 English",   "gender":"Male",   "style":"Casual & Conversational","emoji":"😎"},
    "en-grace_woman": {"label":"Grace",    "lang":"🇬🇧 English",   "gender":"Female", "style":"Elegant & Clear",       "emoji":"🌸"},
    "en-mike_man":    {"label":"Mike",     "lang":"🇬🇧 English",   "gender":"Male",   "style":"Energetic & Young",     "emoji":"⚡"},
    "in-samuel_man":  {"label":"Samuel",   "lang":"🇮🇳 Indian",    "gender":"Male",   "style":"Warm Indian accent",    "emoji":"🪔"},
    "de-spk0_man":    {"label":"Klaus",    "lang":"🇩🇪 German",    "gender":"Male",   "style":"Precise & Formal",      "emoji":"🏛️"},
    "de-spk1_woman":  {"label":"Helga",    "lang":"🇩🇪 German",    "gender":"Female", "style":"Smooth & Crisp",        "emoji":"🌷"},
    "fr-spk0_man":    {"label":"Pierre",   "lang":"🇫🇷 French",    "gender":"Male",   "style":"Romantic & Smooth",     "emoji":"🥐"},
    "fr-spk1_woman":  {"label":"Camille",  "lang":"🇫🇷 French",    "gender":"Female", "style":"Chic & Elegant",        "emoji":"🗼"},
    "it-spk0_woman":  {"label":"Sofia",    "lang":"🇮🇹 Italian",   "gender":"Female", "style":"Expressive & Warm",     "emoji":"🍕"},
    "it-spk1_man":    {"label":"Marco",    "lang":"🇮🇹 Italian",   "gender":"Male",   "style":"Bold & Passionate",     "emoji":"🎭"},
    "jp-spk0_man":    {"label":"Kenji",    "lang":"🇯🇵 Japanese",  "gender":"Male",   "style":"Calm & Measured",       "emoji":"🗾"},
    "jp-spk1_woman":  {"label":"Yuki",     "lang":"🇯🇵 Japanese",  "gender":"Female", "style":"Gentle & Melodic",      "emoji":"🌸"},
    "kr-spk0_woman":  {"label":"Jiyeon",   "lang":"🇰🇷 Korean",    "gender":"Female", "style":"Bright & Modern",       "emoji":"🎵"},
    "kr-spk1_man":    {"label":"Minjun",   "lang":"🇰🇷 Korean",    "gender":"Male",   "style":"Clear & Confident",     "emoji":"🔥"},
    "nl-spk0_man":    {"label":"Daan",     "lang":"🇳🇱 Dutch",     "gender":"Male",   "style":"Direct & Clear",        "emoji":"🌷"},
    "nl-spk1_woman":  {"label":"Fenna",    "lang":"🇳🇱 Dutch",     "gender":"Female", "style":"Lively & Warm",         "emoji":"🧀"},
    "pl-spk0_man":    {"label":"Piotr",    "lang":"🇵🇱 Polish",    "gender":"Male",   "style":"Strong & Clear",        "emoji":"🦅"},
    "pl-spk1_woman":  {"label":"Zofia",    "lang":"🇵🇱 Polish",    "gender":"Female", "style":"Soft & Melodic",        "emoji":"🪻"},
    "pt-spk0_woman":  {"label":"Beatriz",  "lang":"🇵🇹 Portuguese","gender":"Female", "style":"Warm & Musical",        "emoji":"🎶"},
    "pt-spk1_man":    {"label":"Tomas",    "lang":"🇵🇹 Portuguese","gender":"Male",   "style":"Deep & Resonant",       "emoji":"⚓"},
    "sp-spk0_woman":  {"label":"Valentina","lang":"🇪🇸 Spanish",   "gender":"Female", "style":"Vibrant & Rich",        "emoji":"💃"},
    "sp-spk1_man":    {"label":"Alejandro","lang":"🇪🇸 Spanish",   "gender":"Male",   "style":"Smooth & Charismatic",  "emoji":"🎸"},
}

# ── Job tracking (for progress)
_jobs = {}   # job_id -> dict{status,progress,msg,file,error,started}
_gen_lock = threading.Lock()  # one generation at a time

# ── Load model
print("⏳ Loading VibeVoice model... please wait ~1 minute")
MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32

_processor = VibeVoiceStreamingProcessor.from_pretrained(MODEL_PATH)
_model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
    MODEL_PATH, torch_dtype=DTYPE,
    device_map=DEVICE if DEVICE != "mps" else "cpu",
    attn_implementation="sdpa",
)
if DEVICE == "mps":
    _model.to("mps")
_model.eval()
_model.set_ddpm_inference_steps(num_steps=5)
print(f"✅ Model ready on {DEVICE}!")

# ────────────────────────────────────────
# Helpers
# ────────────────────────────────────────
def get_voices():
    out = []
    for f in sorted(glob.glob(os.path.join(VOICES_DIR, "*.pt"))):
        key = os.path.splitext(os.path.basename(f))[0].lower()
        meta = VOICE_META.get(key, {
            "label": key.split("-")[-1].split("_")[0].capitalize(),
            "lang":"🌐 Unknown","gender":"Unknown","style":"Natural","emoji":"🎙️"
        })
        out.append({"id": key, "path": f, **meta})
    return out

_voice_map = {v["id"]: v for v in get_voices()}


def _run_tts_core(text: str, voice_path: str, output_path: str,
                  job_id: str = None, estimated_chars: int = None):
    """Core TTS. Updates _jobs[job_id] progress if job_id given."""
    prefilled = torch.load(voice_path, map_location=DEVICE, weights_only=False)
    text = text.replace("\u2019","'").replace("\u201c",'"').replace("\u201d",'"')

    inputs = _processor.process_input_with_cached_prompt(
        text=text, cached_prompt=prefilled,
        padding=True, return_tensors="pt", return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(DEVICE)

    # Estimate total speech tokens from char count (approx 1 token ≈ 0.133 sec ≈ ~3 chars)
    total_tokens_est = max(1, len(text) // 3)

    class ProgressHook:
        """Captures stdout lines containing token counts."""
        def __init__(self, real_stdout):
            self._r = real_stdout
        def write(self, s):
            self._r.write(s)
            if job_id and "generated" in s and "speech token" in s:
                try:
                    # "Prefilled 500 text tokens, generated 600 speech tokens"
                    part = s.split("generated")[1].strip()
                    done = int(part.split()[0].replace(",",""))
                    pct = min(98, int(done / total_tokens_est * 100))
                    _jobs[job_id]["progress"] = pct
                    _jobs[job_id]["msg"] = f"Generating… {pct}% ({done} tokens)"
                except Exception:
                    pass
        def flush(self):
            self._r.flush()

    old = sys.stdout
    if job_id:
        sys.stdout = ProgressHook(old)

    try:
        with torch.no_grad():
            outputs = _model.generate(
                **inputs, max_new_tokens=None, cfg_scale=1.5,
                tokenizer=_processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=True,
                all_prefilled_outputs=copy.deepcopy(prefilled),
            )
    finally:
        sys.stdout = old

    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        _processor.save_audio(outputs.speech_outputs[0], output_path=output_path)
        return True
    return False


# ────────────────────────────────────────
# Dual-Voice Script Parser
# ────────────────────────────────────────
def parse_dual_script(script: str):
    """
    Parse a script with [MALE]: and [FEMALE]: tags.
    Returns list of (speaker, text) tuples.
    Speaker is 'male' or 'female'.
    Lines without a tag are assigned to the last speaker (or 'male' by default).
    """
    segments = []
    last_speaker = 'male'
    for line in script.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^\[(MALE|FEMALE|HOST|GUEST)\]\s*:?\s*(.+)', line, re.IGNORECASE)
        if m:
            tag = m.group(1).upper()
            text = m.group(2).strip()
            if tag in ('MALE', 'HOST'):
                last_speaker = 'male'
            else:
                last_speaker = 'female'
            if text:
                segments.append((last_speaker, text))
        else:
            # No tag — assign to last speaker
            if line:
                segments.append((last_speaker, line))
    return segments


def merge_wav_files(wav_paths: list, output_path: str):
    """Concatenate multiple WAV files into one. All must have same params."""
    if not wav_paths:
        return False
    data_chunks = []
    params = None
    for p in wav_paths:
        if not os.path.exists(p):
            continue
        with wave.open(p, 'rb') as wf:
            if params is None:
                params = wf.getparams()
            data_chunks.append(wf.readframes(wf.getnframes()))
    if not data_chunks or params is None:
        return False
    with wave.open(output_path, 'wb') as out_wf:
        out_wf.setparams(params)
        for chunk in data_chunks:
            out_wf.writeframes(chunk)
    return True


# ────────────────────────────────────────
# Background job runner (single voice)
# ────────────────────────────────────────
def _bg_generate(job_id, text, voice_id):
    with _gen_lock:
        try:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["msg"]    = "Starting generation…"
            voice = _voice_map.get(voice_id)
            if not voice:
                raise ValueError("Voice not found")
            out_path = os.path.join(OUTPUT_DIR, f"podcast_{voice_id}_{job_id}.wav")
            ok = _run_tts_core(text, voice["path"], out_path, job_id=job_id)
            if ok:
                _jobs[job_id]["status"]   = "done"
                _jobs[job_id]["progress"] = 100
                _jobs[job_id]["file"]     = os.path.basename(out_path)
                _jobs[job_id]["msg"]      = "Done!"
            else:
                raise RuntimeError("No audio output")
        except Exception as e:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(e)
            _jobs[job_id]["msg"]    = f"Error: {e}"


# ────────────────────────────────────────
# Background job runner (dual voice)
# ────────────────────────────────────────
def _bg_generate_dual(job_id, script, male_voice_id, female_voice_id):
    with _gen_lock:
        try:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["msg"]    = "Parsing script…"
            _jobs[job_id]["progress"] = 2

            segments = parse_dual_script(script)
            if not segments:
                raise ValueError("No valid segments found in script. Use [MALE]: or [FEMALE]: tags.")

            male_voice  = _voice_map.get(male_voice_id)
            female_voice = _voice_map.get(female_voice_id)
            if not male_voice:
                raise ValueError(f"Male voice '{male_voice_id}' not found")
            if not female_voice:
                raise ValueError(f"Female voice '{female_voice_id}' not found")

            seg_paths = []
            total = len(segments)
            tmp_dir = os.path.join(OUTPUT_DIR, f"tmp_{job_id}")
            os.makedirs(tmp_dir, exist_ok=True)

            for i, (speaker, text) in enumerate(segments):
                pct = int(5 + (i / total) * 88)
                _jobs[job_id]["progress"] = pct
                _jobs[job_id]["msg"] = f"Generating segment {i+1}/{total} ({speaker.upper()})…"

                seg_path = os.path.join(tmp_dir, f"seg_{i:04d}_{speaker}.wav")
                voice = male_voice if speaker == 'male' else female_voice
                ok = _run_tts_core(text, voice["path"], seg_path)
                if ok:
                    seg_paths.append(seg_path)
                else:
                    print(f"[WARN] Segment {i} failed, skipping: {text[:60]}")

            if not seg_paths:
                raise RuntimeError("All segments failed to generate")

            _jobs[job_id]["msg"] = "Merging audio segments…"
            _jobs[job_id]["progress"] = 95

            out_path = os.path.join(OUTPUT_DIR, f"dual_podcast_{job_id}.wav")
            ok = merge_wav_files(seg_paths, out_path)

            # Cleanup temp files
            for p in seg_paths:
                try: os.remove(p)
                except: pass
            try: os.rmdir(tmp_dir)
            except: pass

            if ok:
                _jobs[job_id]["status"]   = "done"
                _jobs[job_id]["progress"] = 100
                _jobs[job_id]["file"]     = os.path.basename(out_path)
                _jobs[job_id]["msg"]      = f"Done! {len(seg_paths)} segments merged."
            else:
                raise RuntimeError("Failed to merge audio files")

        except Exception as e:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(e)
            _jobs[job_id]["msg"]    = f"Error: {e}"


def _bg_preview(voice_id, cache_path):
    try:
        voice = _voice_map.get(voice_id)
        if not voice:
            return
        text = "Hello! This is a sample of my voice. I hope you enjoy listening."
        _run_tts_core(text, voice["path"], cache_path)
    except Exception as e:
        print(f"Preview error for {voice_id}: {e}")


# ────────────────────────────────────────
# Flask routes
# ────────────────────────────────────────
@app.route("/api/voices")
def api_voices():
    return jsonify([
        {k: v[k] for k in ("id","label","lang","gender","style","emoji")}
        for v in get_voices()
    ])


@app.route("/api/preview/<voice_id>")
def api_preview(voice_id):
    """Returns audio if cached, else starts background generation and returns 202."""
    if voice_id not in _voice_map:
        return jsonify({"error": "Not found"}), 404
    cache = os.path.join(PREVIEW_DIR, f"{voice_id}.wav")
    if os.path.exists(cache):
        return send_file(cache, mimetype="audio/wav")
    # Kick off background preview generation if not already running
    marker = cache + ".generating"
    if not os.path.exists(marker):
        open(marker, "w").close()
        def _run():
            _bg_preview(voice_id, cache)
            if os.path.exists(marker):
                os.remove(marker)
        threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "generating"}), 202


@app.route("/api/preview_status/<voice_id>")
def api_preview_status(voice_id):
    cache = os.path.join(PREVIEW_DIR, f"{voice_id}.wav")
    marker = cache + ".generating"
    if os.path.exists(cache):
        return jsonify({"ready": True})
    if os.path.exists(marker):
        return jsonify({"ready": False, "generating": True})
    return jsonify({"ready": False, "generating": False})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.json or {}
    script   = data.get("script", "").strip()
    voice_id = data.get("voice_id", "")
    if not script:
        return jsonify({"error": "Empty script"}), 400
    if voice_id not in _voice_map:
        return jsonify({"error": "Invalid voice"}), 400
    if _gen_lock.locked():
        return jsonify({"error": "Another generation is already running. Please wait."}), 429

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "status":"queued","progress":0,"msg":"Queued…",
        "file":None,"error":None,"started":time.time()
    }
    threading.Thread(target=_bg_generate, args=(job_id, script, voice_id), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/generate_dual", methods=["POST"])
def api_generate_dual():
    data = request.json or {}
    script         = data.get("script", "").strip()
    male_voice_id  = data.get("male_voice_id", "")
    female_voice_id = data.get("female_voice_id", "")
    if not script:
        return jsonify({"error": "Empty script"}), 400
    if male_voice_id not in _voice_map:
        return jsonify({"error": f"Invalid male voice: {male_voice_id}"}), 400
    if female_voice_id not in _voice_map:
        return jsonify({"error": f"Invalid female voice: {female_voice_id}"}), 400
    if _gen_lock.locked():
        return jsonify({"error": "Another generation is already running. Please wait."}), 429

    # Quick validation
    segments = parse_dual_script(script)
    if not segments:
        return jsonify({"error": "No valid segments found. Use [MALE]: or [FEMALE]: tags in your script."}), 400

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "status":"queued","progress":0,"msg":"Queued…",
        "file":None,"error":None,"started":time.time(),
        "total_segments": len(segments)
    }
    threading.Thread(
        target=_bg_generate_dual,
        args=(job_id, script, male_voice_id, female_voice_id),
        daemon=True
    ).start()
    return jsonify({"job_id": job_id, "total_segments": len(segments)})


@app.route("/api/job/<job_id>")
def api_job(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    elapsed = round(time.time() - job["started"], 1)
    return jsonify({**job, "elapsed": elapsed})


@app.route("/api/download/<filename>")
def api_download(filename):
    # Security: only allow names without directory traversal
    filename = os.path.basename(filename)
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    return send_file(path, as_attachment=True, mimetype="audio/wav")


@app.route("/api/preview_file/<voice_id>")
def api_preview_file(voice_id):
    """Serve cached preview file directly."""
    cache = os.path.join(PREVIEW_DIR, f"{voice_id}.wav")
    if os.path.exists(cache):
        return send_file(cache, mimetype="audio/wav")
    return jsonify({"error":"not ready"}), 404


@app.route("/")
def index():
    return render_template_string(HTML)


# ────────────────────────────────────────
# HTML — ElevenLabs style
# ────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>VibeVoice Studio</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#09090f;--surface:#121220;--card:#181828;--card-hov:#1f1f35;
  --bdr:#252540;--purple:#7c5cfc;--pglow:rgba(124,92,252,.22);
  --pdark:#5a3fd6;--pink:#e040fb;--green:#00e5a0;--amber:#ffbb00;
  --text:#ededff;--muted:#7f7f9f;--radius:14px;
}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
/* scrollbar */
::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:var(--surface)}
::-webkit-scrollbar-thumb{background:var(--bdr);border-radius:4px}

/* ── HEADER */
header{display:flex;align-items:center;justify-content:space-between;padding:18px 40px;
  border-bottom:1px solid var(--bdr);backdrop-filter:blur(14px);
  position:sticky;top:0;z-index:200;background:rgba(9,9,15,.88)}
.logo{display:flex;align-items:center;gap:10px;font-size:1.3rem;font-weight:700}
.dot{width:10px;height:10px;border-radius:50%;background:var(--purple);box-shadow:0 0 16px var(--purple)}
.hbadge{background:linear-gradient(135deg,var(--purple),var(--pink));border-radius:8px;
  padding:3px 12px;font-size:.68rem;font-weight:700;letter-spacing:.6px}

/* ── LAYOUT */
.wrap{display:grid;grid-template-columns:1fr 1.15fr;gap:32px;padding:32px 40px;
  max-width:1450px;margin:0 auto}
@media(max-width:950px){.wrap{grid-template-columns:1fr;padding:20px}}

/* ── LABELS */
.label{font-size:.68rem;font-weight:600;letter-spacing:1.4px;text-transform:uppercase;
  color:var(--muted);margin-bottom:12px}

/* ── SEARCH & FILTER */
.vsearch{width:100%;background:var(--surface);border:1px solid var(--bdr);
  border-radius:10px;padding:10px 16px;color:var(--text);font-size:.88rem;
  outline:none;margin-bottom:14px;transition:border .2s;font-family:inherit}
.vsearch:focus{border-color:var(--purple)}
.vsearch::placeholder{color:var(--muted)}
.tabs{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:16px}
.tab{padding:5px 14px;border-radius:20px;font-size:.75rem;font-weight:500;
  cursor:pointer;border:1px solid var(--bdr);background:transparent;
  color:var(--muted);transition:all .2s;font-family:inherit}
.tab.on,.tab:hover{border-color:var(--purple);color:var(--purple);background:var(--pglow)}

/* ── VOICE GRID */
.vgrid{display:grid;grid-template-columns:1fr 1fr;gap:11px;
  max-height:600px;overflow-y:auto;padding-right:4px}
.vcard{background:var(--card);border:2px solid var(--bdr);border-radius:var(--radius);
  padding:14px;cursor:pointer;transition:all .22s;position:relative;overflow:hidden}
.vcard::before{content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,var(--pglow),transparent);opacity:0;transition:opacity .22s}
.vcard:hover{border-color:var(--purple);transform:translateY(-2px);box-shadow:0 8px 28px var(--pglow)}
.vcard:hover::before,.vcard.sel::before{opacity:1}
.vcard.sel{border-color:var(--purple);background:var(--card-hov);box-shadow:0 0 0 3px var(--pglow)}
.vtop{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.vav{width:38px;height:38px;border-radius:50%;
  background:linear-gradient(135deg,var(--purple),var(--pink));
  display:flex;align-items:center;justify-content:center;font-size:1rem}
.gbadge{font-size:.6rem;padding:2px 7px;border-radius:10px;font-weight:600}
.male{background:rgba(64,158,255,.15);color:#409eff;border:1px solid rgba(64,158,255,.3)}
.female{background:rgba(224,64,251,.15);color:#e040fb;border:1px solid rgba(224,64,251,.3)}
.vname{font-weight:600;font-size:.9rem;margin-bottom:1px}
.vlang{font-size:.7rem;color:var(--muted);margin-bottom:3px}
.vstyle{font-size:.68rem;color:var(--muted);font-style:italic}
.chk{position:absolute;top:9px;right:9px;width:20px;height:20px;border-radius:50%;
  background:var(--purple);display:none;align-items:center;justify-content:center;
  font-size:.68rem;font-weight:700}
.vcard.sel .chk{display:flex}

/* PREVIEW BUTTON */
.prevbtn{display:flex;align-items:center;justify-content:center;gap:6px;
  width:100%;margin-top:10px;padding:7px 0;
  border:1px solid var(--bdr);border-radius:8px;
  background:transparent;color:var(--muted);font-size:.73rem;
  cursor:pointer;transition:all .2s;font-family:inherit;position:relative}
.prevbtn:hover{border-color:var(--green);color:var(--green)}
.prevbtn.playing{border-color:var(--green);color:var(--green);animation:rim 1.4s ease infinite}
.prevbtn.loading{border-color:var(--amber);color:var(--amber)}
.prevbtn .pspinner{width:12px;height:12px;border:2px solid var(--amber);
  border-top-color:transparent;border-radius:50%;display:none;
  animation:spin .6s linear infinite}
.prevbtn.loading .pspinner{display:inline-block}
@keyframes rim{0%,100%{box-shadow:0 0 0 0 rgba(0,229,160,.35)}50%{box-shadow:0 0 0 6px rgba(0,229,160,0)}}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Right panel */
.rpanel{display:flex;flex-direction:column;gap:20px}

/* Selected bar */
.selbar{background:linear-gradient(135deg,rgba(124,92,252,.12),rgba(224,64,251,.06));
  border:1px solid rgba(124,92,252,.38);border-radius:var(--radius);
  padding:14px 18px;display:none;align-items:center;gap:14px}
.selbar.vis{display:flex}
.sbem{font-size:1.7rem}
.sbinfo{flex:1}.sbname{font-weight:600;font-size:.95rem}
.sbsub{font-size:.74rem;color:var(--muted);margin-top:2px}
.sbuse{background:var(--purple);color:#fff;border:none;border-radius:8px;
  padding:8px 16px;font-size:.8rem;font-weight:600;cursor:pointer;transition:bg .2s;font-family:inherit}
.sbuse:hover{background:var(--pdark)}

/* status */
.sbar{display:flex;align-items:center;gap:10px;padding:11px 15px;
  background:rgba(124,92,252,.07);border:1px solid rgba(124,92,252,.2);
  border-radius:10px;font-size:.8rem;color:var(--muted)}
.sdot{width:8px;height:8px;border-radius:50%;background:var(--green);
  box-shadow:0 0 8px var(--green);flex-shrink:0;animation:blink 2s ease infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.4}}

/* editor */
.edwrap{background:var(--surface);border:1px solid var(--bdr);
  border-radius:var(--radius);overflow:hidden;transition:border .2s}
.edwrap:focus-within{border-color:var(--purple);box-shadow:0 0 0 3px var(--pglow)}
.edhead{display:flex;align-items:center;justify-content:space-between;
  padding:11px 15px;border-bottom:1px solid var(--bdr)}
.edhead span{font-size:.78rem;color:var(--muted);font-weight:500}
.ccount{font-size:.72rem;color:var(--muted)}
textarea{width:100%;min-height:220px;background:transparent;border:none;
  outline:none;resize:vertical;padding:15px;color:var(--text);
  font-size:.88rem;font-family:'Inter',sans-serif;line-height:1.75}
textarea::placeholder{color:rgba(127,127,159,.4)}

/* settings */
.srow{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.sbox{background:var(--surface);border:1px solid var(--bdr);border-radius:10px;padding:13px 15px}
.sbox label{font-size:.68rem;color:var(--muted);font-weight:500;display:block;
  margin-bottom:6px;text-transform:uppercase;letter-spacing:.6px}
select{width:100%;background:var(--card);border:none;outline:none;
  color:var(--text);font-size:.84rem;font-family:inherit;border-radius:4px;padding:2px}
select option{background:var(--card)}
input[type=range]{width:100%;accent-color:var(--purple);cursor:pointer;margin-top:2px}
.rval{font-size:.7rem;color:var(--muted);margin-top:4px}

/* generate btn */
.genbtn{width:100%;padding:15px;
  background:linear-gradient(135deg,var(--purple),var(--pdark));
  border:none;border-radius:var(--radius);color:#fff;
  font-size:.95rem;font-weight:700;cursor:pointer;
  display:flex;align-items:center;justify-content:center;gap:10px;
  transition:all .22s;position:relative;overflow:hidden;font-family:inherit}
.genbtn:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 12px 32px var(--pglow)}
.genbtn:disabled{opacity:.45;cursor:not-allowed;transform:none}
.gspinner{width:18px;height:18px;border:2px solid rgba(255,255,255,.3);
  border-top-color:#fff;border-radius:50%;display:none;animation:spin .7s linear infinite}
.genbtn.ld .gspinner{display:block}

/* ── PROGRESS CARD */
.progcard{background:var(--card);border:1px solid var(--bdr);
  border-radius:var(--radius);padding:20px;display:none}
.progcard.vis{display:block}
.progtop{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}
.progtitle{font-weight:600;font-size:.92rem}
.progpct{font-size:1.3rem;font-weight:700;color:var(--purple)}
.progbar-wrap{background:rgba(124,92,252,.12);border-radius:50px;
  height:10px;overflow:hidden;margin-bottom:10px}
.progbar{height:100%;border-radius:50px;
  background:linear-gradient(90deg,var(--purple),var(--pink));
  width:0%;transition:width .5s ease;box-shadow:0 0 12px var(--pglow)}
.progmsg{font-size:.78rem;color:var(--muted);display:flex;align-items:center;
  justify-content:space-between}
.elapsed{font-size:.75rem;color:var(--muted)}

/* ── OUTPUT CARD */
.outcard{background:var(--card);border:1px solid var(--bdr);
  border-radius:var(--radius);padding:20px;display:none}
.outcard.vis{display:block}
.outtop{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}
.outtitle{font-weight:600;font-size:.92rem}
.outmeta{font-size:.73rem;color:var(--muted)}
audio{width:100%;border-radius:8px;margin-bottom:13px;
  background:var(--card-hov);outline:none}
.dlbtn{display:flex;align-items:center;justify-content:center;gap:8px;
  width:100%;padding:11px;
  background:linear-gradient(135deg,var(--green),#00c983);
  border:none;border-radius:10px;color:#000;font-weight:700;
  font-size:.86rem;cursor:pointer;transition:all .2s;font-family:inherit}
.dlbtn:hover{transform:translateY(-1px);box-shadow:0 8px 24px rgba(0,229,160,.25)}

/* toast */
.toast{position:fixed;bottom:28px;right:28px;background:var(--card);
  border:1px solid var(--bdr);border-radius:10px;padding:13px 18px;
  font-size:.83rem;display:none;z-index:999;box-shadow:0 8px 32px rgba(0,0,0,.45)}
.toast.show{display:block;animation:tIn .28s ease}
.toast.err{border-color:#ff4444;color:#ff8888}
.toast.ok{border-color:var(--green);color:var(--green)}
@keyframes tIn{from{transform:translateY(16px);opacity:0}to{transform:none;opacity:1}}

/* ── MODE SWITCHER */
.mode-tabs{display:flex;gap:0;margin-bottom:20px;background:var(--surface);
  border:1px solid var(--bdr);border-radius:12px;padding:4px;overflow:hidden}
.mode-tab{flex:1;padding:10px 16px;border:none;border-radius:10px;cursor:pointer;
  font-size:.82rem;font-weight:600;font-family:inherit;transition:all .22s;
  background:transparent;color:var(--muted);display:flex;align-items:center;justify-content:center;gap:7px}
.mode-tab.active{background:linear-gradient(135deg,var(--purple),var(--pdark));color:#fff;
  box-shadow:0 4px 16px var(--pglow)}
.mode-tab:not(.active):hover{color:var(--text);background:var(--card)}

/* ── DUAL VOICE PANEL */
.dual-panel{display:none;flex-direction:column;gap:18px}
.dual-panel.vis{display:flex}
.voice-pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.vp-box{background:var(--surface);border:1px solid var(--bdr);border-radius:12px;padding:14px 16px}
.vp-box.male-box{border-top:2px solid #409eff;}
.vp-box.female-box{border-top:2px solid #e040fb;}
.vp-label{font-size:.68rem;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;
  margin-bottom:8px;display:flex;align-items:center;gap:7px}
.vp-label.ml{color:#409eff;}.vp-label.fl{color:#e040fb;}
.vp-sel{width:100%;background:var(--card);border:none;outline:none;
  color:var(--text);font-size:.82rem;font-family:inherit;border-radius:6px;
  padding:6px 8px;border:1px solid var(--bdr)}
.script-hint{background:linear-gradient(135deg,rgba(124,92,252,.08),rgba(224,64,251,.05));
  border:1px solid rgba(124,92,252,.25);border-radius:10px;padding:12px 15px;font-size:.78rem;line-height:1.7}
.script-hint strong{color:var(--purple)}
.hint-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}
.hint-block{background:var(--card);border-radius:8px;padding:8px 10px;font-size:.72rem;font-family:monospace;line-height:1.6;border:1px solid var(--bdr)}
.hint-block.hm{color:#409eff;border-color:rgba(64,158,255,.2)}
.hint-block.hf{color:#e040fb;border-color:rgba(224,64,251,.2)}
.seg-info{font-size:.73rem;color:var(--muted);text-align:center;margin-top:4px}
.dual-genbtn{width:100%;padding:15px;
  background:linear-gradient(135deg,#409eff,#e040fb);
  border:none;border-radius:var(--radius);color:#fff;
  font-size:.95rem;font-weight:700;cursor:pointer;
  display:flex;align-items:center;justify-content:center;gap:10px;
  transition:all .22s;font-family:inherit}
.dual-genbtn:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 12px 32px rgba(64,158,255,.3)}
.dual-genbtn:disabled{opacity:.45;cursor:not-allowed;transform:none}
</style>
</head>
<body>

<header>
  <div class="logo"><span class="dot"></span>VibeVoice Studio</div>
  <div class="hbadge">LOCAL AI</div>
</header>

<div class="wrap">
  <!-- LEFT: Voices -->
  <div>
    <div class="label">Choose Voice</div>
    <input class="vsearch" id="vsearch" placeholder="🔍 Search by name, language, style…"/>
    <div class="tabs" id="tabs">
      <button class="tab on" onclick="setFilter(this,'all')">All</button>
      <button class="tab" onclick="setFilter(this,'Male')">Male</button>
      <button class="tab" onclick="setFilter(this,'Female')">Female</button>
      <button class="tab" onclick="setFilter(this,'English')">🇬🇧 English</button>
      <button class="tab" onclick="setFilter(this,'Indian')">🇮🇳 Indian</button>
      <button class="tab" onclick="setFilter(this,'Spanish')">🇪🇸 Spanish</button>
      <button class="tab" onclick="setFilter(this,'German')">🇩🇪 German</button>
      <button class="tab" onclick="setFilter(this,'French')">🇫🇷 French</button>
    </div>
    <div class="vgrid" id="vgrid"></div>
  </div>

  <!-- RIGHT -->
  <div class="rpanel">

    <!-- Status -->
    <div class="sbar"><span class="sdot"></span><span>AI model loaded &amp; ready</span></div>

    <!-- Mode Switcher -->
    <div class="mode-tabs">
      <button class="mode-tab active" id="tab-single" onclick="switchMode('single')">🎙️ Single Voice</button>
      <button class="mode-tab" id="tab-dual" onclick="switchMode('dual')">🎭 Dual Voice Podcast</button>
    </div>

    <!-- ═══ SINGLE VOICE PANEL ═══ -->
    <div id="single-panel">

      <!-- Selected bar -->
      <div class="selbar" id="selbar" style="margin-bottom:16px">
        <div class="sbem" id="sbem">🎙️</div>
        <div class="sbinfo">
          <div class="sbname" id="sbname">—</div>
          <div class="sbsub"  id="sbsub">—</div>
        </div>
        <button class="sbuse" onclick="focusEditor()">Use This Voice ↓</button>
      </div>

      <!-- Editor -->
      <div style="margin-bottom:16px">
        <div class="label">Your Script</div>
        <div class="edwrap">
          <div class="edhead">
            <span>✍️ Paste your podcast script</span>
            <span class="ccount" id="ccount">0 chars</span>
          </div>
          <textarea id="script" placeholder="Paste your podcast / narration script here…&#10;&#10;Tip: Keep under ~10 minutes for best results." oninput="updateCount()"></textarea>
        </div>
      </div>

      <!-- Settings -->
      <div class="srow" style="margin-bottom:16px">
        <div class="sbox">
          <label>Voice</label>
          <select id="vsel"><option value="">— pick from grid →</option></select>
        </div>
        <div class="sbox">
          <label>Stability</label>
          <input type="range" min="1" max="10" value="5" id="cfg" oninput="updateCfg()"/>
          <div class="rval" id="cfgval">Balanced (1.5)</div>
        </div>
      </div>

      <!-- Generate -->
      <button class="genbtn" id="genbtn" onclick="generate()">
        <div class="gspinner"></div>
        <span id="gbtnlabel">🎧 Generate Podcast Audio</span>
      </button>
    </div>

    <!-- ═══ DUAL VOICE PANEL ═══ -->
    <div class="dual-panel" id="dual-panel">

      <!-- Script format hint -->
      <div class="script-hint">
        <strong>📋 Dual Voice Script Format</strong><br>
        Apne script mein har line pe <code>[MALE]:</code> ya <code>[FEMALE]:</code> tag lagao:
        <div class="hint-grid">
          <div class="hint-block hm">[MALE]: Hello, welcome to our podcast!</div>
          <div class="hint-block hf">[FEMALE]: Thanks for joining us today.</div>
          <div class="hint-block hm">[HOST]: Aaj ka topic kya hai?</div>
          <div class="hint-block hf">[GUEST]: Hum AI ke baare mein baat karenge.</div>
        </div>
      </div>

      <!-- Voice selectors -->
      <div class="voice-pair">
        <div class="vp-box male-box">
          <div class="vp-label ml">🔵 Male Voice</div>
          <select class="vp-sel" id="dual-male-sel">
            <option value="">— Select Male Voice —</option>
          </select>
        </div>
        <div class="vp-box female-box">
          <div class="vp-label fl">🔴 Female Voice</div>
          <select class="vp-sel" id="dual-female-sel">
            <option value="">— Select Female Voice —</option>
          </select>
        </div>
      </div>

      <!-- Dual Script Editor -->
      <div>
        <div class="label">Dual-Voice Script</div>
        <div class="edwrap">
          <div class="edhead">
            <span>✍️ Script with [MALE] &amp; [FEMALE] tags</span>
            <span class="ccount" id="dccount">0 chars</span>
          </div>
          <textarea id="dual-script"
            placeholder="[MALE]: Namaskar doston, aaj ke podcast mein aapka swagat hai!&#10;[FEMALE]: Haan bilkul, aaj hum ek bahut interesting topic discuss karenge.&#10;[MALE]: Toh shuru karte hain. Aaj ka topic hai Artificial Intelligence.&#10;[FEMALE]: AI ne hamaari zindagi kitna badal diya hai, hai na?" 
            oninput="dualScriptCount()" style="min-height:260px"></textarea>
        </div>
        <div class="seg-info" id="seg-info">— Write script above to see segment count —</div>
      </div>

      <!-- Dual Generate Button -->
      <button class="dual-genbtn" id="dual-genbtn" onclick="generateDual()">
        <div class="gspinner" id="dual-spinner"></div>
        <span id="dual-gbtnlabel">🎭 Generate Dual-Voice Podcast</span>
      </button>
    </div>

    <!-- Progress (shared) -->
    <div class="progcard" id="progcard">
      <div class="progtop">
        <div class="progtitle">⏳ Generating Audio…</div>
        <div class="progpct" id="progpct">0%</div>
      </div>
      <div class="progbar-wrap"><div class="progbar" id="progbar"></div></div>
      <div class="progmsg">
        <span id="progmsg">Starting…</span>
        <span class="elapsed" id="elapsed">0s</span>
      </div>
    </div>

    <!-- Output (shared) -->
    <div class="outcard" id="outcard">
      <div class="outtop">
        <div class="outtitle" id="out-title">🎉 Podcast Ready!</div>
        <div class="outmeta" id="outmeta">—</div>
      </div>
      <audio id="aplayer" controls></audio>
      <button class="dlbtn" onclick="dlAudio()">⬇️ Download .wav</button>
    </div>

  </div>
</div>

<div class="toast" id="toast"></div>

<script>
let voices=[], selId=null, curPrevAudio=null, curPrevBtn=null;
let filter='all', search='', curJobId=null, pollTimer=null, elapsedTimer=null;
let elapsedSec=0, genFile=null;

const CFG_MAP={1:1.0,2:1.1,3:1.2,4:1.3,5:1.5,6:1.7,7:1.9,8:2.1,9:2.3,10:2.5};
const CFG_LBL={1:'Very Stable (1.0)',2:'Stable (1.1)',3:'Natural (1.2)',4:'Balanced (1.3)',
  5:'Balanced (1.5)',6:'Expressive (1.7)',7:'Expressive (1.9)',8:'Dynamic (2.1)',
  9:'Dynamic (2.3)',10:'Very Dynamic (2.5)'};

/* ── Load voices */
async function load(){
  const r=await fetch('/api/voices');
  voices=await r.json();
  render();
  // Single voice selector
  const sel=document.getElementById('vsel');
  voices.forEach(v=>{const o=document.createElement('option');o.value=v.id;
    o.textContent=`${v.emoji} ${v.label} (${v.lang})`;sel.appendChild(o);});
  sel.onchange=()=>{const v=voices.find(x=>x.id===sel.value);if(v)pick(v.id);};
  // Dual voice selectors
  const msel=document.getElementById('dual-male-sel');
  const fsel=document.getElementById('dual-female-sel');
  voices.filter(v=>v.gender==='Male').forEach(v=>{
    const o=document.createElement('option');o.value=v.id;
    o.textContent=`${v.emoji} ${v.label} (${v.lang})`;msel.appendChild(o);
  });
  voices.filter(v=>v.gender==='Female').forEach(v=>{
    const o=document.createElement('option');o.value=v.id;
    o.textContent=`${v.emoji} ${v.label} (${v.lang})`;fsel.appendChild(o);
  });
  // Auto-select first available
  if(msel.options.length>1) msel.value=msel.options[1].value;
  if(fsel.options.length>1) fsel.value=fsel.options[1].value;
}

/* ── Render grid */
function render(){
  const f=voices.filter(v=>{
    const mf=filter==='all'||v.gender===filter||v.lang.toLowerCase().includes(filter.toLowerCase());
    const ms=!search||v.label.toLowerCase().includes(search)||v.lang.toLowerCase().includes(search)||v.style.toLowerCase().includes(search);
    return mf&&ms;
  });
  document.getElementById('vgrid').innerHTML=f.map(v=>`
  <div class="vcard${selId===v.id?' sel':''}" id="vc-${v.id}" onclick="pick('${v.id}')">
    <div class="chk">✓</div>
    <div class="vtop">
      <div class="vav">${v.emoji}</div>
      <span class="gbadge ${v.gender.toLowerCase()}">${v.gender}</span>
    </div>
    <div class="vname">${v.label}</div>
    <div class="vlang">${v.lang}</div>
    <div class="vstyle">${v.style}</div>
    <button class="prevbtn" id="pb-${v.id}" onclick="prevVoice(event,'${v.id}')">
      <span class="pspinner"></span>
      <span class="pbtxt">▶ Preview Voice</span>
    </button>
  </div>`).join('');
}

/* ── Filter */
function setFilter(el,val){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  el.classList.add('on');filter=val;render();
}
document.getElementById('vsearch').oninput=function(){search=this.value.toLowerCase();render();};

/* ── Pick voice */
function pick(id){
  selId=id;
  document.getElementById('vsel').value=id;
  const v=voices.find(x=>x.id===id);
  if(v){
    document.getElementById('selbar').classList.add('vis');
    document.getElementById('sbem').textContent=v.emoji;
    document.getElementById('sbname').textContent=v.label;
    document.getElementById('sbsub').textContent=`${v.lang} · ${v.gender} · ${v.style}`;
  }
  render();
}

/* ── Preview voice */
async function prevVoice(e,id){
  e.stopPropagation();
  const btn=document.getElementById(`pb-${id}`);
  const txt=btn.querySelector('.pbtxt');

  // Stop currently playing
  if(curPrevAudio){
    curPrevAudio.pause();curPrevAudio=null;
    if(curPrevBtn){curPrevBtn.classList.remove('playing','loading');
      curPrevBtn.querySelector('.pspinner').style.display='none';
      curPrevBtn.querySelector('.pbtxt').textContent='▶ Preview Voice';}
    if(curPrevBtn===btn){curPrevBtn=null;return;}
  }

  // Try to get cached audio first
  const cacheCheck=await fetch(`/api/preview_status/${id}`);
  const status=await cacheCheck.json();

  if(status.ready){
    // Play immediately
    playPreview(id,btn);
    return;
  }

  // Not ready — kick off generation
  btn.classList.add('loading');
  btn.querySelector('.pspinner').style.display='inline-block';
  txt.textContent='Generating preview…';
  curPrevBtn=btn;

  // Start generation
  await fetch(`/api/preview/${id}`);

  // Poll until ready
  const poll=setInterval(async()=>{
    const chk=await fetch(`/api/preview_status/${id}`);
    const st=await chk.json();
    if(st.ready){
      clearInterval(poll);
      btn.classList.remove('loading');
      btn.querySelector('.pspinner').style.display='none';
      txt.textContent='▶ Preview Voice';
      playPreview(id,btn);
    }
  },2000);
}

function playPreview(id,btn){
  const txt=btn.querySelector('.pbtxt');
  const audio=new Audio(`/api/preview_file/${id}`);
  curPrevAudio=audio;curPrevBtn=btn;
  btn.classList.add('playing');
  txt.textContent='⏸ Playing…';
  audio.oncanplay=()=>audio.play();
  audio.onended=()=>{
    btn.classList.remove('playing');
    txt.textContent='▶ Preview Voice';
    curPrevAudio=null;curPrevBtn=null;
  };
  audio.onerror=()=>{
    btn.classList.remove('playing');
    txt.textContent='▶ Preview Voice';
    toast('Preview failed',true);
    curPrevAudio=null;curPrevBtn=null;
  };
}

/* ── Settings */
function updateCfg(){
  const v=document.getElementById('cfg').value;
  document.getElementById('cfgval').textContent=CFG_LBL[v]||'Balanced';
}
function updateCount(){
  const l=document.getElementById('script').value.length;
  document.getElementById('ccount').textContent=`${l.toLocaleString()} chars`;
}
function focusEditor(){
  document.getElementById('script').scrollIntoView({behavior:'smooth'});
  document.getElementById('script').focus();
}

/* ── Mode switch */
function switchMode(mode){
  document.getElementById('tab-single').classList.toggle('active', mode==='single');
  document.getElementById('tab-dual').classList.toggle('active', mode==='dual');
  document.getElementById('single-panel').style.display=(mode==='single')?'':'none';
  document.getElementById('dual-panel').classList.toggle('vis', mode==='dual');
  document.getElementById('outcard').classList.remove('vis');
  document.getElementById('progcard').classList.remove('vis');
}

/* ── Dual script count + segment preview */
function dualScriptCount(){
  const txt=document.getElementById('dual-script').value;
  document.getElementById('dccount').textContent=`${txt.length.toLocaleString()} chars`;
  // Count segments
  const lines=txt.split('\n').filter(l=>/^\[(MALE|FEMALE|HOST|GUEST)\]/i.test(l.trim())||l.trim().length>0);
  const tagged=txt.split('\n').filter(l=>/^\[(MALE|FEMALE|HOST|GUEST)\]/i.test(l.trim()));
  const males=txt.split('\n').filter(l=>/^\[(MALE|HOST)\]/i.test(l.trim()));
  const females=txt.split('\n').filter(l=>/^\[(FEMALE|GUEST)\]/i.test(l.trim()));
  if(tagged.length>0){
    document.getElementById('seg-info').textContent=
      `🔵 ${males.length} male segments · 🔴 ${females.length} female segments · ${tagged.length} total`;
  } else {
    document.getElementById('seg-info').textContent='— Add [MALE]: or [FEMALE]: tags to your lines —';
  }
}

/* ── Generate */
async function generate(){
  const script=document.getElementById('script').value.trim();
  const vid=document.getElementById('vsel').value||selId;
  if(!script){toast('Please enter a script first!',true);return;}
  if(!vid){toast('Please select a voice!',true);return;}

  // Reset UI
  document.getElementById('outcard').classList.remove('vis');
  document.getElementById('progcard').classList.add('vis');
  setProgress(0,'Starting generation…');
  elapsedSec=0;
  document.getElementById('elapsed').textContent='0s';

  const btn=document.getElementById('genbtn');
  btn.disabled=true;btn.classList.add('ld');
  document.getElementById('gbtnlabel').textContent='Generating…';

  // Start elapsed timer
  clearInterval(elapsedTimer);
  elapsedTimer=setInterval(()=>{
    elapsedSec++;
    document.getElementById('elapsed').textContent=elapsedSec+'s';
  },1000);

  try{
    const r=await fetch('/api/generate',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({script,voice_id:vid})
    });
    const d=await r.json();
    if(d.error){throw new Error(d.error);}
    curJobId=d.job_id;
    pollProgress(d.job_id);
  }catch(e){
    endGen();
    toast('Error: '+e.message,true);
  }
}

/* ── Poll progress */
function pollProgress(jid){
  clearInterval(pollTimer);
  pollTimer=setInterval(async()=>{
    try{
      const r=await fetch(`/api/job/${jid}`);
      const d=await r.json();
      if(d.status==='running'||d.status==='queued'){
        setProgress(d.progress||0, d.msg||'Processing…');
      } else if(d.status==='done'){
        clearInterval(pollTimer);
        setProgress(100,'Complete!');
        setTimeout(()=>showOutput(d.file,d.elapsed||elapsedSec),400);
      } else if(d.status==='error'){
        clearInterval(pollTimer);
        endGen();
        toast('Error: '+(d.error||'unknown'),true);
      }
    }catch(e){}
  },800);
}

function setProgress(pct,msg){
  document.getElementById('progpct').textContent=pct+'%';
  document.getElementById('progbar').style.width=pct+'%';
  document.getElementById('progmsg').textContent=msg;
}

function showOutput(file,secs,segments){
  genFile=file;
  document.getElementById('aplayer').src=`/api/download/${file}`;
  let meta=`Generated in ${secs}s`;
  if(segments) meta+=` · ${segments} segments merged`;
  document.getElementById('outmeta').textContent=meta;
  document.getElementById('outcard').classList.add('vis');
  document.getElementById('outcard').scrollIntoView({behavior:'smooth'});
  toast('✅ Podcast ready!',false);
  endGen();
}

function endGen(){
  clearInterval(elapsedTimer);clearInterval(pollTimer);
  // Reset single voice btn
  const btn=document.getElementById('genbtn');
  btn.disabled=false;btn.classList.remove('ld');
  document.getElementById('gbtnlabel').textContent='🎧 Generate Podcast Audio';
  // Reset dual voice btn
  const dbtn=document.getElementById('dual-genbtn');
  dbtn.disabled=false;
  document.getElementById('dual-spinner').style.display='none';
  document.getElementById('dual-gbtnlabel').textContent='🎭 Generate Dual-Voice Podcast';
}

function dlAudio(){
  if(!genFile)return;
  const a=document.createElement('a');a.href=`/api/download/${genFile}`;
  a.download=genFile;a.click();
}

function toast(msg,err){
  const t=document.getElementById('toast');
  t.textContent=msg;t.className='toast show '+(err?'err':'ok');
  setTimeout(()=>t.classList.remove('show'),3500);
}

load();

/* ── Dual Voice Generate */
async function generateDual(){
  const script=document.getElementById('dual-script').value.trim();
  const maleId=document.getElementById('dual-male-sel').value;
  const femaleId=document.getElementById('dual-female-sel').value;
  if(!script){toast('Please enter a dual-voice script!',true);return;}
  if(!maleId){toast('Please select a MALE voice!',true);return;}
  if(!femaleId){toast('Please select a FEMALE voice!',true);return;}

  // Reset UI
  document.getElementById('outcard').classList.remove('vis');
  document.getElementById('progcard').classList.add('vis');
  setProgress(2,'Parsing script…');
  elapsedSec=0;
  document.getElementById('elapsed').textContent='0s';
  document.getElementById('out-title').textContent='🎉 Dual Podcast Ready!';

  const dbtn=document.getElementById('dual-genbtn');
  dbtn.disabled=true;
  document.getElementById('dual-spinner').style.display='inline-block';
  document.getElementById('dual-gbtnlabel').textContent='Generating…';

  clearInterval(elapsedTimer);
  elapsedTimer=setInterval(()=>{elapsedSec++;
    document.getElementById('elapsed').textContent=elapsedSec+'s';},1000);

  try{
    const r=await fetch('/api/generate_dual',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({script,male_voice_id:maleId,female_voice_id:femaleId})
    });
    const d=await r.json();
    if(d.error){throw new Error(d.error);}
    curJobId=d.job_id;
    const totalSegs=d.total_segments||0;
    pollProgressDual(d.job_id, totalSegs);
  }catch(e){
    endGen();
    toast('Error: '+e.message,true);
  }
}

function pollProgressDual(jid, totalSegs){
  clearInterval(pollTimer);
  pollTimer=setInterval(async()=>{
    try{
      const r=await fetch(`/api/job/${jid}`);
      const d=await r.json();
      if(d.status==='running'||d.status==='queued'){
        setProgress(d.progress||0, d.msg||'Processing…');
      } else if(d.status==='done'){
        clearInterval(pollTimer);
        setProgress(100,'Complete!');
        setTimeout(()=>showOutput(d.file,d.elapsed||elapsedSec,totalSegs),400);
      } else if(d.status==='error'){
        clearInterval(pollTimer);
        endGen();
        toast('Error: '+(d.error||'unknown'),true);
      }
    }catch(e){}
  },800);
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    print("\n🌐 VibeVoice Studio → http://127.0.0.1:5050\n")
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
