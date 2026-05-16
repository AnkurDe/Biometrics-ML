"""Flask web UI for handwriting verification inference."""

import sys
import tempfile
from pathlib import Path

from flask import Flask, render_template_string, request
from werkzeug.utils import secure_filename

_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))

from model_output import list_trained_persons, run_inference

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Biometrics ML — Handwriting Verification</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      max-width: 640px;
      margin: 2rem auto;
      padding: 0 1rem;
      color: #1a1a1a;
      line-height: 1.5;
    }
    h1 { font-size: 1.35rem; margin-bottom: 0.25rem; }
    .subtitle { color: #555; font-size: 0.9rem; margin-bottom: 1.5rem; }
    label { display: block; font-weight: 600; margin-top: 1rem; margin-bottom: 0.35rem; }
    select, input[type="file"] { width: 100%; padding: 0.5rem; font-size: 1rem; }
    .source-tabs {
      display: flex;
      gap: 0.5rem;
      margin-top: 0.5rem;
    }
    .source-tabs button {
      margin-top: 0;
      flex: 1;
      background: #e5e7eb;
      color: #374151;
    }
    .source-tabs button.active {
      background: #2563eb;
      color: #fff;
    }
    .source-panel { display: none; margin-top: 0.5rem; }
    .source-panel.active { display: block; }
    #camera-video, #capture-preview {
      width: 100%;
      max-height: 320px;
      border-radius: 6px;
      background: #111;
      object-fit: contain;
    }
    #capture-preview { display: none; margin-top: 0.5rem; }
    .camera-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 0.75rem;
    }
    .camera-actions button { margin-top: 0; }
    .camera-actions button.secondary {
      background: #6b7280;
    }
    .camera-actions button.secondary:hover { background: #4b5563; }
    .camera-hint { color: #666; font-size: 0.85rem; margin-top: 0.5rem; }
    button {
      margin-top: 1.25rem;
      padding: 0.6rem 1.25rem;
      font-size: 1rem;
      background: #2563eb;
      color: #fff;
      border: none;
      border-radius: 6px;
      cursor: pointer;
    }
    button:hover { background: #1d4ed8; }
    .error {
      margin-top: 1.25rem;
      padding: 0.75rem 1rem;
      background: #fef2f2;
      border: 1px solid #fecaca;
      border-radius: 6px;
      color: #991b1b;
    }
    .result {
      margin-top: 1.25rem;
      padding: 1rem;
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      border-radius: 6px;
    }
    .result.no-match {
      background: #fff7ed;
      border-color: #fed7aa;
    }
    .result h2 { font-size: 1.1rem; margin: 0 0 0.5rem; }
    .meta { color: #444; font-size: 0.9rem; }
  </style>
</head>
<body>
  <h1>Handwriting Verification</h1>
  <p class="subtitle">Upload or capture a handwriting image, then select a person to verify against.</p>

  <form id="analyze-form" method="post" enctype="multipart/form-data">
    <label for="person">Person</label>
    <select id="person" name="person" required>
      <option value="" disabled {% if not selected_person %}selected{% endif %}>— Select —</option>
      {% for name in persons %}
      <option value="{{ name }}" {% if name == selected_person %}selected{% endif %}>{{ name }}</option>
      {% endfor %}
    </select>

    <label>Handwriting image</label>
    <div class="source-tabs">
      <button type="button" class="tab active" data-panel="upload">Upload file</button>
      <button type="button" class="tab" data-panel="camera">Use camera</button>
    </div>

    <div id="panel-upload" class="source-panel active">
      <input id="image" type="file" name="image" accept="image/*,.png,.jpg,.jpeg,.bmp,.webp,.tiff">
    </div>

    <div id="panel-camera" class="source-panel">
      <video id="camera-video" autoplay playsinline muted></video>
      <canvas id="camera-canvas" hidden></canvas>
      <img id="capture-preview" alt="Captured preview">
      <p class="camera-hint" id="camera-hint">Allow camera access, then capture a clear photo of the handwriting.</p>
      <div class="camera-actions">
        <button type="button" id="btn-start-camera">Start camera</button>
        <button type="button" id="btn-capture" disabled>Capture photo</button>
        <button type="button" id="btn-retake" class="secondary" disabled>Retake</button>
        <button type="button" id="btn-stop-camera" class="secondary" disabled>Stop camera</button>
      </div>
    </div>

    <button type="submit">Analyze</button>
  </form>

  <script>
  (function () {
    const form = document.getElementById("analyze-form");
    const fileInput = document.getElementById("image");
    const tabs = document.querySelectorAll(".source-tabs .tab");
    const panelUpload = document.getElementById("panel-upload");
    const panelCamera = document.getElementById("panel-camera");
    const video = document.getElementById("camera-video");
    const canvas = document.getElementById("camera-canvas");
    const preview = document.getElementById("capture-preview");
    const hint = document.getElementById("camera-hint");
    const btnStart = document.getElementById("btn-start-camera");
    const btnCapture = document.getElementById("btn-capture");
    const btnRetake = document.getElementById("btn-retake");
    const btnStop = document.getElementById("btn-stop-camera");

    let stream = null;
    let activeSource = "upload";
    let hasCapture = false;

    function setFileOnInput(file) {
      const dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
    }

    function clearFileInput() {
      fileInput.value = "";
      hasCapture = false;
    }

    async function stopCamera() {
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
        stream = null;
      }
      video.srcObject = null;
      video.style.display = "block";
      preview.style.display = "none";
      btnCapture.disabled = true;
      btnStop.disabled = true;
      btnStart.disabled = false;
    }

    async function startCamera() {
      hint.textContent = "Requesting camera access…";
      try {
        await stopCamera();
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" } },
          audio: false,
        });
        video.srcObject = stream;
        hint.textContent = "Position the handwriting in frame, then tap Capture photo.";
        btnCapture.disabled = false;
        btnStop.disabled = false;
        btnStart.disabled = true;
        btnRetake.disabled = true;
        hasCapture = false;
        clearFileInput();
      } catch (err) {
        hint.textContent = "Could not access camera: " + err.message;
      }
    }

    function capturePhoto() {
      if (!stream) return;
      const w = video.videoWidth;
      const h = video.videoHeight;
      if (!w || !h) {
        hint.textContent = "Camera not ready yet. Wait a moment and try again.";
        return;
      }
      canvas.width = w;
      canvas.height = h;
      canvas.getContext("2d").drawImage(video, 0, 0, w, h);
      canvas.toBlob(function (blob) {
        if (!blob) return;
        const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
        setFileOnInput(file);
        hasCapture = true;
        preview.src = URL.createObjectURL(blob);
        preview.style.display = "block";
        video.style.display = "none";
        hint.textContent = "Photo captured. Submit Analyze or Retake to capture again.";
        btnRetake.disabled = false;
        btnCapture.disabled = true;
      }, "image/jpeg", 0.92);
    }

    function retake() {
      preview.style.display = "none";
      video.style.display = "block";
      clearFileInput();
      btnRetake.disabled = true;
      if (stream) btnCapture.disabled = false;
      hint.textContent = "Position the handwriting in frame, then tap Capture photo.";
    }

    tabs.forEach((tab) => {
      tab.addEventListener("click", function () {
        const panel = tab.dataset.panel;
        activeSource = panel;
        tabs.forEach((t) => t.classList.toggle("active", t === tab));
        panelUpload.classList.toggle("active", panel === "upload");
        panelCamera.classList.toggle("active", panel === "camera");
        if (panel === "upload") {
          stopCamera();
        } else {
          clearFileInput();
        }
      });
    });

    btnStart.addEventListener("click", startCamera);
    btnCapture.addEventListener("click", capturePhoto);
    btnRetake.addEventListener("click", retake);
    btnStop.addEventListener("click", stopCamera);

    form.addEventListener("submit", function (e) {
      if (!fileInput.files || fileInput.files.length === 0) {
        e.preventDefault();
        if (activeSource === "camera") {
          alert("Please start the camera and capture a photo before analyzing.");
        } else {
          alert("Please choose an image file to upload.");
        }
      }
    });

    window.addEventListener("beforeunload", stopCamera);
  })();
  </script>

  {% if error %}
  <div class="error">{{ error }}</div>
  {% endif %}

  {% if result %}
  <div class="result {% if result.final_label == 0 %}no-match{% endif %}">
    <h2>{{ result.final_decision }}</h2>
    <p class="meta">
      Confidence: {{ "%.1f"|format(result.confidence) }}%<br>
      Segments: {{ result.num_segments }}<br>
      Match votes: {{ result.match_votes }} · No-match votes: {{ result.no_match_votes }}
    </p>
  </div>
  {% endif %}
</body>
</html>
"""


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    persons = list_trained_persons()
    error = None
    result = None
    selected_person = request.form.get("person", "")

    if request.method == "POST":
        person = selected_person.strip()
        file = request.files.get("image")

        if not person:
            error = "Please select a person."
        elif not file or (not file.filename and not file.content_length):
            error = "Please upload or capture an image."
        else:
            filename = secure_filename(file.filename) if file.filename else "capture.jpg"
            if not _allowed_file(filename):
                error = "Unsupported file type. Use PNG, JPG, JPEG, or BMP."
            else:
                suffix = Path(filename).suffix or ".jpg"
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp_path = tmp.name
                        file.save(tmp_path)
                    result = run_inference(tmp_path, person)
                except FileNotFoundError as e:
                    error = str(e)
                except ValueError as e:
                    error = str(e)
                except Exception as e:
                    error = f"Unexpected error: {e}"
                finally:
                    if tmp_path and Path(tmp_path).exists():
                        Path(tmp_path).unlink()

    return render_template_string(
        INDEX_HTML,
        persons=persons,
        selected_person=selected_person,
        error=error,
        result=result,
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
