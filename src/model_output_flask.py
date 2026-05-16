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
  <p class="subtitle">Upload a handwriting image and select a person to verify against.</p>

  <form method="post" enctype="multipart/form-data">
    <label for="person">Person</label>
    <select id="person" name="person" required>
      <option value="" disabled {% if not selected_person %}selected{% endif %}>— Select —</option>
      {% for name in persons %}
      <option value="{{ name }}" {% if name == selected_person %}selected{% endif %}>{{ name }}</option>
      {% endfor %}
    </select>

    <label for="image">Handwriting image</label>
    <input id="image" type="file" name="image" accept=".png,.jpg,.jpeg,.bmp,.webp,.tiff" required>

    <button type="submit">Analyze</button>
  </form>

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
        elif not file or not file.filename:
            error = "Please upload an image file."
        elif not _allowed_file(file.filename):
            error = "Unsupported file type. Use PNG, JPG, JPEG, or BMP."
        else:
            suffix = Path(secure_filename(file.filename)).suffix or ".png"
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
