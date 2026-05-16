"""Tkinter UI for handwriting verification inference."""

import pickle
import sys
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image
from tkinter import (
    Tk,
    StringVar,
    messagebox,
    filedialog,
    ttk,
)

_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))

from preproc_pipeline import preproc_image
from mk_vectors_v6 import load_model, extract_features, preprocess, FEATURE_DIM

BASE_DIR = _SRC_DIR.parent
MODELS_DIR = BASE_DIR / "Trained_Models"

_alexnet_model = None


def _get_alexnet():
    global _alexnet_model
    if _alexnet_model is None:
        _alexnet_model = load_model()
    return _alexnet_model


def list_trained_persons():
    """Return sorted person names from Trained_Models/*.pkl stems."""
    if not MODELS_DIR.is_dir():
        return []
    return sorted(p.stem for p in MODELS_DIR.glob("*.pkl"))


def load_person_model(person_name: str):
    """Load the sklearn pipeline for a person."""
    pkl_path = MODELS_DIR / f"{person_name}.pkl"
    if not pkl_path.is_file():
        raise FileNotFoundError(f"No trained model found for '{person_name}'.")
    with open(pkl_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"]


def segment_to_features(alexnet, segment_np):
    """Convert a grayscale segment patch to a 6096-d feature vector."""
    if segment_np.ndim == 2:
        rgb = cv2.cvtColor(segment_np, cv2.COLOR_GRAY2RGB)
    else:
        rgb = segment_np
    pil_img = Image.fromarray(rgb)
    tensor = preprocess(pil_img)
    return extract_features(alexnet, tensor)


def features_to_dataframe(feat_vector):
    """Build a one-row DataFrame matching training feature columns."""
    columns = [f"feat_{i}" for i in range(FEATURE_DIM)]
    return pd.DataFrame([feat_vector], columns=columns)


def run_inference(image_path: str, person_name: str) -> dict:
    """
    Run the full inference pipeline on an image for a given person.

    Returns a dict with final decision, confidence, and per-segment details.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    segments = preproc_image(img)
    if not segments:
        raise ValueError("No handwriting segments detected in the image.")

    pipeline = load_person_model(person_name)
    alexnet = _get_alexnet()

    segment_predictions = []
    for _, _, seg in segments:
        feats = segment_to_features(alexnet, seg)
        features_df = features_to_dataframe(feats)
        pred = int(pipeline.predict(features_df)[0])
        segment_predictions.append(pred)

    vote_counts = Counter(segment_predictions)
    final_label, match_votes = vote_counts.most_common(1)[0]
    no_match_votes = vote_counts.get(1 - final_label, 0)
    total = len(segment_predictions)
    confidence = (match_votes / total) * 100

    if final_label == 1:
        final_decision = f"Match — handwriting appears to be {person_name}"
    else:
        final_decision = f"No match — handwriting does not appear to be {person_name}"

    return {
        "person": person_name,
        "num_segments": total,
        "segment_predictions": segment_predictions,
        "final_label": final_label,
        "final_decision": final_decision,
        "confidence": confidence,
        "match_votes": vote_counts.get(1, 0),
        "no_match_votes": vote_counts.get(0, 0),
    }


class ModelOutputApp:
    """Tkinter application for handwriting verification."""

    def __init__(self):
        self.root = Tk()
        self.root.title("Biometrics ML — Handwriting Verification")
        self.root.minsize(520, 320)

        self.person_var = StringVar()
        self.image_path_var = StringVar()
        self.status_var = StringVar(value="Select a person and an image, then click Analyze.")

        self._build_ui()
        self._refresh_persons()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Person:").grid(row=0, column=0, sticky="w", **pad)
        self.person_combo = ttk.Combobox(
            main, textvariable=self.person_var, state="readonly", width=40
        )
        self.person_combo.grid(row=0, column=1, columnspan=2, sticky="ew", **pad)

        ttk.Label(main, text="Image:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(main, textvariable=self.image_path_var, width=42).grid(
            row=1, column=1, sticky="ew", **pad
        )
        ttk.Button(main, text="Browse…", command=self._browse_image).grid(
            row=1, column=2, **pad
        )

        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=8)
        self.analyze_btn = ttk.Button(btn_frame, text="Analyze", command=self._analyze)
        self.analyze_btn.pack()

        ttk.Separator(main, orient="horizontal").grid(
            row=3, column=0, columnspan=3, sticky="ew", pady=8
        )

        ttk.Label(main, text="Result:").grid(row=4, column=0, sticky="nw", **pad)
        self.result_text = ttk.Label(
            main, text="—", wraplength=400, justify="left"
        )
        self.result_text.grid(row=4, column=1, columnspan=2, sticky="w", **pad)

        ttk.Label(main, textvariable=self.status_var, foreground="gray").grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(12, 0)
        )

        main.columnconfigure(1, weight=1)

    def _refresh_persons(self):
        persons = list_trained_persons()
        self.person_combo["values"] = persons
        if persons:
            self.person_var.set(persons[0])

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select handwriting image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.image_path_var.set(path)

    def _analyze(self):
        person = self.person_var.get().strip()
        image_path = self.image_path_var.get().strip()

        if not person:
            messagebox.showerror("Error", "Please select a person.")
            return
        if not image_path:
            messagebox.showerror("Error", "Please select an image file.")
            return

        self.analyze_btn.config(state="disabled")
        self.status_var.set("Processing… (AlexNet may load on first run)")
        self.root.config(cursor="watch")
        self.root.update()

        try:
            result = run_inference(image_path, person)
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Analysis failed.")
            return
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Analysis failed.")
            return
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {e}")
            self.status_var.set("Analysis failed.")
            return
        finally:
            self.analyze_btn.config(state="normal")
            self.root.config(cursor="")
            self.root.update()

        self.result_text.config(text=result["final_decision"])
        self.status_var.set(
            f"Confidence: {result['confidence']:.1f}%  |  "
            f"Segments: {result['num_segments']}  |  "
            f"Match votes: {result['match_votes']}  |  "
            f"No-match votes: {result['no_match_votes']}"
        )

    def mainloop(self):
        self.root.mainloop()


if __name__ == "__main__":
    ModelOutputApp().mainloop()
