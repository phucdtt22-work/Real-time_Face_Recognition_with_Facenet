# Real-time Face Recognition with FaceNet

This project provides a real-time face recognition application using FaceNet embeddings. It includes registration and streaming apps organized in a clean folder structure.

## Folder Structure

- `src/` – Application sources (`StreamingApp.py`, `Register.py`)
- `scripts/` – Utility scripts (add future helpers here)
- `models/` – Model files (pretrained FaceNet, optional)
- `data/` – Data artifacts (embeddings, label maps)
- `assets/` – Sample images/videos
- `docs/` – Documentation and notes

## Quick Start

### 1) Environment

Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Register Faces

Run the registration app to capture faces and store embeddings:

```bash
python src/Register.py --output data/embeddings.npy --labels data/labels.json
```

### 3) Run Streaming Recognition

Start the streaming application and perform real-time recognition:

```bash
python src/StreamingApp.py --embeddings data/embeddings.npy --labels data/labels.json
```

Adjust camera index or input source in the scripts as needed.

## Notes

- Large datasets and raw video files should be kept outside the repo.
- Place FaceNet model files in `models/` if required and update paths.

## License

Internal use only. Ensure rights to datasets and model weights.
