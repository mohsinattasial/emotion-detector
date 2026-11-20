# Facial Expression Model Demo — Streamlit

A Streamlit web app that loads the pretrained FER+ ONNX classifier through OpenCV and estimates expression categories on detected faces.

## Run
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The first run may download model files; internet connectivity is needed. Application logic and model asset URLs are in `app.py`.

**Important:** Facial expressions do not reliably reveal true feelings or intentions. Results can be wrong or biased; do not use them to make consequential decisions.

Files: `app.py`, `requirements.txt`, `README.md`.

Author: Mohsin Atta
