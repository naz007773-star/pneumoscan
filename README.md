# PneumoScan: chest X-ray pneumonia analysis (educational prototype)

Full-stack final-year project: FastAPI backend, PyTorch classifier, lavender web UI with a **user portal** (`/`) and a **separate admin portal** (`/admin`).

> Not a medical device. Output is experimental, may be wrong, and must never replace a clinician.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```
Open http://127.0.0.1:8000 (users) and http://127.0.0.1:8000/admin (admin).

First run creates an admin account: **admin@pneumoscan.local / Admin@12345**. Change it by setting
`ADMIN_EMAIL` and `ADMIN_PASSWORD` *before the first run* (or delete `data/app.db` and restart), then keep the password out of source control.

## Demo mode vs real predictions
Without `models/model.pt` the app runs in **demo mode**: results are placeholders and a banner says so. To get real predictions, train a model.

## Train the model
1. Download the Kermany pediatric chest X-ray dataset (Mendeley Data `rscbjbr9sj`, or the Kaggle `chest-xray-pneumonia` copy). Check its licence and cite it.
2. Arrange as `DATA/train/<CLASS>/*.jpeg` and `DATA/test/<CLASS>/*.jpeg`. The Kaggle copy has `NORMAL` and `PNEUMONIA` only, so you get a binary model. Use `BACTERIA`/`VIRUS` folders only if you've verified the labels.
3. Run:
```bash
python train.py --data path/to/chest_xray --arch densenet121 --epochs 8
```
This does class-weighted training, picks the best epoch by balanced accuracy, calibrates with temperature scaling, evaluates on the untouched test set, and writes `models/model.pt` (metrics included, shown in the Model pages). Compare `--arch resnet18 | densenet121 | efficientnet_b0` and report all three. Restart the server afterwards.

## Features
**User portal:** register/login, dashboard stats, image upload with preview and drag-and-drop, video upload (frame sampling, X-ray gate, near-duplicate removal, documented aggregation), "Needs review" state below the confidence threshold, searchable history, delete, model info, profile and password change, responsive drawer sidebar.

**Admin portal (separate login, admin accounts cannot use the user portal):** overview stats and 14-day activity chart, user management (activate/deactivate, promote/demote, delete with cascade), all-analyses metadata with search and delete, model metrics and confusion matrix. Admins cannot view patient images by design.

**Security:** scrypt password hashing, random server-side session tokens (hashed in DB, 12 h expiry), login throttling, file type/size checks, images re-encoded as JPEG (strips EXIF), per-user record access, no secrets in code.

## Configuration (environment variables)
`ADMIN_EMAIL`, `ADMIN_PASSWORD`, `REVIEW_THRESHOLD` (default 0.75), `FRAME_EVERY_SEC` (1.0), `MAX_FRAMES` (12).

## Known limitations (put these in your report)
- Pediatric, single-hospital training data; unknown behaviour on adults or other scanners.
- The "looks like an X-ray" gate uses simple colour/contrast statistics, not a validated OOD detector.
- A random validation split without patient IDs can leak patients; the held-out test set is the number to trust.
- Viral vs bacterial from X-ray alone is unreliable; video classification is unvalidated.
- No Grad-CAM yet (listed as future work; heatmaps need validation before display).
- For deployment add HTTPS, a production DB, stronger rate limiting, and an ethics/privacy review.

## Structure
```
backend/main.py       API, auth, admin, video pipeline
backend/inference.py  model loading, preprocessing, X-ray gate
train.py              training, calibration, evaluation
static/               index.html (login), app.html (user), admin.html, style.css, common.js
models/model.pt       created by train.py
data/                 SQLite DB + stored images (created at runtime)
```
