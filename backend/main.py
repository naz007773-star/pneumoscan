"""PneumoScan API: user portal (/) and separate admin portal (/admin)."""
import io, json, re, secrets, sqlite3, tempfile, hashlib, hmac, os, time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from .inference import REVIEW_THRESHOLD, classifier, looks_like_xray

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
STATIC = ROOT / "static"
DATA.mkdir(exist_ok=True); UPLOADS.mkdir(exist_ok=True)
DB_PATH = DATA / "app.db"

MAX_IMG = 10 * 1024 * 1024
MAX_VIDEO = 60 * 1024 * 1024
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
FRAME_EVERY_SEC = float(os.getenv("FRAME_EVERY_SEC", "1.0"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "12"))
SESSION_HOURS = 12
Image.MAX_IMAGE_PIXELS = 40_000_000
DISCLAIMER = "Educational prototype. Not a medical diagnosis. Must be reviewed by a qualified professional."

app = FastAPI(title="PneumoScan", docs_url=None, redoc_url=None)


# ---------------- database ----------------
def q(sql, args=(), one=False, commit=False):
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    try:
        cur = c.execute(sql, args)
        if commit:
            c.commit()
            return cur.lastrowid
        rows = cur.fetchall()
        return (rows[0] if rows else None) if one else rows
    finally:
        c.close()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def hash_pw(pw, salt=None):
    salt = salt or secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return salt.hex() + "$" + h.hex()


def check_pw(pw, stored):
    s, h = stored.split("$")
    t = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(s), n=2**14, r=8, p=1, dklen=32).hex()
    return hmac.compare_digest(t, h)


def init_db():
    q("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
        pw TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'user', active INTEGER NOT NULL DEFAULT 1,
        created TEXT NOT NULL)""", commit=True)
    q("""CREATE TABLE IF NOT EXISTS sessions(
        token TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        expires TEXT NOT NULL)""", commit=True)
    q("""CREATE TABLE IF NOT EXISTS analyses(
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        kind TEXT NOT NULL, filename TEXT, stored TEXT, label TEXT, confidence REAL, status TEXT,
        probs TEXT, details TEXT, model_version TEXT, created TEXT NOT NULL)""", commit=True)
    email = os.getenv("ADMIN_EMAIL", "admin@pneumoscan.local").lower()
    if not q("SELECT 1 FROM users WHERE role='admin'", one=True):
        pw = os.getenv("ADMIN_PASSWORD", "Admin@12345")
        q("INSERT INTO users(name,email,pw,role,created) VALUES(?,?,?,?,?)",
          ("Administrator", email, hash_pw(pw), "admin", now()), commit=True)
        print(f"[SETUP] Admin created -> {email} / {pw}   (CHANGE THIS PASSWORD)")


init_db()


# ---------------- auth ----------------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
failed = defaultdict(list)


class Creds(BaseModel):
    email: str
    password: str
    name: Optional[str] = None


class ProfileIn(BaseModel):
    name: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None


class UserPatch(BaseModel):
    active: Optional[bool] = None
    role: Optional[str] = None


def check_pw_policy(pw):
    if len(pw) < 8 or not re.search(r"[A-Za-z]", pw) or not re.search(r"\d", pw):
        raise HTTPException(400, "Password needs 8+ characters with letters and numbers.")


def new_session(uid):
    tok = secrets.token_urlsafe(32)
    exp = (datetime.now(timezone.utc) + timedelta(hours=SESSION_HOURS)).isoformat(timespec="seconds")
    q("INSERT INTO sessions VALUES(?,?,?)", (sha(tok), uid, exp), commit=True)
    return tok


def pub(u):
    return {"id": u["id"], "name": u["name"], "email": u["email"], "role": u["role"]}


def current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not signed in")
    row = q("""SELECT u.*, s.expires FROM sessions s JOIN users u ON u.id=s.user_id
               WHERE s.token=?""", (sha(authorization[7:]),), one=True)
    if not row or row["expires"] < now() or not row["active"]:
        raise HTTPException(401, "Session expired")
    return row


def admin_user(u=Depends(current_user)):
    if u["role"] != "admin":
        raise HTTPException(403, "Admin access required")
    return u


def login_flow(c: Creds, want_admin: bool):
    email = c.email.strip().lower()
    recent = [t for t in failed[email] if time.time() - t < 600]
    failed[email] = recent
    if len(recent) >= 5:
        raise HTTPException(429, "Too many failed attempts. Try again in 10 minutes.")
    u = q("SELECT * FROM users WHERE email=?", (email,), one=True)
    if not u or not check_pw(c.password, u["pw"]):
        failed[email].append(time.time())
        raise HTTPException(401, "Incorrect email or password")
    if not u["active"]:
        raise HTTPException(403, "This account is deactivated. Contact an administrator.")
    if want_admin and u["role"] != "admin":
        raise HTTPException(403, "This account does not have admin access")
    if not want_admin and u["role"] == "admin":
        raise HTTPException(403, "Admins must sign in at /admin")
    failed.pop(email, None)
    return {"token": new_session(u["id"]), "user": pub(u)}


@app.post("/api/auth/register")
def register(c: Creds):
    email = c.email.strip().lower()
    name = (c.name or "").strip()
    if not EMAIL_RE.match(email) or not name:
        raise HTTPException(400, "Enter your name and a valid email.")
    check_pw_policy(c.password)
    if q("SELECT 1 FROM users WHERE email=?", (email,), one=True):
        raise HTTPException(409, "An account with this email already exists.")
    q("INSERT INTO users(name,email,pw,created) VALUES(?,?,?,?)",
      (name[:80], email, hash_pw(c.password), now()), commit=True)
    return login_flow(Creds(email=email, password=c.password), False)


@app.post("/api/auth/login")
def login(c: Creds):
    return login_flow(c, False)


@app.post("/api/admin/login")
def admin_login(c: Creds):
    return login_flow(c, True)


@app.post("/api/auth/logout")
def logout(authorization: Optional[str] = Header(None), u=Depends(current_user)):
    q("DELETE FROM sessions WHERE token=?", (sha(authorization[7:]),), commit=True)
    return {"ok": True}


@app.get("/api/me")
def me(u=Depends(current_user)):
    return pub(u)


@app.put("/api/me")
def update_me(p: ProfileIn, u=Depends(current_user)):
    if p.name and p.name.strip():
        q("UPDATE users SET name=? WHERE id=?", (p.name.strip()[:80], u["id"]), commit=True)
    if p.new_password:
        if not p.current_password or not check_pw(p.current_password, u["pw"]):
            raise HTTPException(400, "Current password is incorrect.")
        check_pw_policy(p.new_password)
        q("UPDATE users SET pw=? WHERE id=?", (hash_pw(p.new_password), u["id"]), commit=True)
    return pub(q("SELECT * FROM users WHERE id=?", (u["id"],), one=True))


# ---------------- analysis helpers ----------------
def ser(r):
    d = dict(r)
    d["probs"] = json.loads(d["probs"] or "{}")
    d["details"] = json.loads(d["details"] or "{}")
    d["has_file"] = bool(d.pop("stored", None))
    return d


def thumb_bytes(img: Image.Image):
    im = img.convert("RGB")
    im.thumbnail((1024, 1024))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=88)  # re-encoding also strips EXIF/metadata
    return b.getvalue()


def store(user, kind, filename, img, res, details, save_file):
    stored = None
    if save_file and img is not None:
        stored = secrets.token_hex(16) + ".jpg"
        (UPLOADS / stored).write_bytes(thumb_bytes(img))
    details = {**details, "demo": res.get("demo", classifier.demo), "disclaimer": DISCLAIMER}
    aid = q("""INSERT INTO analyses(user_id,kind,filename,stored,label,confidence,status,probs,details,
               model_version,created) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (user["id"], kind, (filename or "upload")[:120], stored, res["label"], res["confidence"],
             res["status"], json.dumps(res["probs"]), json.dumps(details),
             res.get("model_version", classifier.version), now()), commit=True)
    return ser(q("SELECT * FROM analyses WHERE id=?", (aid,), one=True))


def rejected(why):
    return {"label": "NOT_A_XRAY", "confidence": 0.0, "probs": {}, "status": "rejected",
            "model_version": classifier.version, "note": why}


def ahash(img):
    a = np.asarray(img.convert("L").resize((8, 8)), dtype=float)
    return (a > a.mean()).flatten()


@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...), save_file: bool = Form(True),
                        user=Depends(current_user)):
    if Path(file.filename or "").suffix.lower() not in IMG_EXT:
        raise HTTPException(400, "Unsupported file type. Use JPG, PNG, BMP or WEBP.")
    data = await file.read(MAX_IMG + 1)
    if len(data) > MAX_IMG:
        raise HTTPException(413, "Image is larger than 10 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        img = img.convert("RGB")
    except Exception:
        raise HTTPException(400, "This file could not be read as an image.")
    ok, why = looks_like_xray(img)
    res = classifier.predict(img) if ok else rejected(why)
    return store(user, "image", file.filename, img, res, {"note": res.get("note", "")}, save_file)


@app.post("/api/analyze/video")
async def analyze_video(file: UploadFile = File(...), save_file: bool = Form(True),
                        user=Depends(current_user)):
    try:
        import cv2
    except ImportError:
        raise HTTPException(501, "Video support needs opencv-python-headless.")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VID_EXT:
        raise HTTPException(400, "Unsupported video type. Use MP4, AVI, MOV, MKV or WEBM.")
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_VIDEO:
                tmp.close(); os.unlink(tmp.name)
                raise HTTPException(413, "Video is larger than 60 MB.")
            tmp.write(chunk)
        path = tmp.name
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise HTTPException(400, "This video could not be opened.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(fps * FRAME_EVERY_SEC))
        i, sampled = 0, []
        while len(sampled) < MAX_FRAMES * 3 and i < 30000:
            if not cap.grab():
                break
            if i % step == 0:
                ok, fr = cap.retrieve()
                if ok:
                    sampled.append((i / fps, Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
            i += 1
        cap.release()
    finally:
        os.unlink(path)
    if not sampled:
        raise HTTPException(400, "No frames could be read from this video.")

    frames, hashes, skipped = [], [], {"not_xray": 0, "duplicate": 0}
    first_img = None
    for t, im in sampled:
        if len(frames) >= MAX_FRAMES:
            break
        ok, _ = looks_like_xray(im)
        if not ok:
            skipped["not_xray"] += 1
            continue
        h = ahash(im)
        if any((h != o).sum() <= 3 for o in hashes):
            skipped["duplicate"] += 1
            continue
        hashes.append(h)
        r = classifier.predict(im)
        first_img = first_img or im
        frames.append({"t": round(t, 1), "label": r["label"], "confidence": r["confidence"],
                       "probs": r["probs"], "status": r["status"]})
    details = {"sampled": len(sampled), "analysed": len(frames), "skipped": skipped,
               "frames": frames,
               "rule": "Mean of class probabilities over analysed frames; flagged for review if "
                       f"confidence < {REVIEW_THRESHOLD:.2f} or fewer than 60% of frames agree."}
    if not frames:
        res = rejected("No suitable chest X-ray frames were found in this video.")
        return store(user, "video", file.filename, None, res, details, False)
    classes = classifier.classes
    mean = {c: float(np.mean([f["probs"][c] for f in frames])) for c in classes}
    label = max(mean, key=mean.get)
    agree = sum(f["label"] == label for f in frames) / len(frames)
    details["agreement"] = round(agree, 2)
    status = "ok" if mean[label] >= REVIEW_THRESHOLD and agree >= 0.6 else "needs_review"
    res = {"label": label, "confidence": mean[label], "probs": mean, "status": status,
           "model_version": classifier.version, "demo": classifier.demo}
    return store(user, "video", file.filename, first_img, res, details, save_file)


# ---------------- user data ----------------
@app.get("/api/analyses")
def list_analyses(search: str = "", user=Depends(current_user)):
    rows = q("SELECT * FROM analyses WHERE user_id=? AND (filename LIKE ? OR label LIKE ?) "
             "ORDER BY id DESC LIMIT 500", (user["id"], f"%{search}%", f"%{search}%"))
    return [ser(r) for r in rows]


def own(aid, user):
    r = q("SELECT * FROM analyses WHERE id=? AND user_id=?", (aid, user["id"]), one=True)
    if not r:
        raise HTTPException(404, "Record not found")
    return r


@app.get("/api/analyses/{aid}")
def get_analysis(aid: int, user=Depends(current_user)):
    return ser(own(aid, user))


@app.get("/api/analyses/{aid}/file")
def get_file(aid: int, user=Depends(current_user)):
    r = own(aid, user)
    if not r["stored"]:
        raise HTTPException(404, "No image stored for this record")
    return FileResponse(UPLOADS / r["stored"], media_type="image/jpeg",
                        headers={"Cache-Control": "private, no-store"})


def delete_record(r):
    if r["stored"]:
        (UPLOADS / r["stored"]).unlink(missing_ok=True)
    q("DELETE FROM analyses WHERE id=?", (r["id"],), commit=True)


@app.delete("/api/analyses/{aid}")
def del_analysis(aid: int, user=Depends(current_user)):
    delete_record(own(aid, user))
    return {"ok": True}


@app.get("/api/stats")
def stats(user=Depends(current_user)):
    rows = q("SELECT label,status,COUNT(*) n FROM analyses WHERE user_id=? GROUP BY label,status", (user["id"],))
    out = {"total": 0, "by_label": {}, "needs_review": 0, "rejected": 0}
    for r in rows:
        out["total"] += r["n"]
        if r["status"] == "rejected":
            out["rejected"] += r["n"]
        else:
            out["by_label"][r["label"]] = out["by_label"].get(r["label"], 0) + r["n"]
            if r["status"] == "needs_review":
                out["needs_review"] += r["n"]
    return out


@app.get("/api/model")
def model_info(user=Depends(current_user)):
    return {**classifier.info(), "disclaimer": DISCLAIMER}


# ---------------- admin ----------------
@app.get("/api/admin/stats")
def admin_stats(_=Depends(admin_user)):
    one = lambda s, a=(): q(s, a, one=True)[0]
    since = (datetime.now(timezone.utc) - timedelta(days=13)).date().isoformat()
    daily = {r["d"]: r["n"] for r in q(
        "SELECT substr(created,1,10) d, COUNT(*) n FROM analyses WHERE created>=? GROUP BY d", (since,))}
    days = [(datetime.now(timezone.utc).date() - timedelta(days=i)).isoformat() for i in range(13, -1, -1)]
    return {
        "users": one("SELECT COUNT(*) FROM users WHERE role='user'"),
        "active_users": one("SELECT COUNT(*) FROM users WHERE role='user' AND active=1"),
        "analyses": one("SELECT COUNT(*) FROM analyses"),
        "videos": one("SELECT COUNT(*) FROM analyses WHERE kind='video'"),
        "needs_review": one("SELECT COUNT(*) FROM analyses WHERE status='needs_review'"),
        "rejected": one("SELECT COUNT(*) FROM analyses WHERE status='rejected'"),
        "by_label": {r["label"]: r["n"] for r in q(
            "SELECT label, COUNT(*) n FROM analyses WHERE status!='rejected' GROUP BY label")},
        "daily": [{"date": d, "count": daily.get(d, 0)} for d in days],
        "model": classifier.info(),
        "storage_mb": round(sum(f.stat().st_size for f in UPLOADS.glob("*")) / 1e6, 2),
    }


@app.get("/api/admin/users")
def admin_users(_=Depends(admin_user)):
    rows = q("""SELECT u.id,u.name,u.email,u.role,u.active,u.created,
                (SELECT COUNT(*) FROM analyses a WHERE a.user_id=u.id) n
                FROM users u ORDER BY u.id""")
    return [dict(r) for r in rows]


@app.patch("/api/admin/users/{uid}")
def admin_patch_user(uid: int, p: UserPatch, admin=Depends(admin_user)):
    if uid == admin["id"]:
        raise HTTPException(400, "You cannot change your own account here.")
    if not q("SELECT 1 FROM users WHERE id=?", (uid,), one=True):
        raise HTTPException(404, "User not found")
    if p.active is not None:
        q("UPDATE users SET active=? WHERE id=?", (int(p.active), uid), commit=True)
        if not p.active:
            q("DELETE FROM sessions WHERE user_id=?", (uid,), commit=True)
    if p.role in ("user", "admin"):
        q("UPDATE users SET role=? WHERE id=?", (p.role, uid), commit=True)
    return {"ok": True}


@app.delete("/api/admin/users/{uid}")
def admin_del_user(uid: int, admin=Depends(admin_user)):
    if uid == admin["id"]:
        raise HTTPException(400, "You cannot delete your own account.")
    for r in q("SELECT * FROM analyses WHERE user_id=?", (uid,)):
        if r["stored"]:
            (UPLOADS / r["stored"]).unlink(missing_ok=True)
    q("DELETE FROM users WHERE id=?", (uid,), commit=True)
    return {"ok": True}


@app.get("/api/admin/analyses")
def admin_analyses(search: str = "", _=Depends(admin_user)):
    # Metadata only: admins do not get access to patient images.
    rows = q("""SELECT a.id,a.kind,a.filename,a.label,a.confidence,a.status,a.model_version,a.created,
                u.email FROM analyses a JOIN users u ON u.id=a.user_id
                WHERE u.email LIKE ? OR a.filename LIKE ? OR a.label LIKE ?
                ORDER BY a.id DESC LIMIT 500""", (f"%{search}%",) * 3)
    return [dict(r) for r in rows]


@app.delete("/api/admin/analyses/{aid}")
def admin_del_analysis(aid: int, _=Depends(admin_user)):
    r = q("SELECT * FROM analyses WHERE id=?", (aid,), one=True)
    if not r:
        raise HTTPException(404, "Record not found")
    delete_record(r)
    return {"ok": True}


# ---------------- pages ----------------
@app.get("/")
def page_index():
    return FileResponse(STATIC / "index.html")


@app.get("/app")
def page_app():
    return FileResponse(STATIC / "app.html")


@app.get("/admin")
def page_admin():
    return FileResponse(STATIC / "admin.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
