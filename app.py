import os
from flask import Flask, redirect, render_template, request, session, g, jsonify
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
import sqlite3
from model import detector
import time

from helpers import login_required, extract_text_from_pdf


# Create Flask app, configure app
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["DEBUG"] = True
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Create connection to store requests individually
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect("candidates.db")
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()
        
# Create table for candidates.db
with app.app_context():
    db = get_db()
    db.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, hash TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY AUTOINCREMENT, uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP, file_name TEXT NOT NULL, file_type TEXT NOT NULL, file_size TEXT NOT NULL, uploader_id INTEGER NOT NULL, FOREIGN KEY (uploader_id) REFERENCES users(id) ON DELETE CASCADE)")
    db.execute("CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY AUTOINCREMENT, inferred_at DATETIME DEFAULT CURRENT_TIMESTAMP, submission_id INTEGER NOT NULL, verdict TEXT NOT NULL, confidence INTEGER NOT NULL,  FOREIGN KEY (submission_id) REFERENCES files(id) ON DELETE CASCADE)")
    db.execute("CREATE TABLE IF NOT EXISTS stats (id INTEGER PRIMARY KEY AUTOINCREMENT, submission_id INTEGER NOT NULL, t_extract_ms INTEGER NOT NULL, t_infer_ms INTEGER NOT NULL, t_total_ms INTEGER NOT NULL, FOREIGN KEY (submission_id) REFERENCES files(id) ON DELETE CASCADE)")
    db.commit()


#Ensure caches are not used
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

@app.route("/")
@login_required
def index():
    """Create new connection with db"""
    db = get_db()
    """Show portfolio of candidates"""
    candidates = db.execute("SELECT username FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    return render_template("index.html", candidates=candidates["username"])

#Ensure file upload exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok= True)

@app.route("/detect", methods=["GET", "POST"])
@login_required
def load():
    db = get_db()
    candidates = db.execute("SELECT username FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    return render_template("detect.html", candidates=candidates["username"])


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    try:
        if request.method == "POST":
            if "file" not in request.files:
                return jsonify({"error": "No file part"}), 400
            file = request.files["file"]
            
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            if not file.filename.lower().endswith('.pdf'):
                return jsonify({"error": "File must be of pdf format"}), 400
            
            file.stream.seek(0,2)
            file_size = file.stream.tell()
            file.stream.seek(0)
            
            t0 = time.perf_counter()
            text_file = extract_text_from_pdf(file)
            t1 = time.perf_counter()
            
            if not text_file.strip():
                return jsonify({"error": "Could not extract text from file"}), 400
            
            t2 = time.perf_counter()
            prediction, confidence = detector.predict(text_file)
            t3 = time.perf_counter()
            
            #Store metadata for each run
            submission_id = record_meta(file_name = file.filename, file_type = ".pdf", file_size = file_size, prediction=prediction, confidence=confidence)
            
            #Record statistics for each run
            stats = record_stats(submission_id = submission_id, t_extract_ms = round((t1 - t0) * 1000), t_infer_ms = round((t3 - t2) * 1000), t_total_ms=round((t3 - t0) * 1000))
            
            return jsonify({
                "success": True,
                "submission_id": submission_id,
                "verdict": prediction,
                "confidence": confidence,
            }), 200
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

def record_meta(file_name, file_type, file_size, prediction, confidence):
    db = get_db()
    with db:
        cur = db.execute("INSERT INTO files (uploader_id, file_name, file_type, file_size) VALUES (?, ?, ?, ?)", (session["user_id"], file_name, file_type, file_size))
        submission_id = cur.lastrowid
        db.execute("INSERT INTO results (submission_id, verdict, confidence) VALUES (?, ?, ?)", (submission_id, prediction, confidence))
        return submission_id

   
def record_stats(submission_id, t_extract_ms, t_infer_ms, t_total_ms):
    db = get_db()
    with db:
        stat = db.execute("INSERT INTO stats (submission_id, t_extract_ms, t_infer_ms, t_total_ms) VALUES (?, ?, ?, ?)", (submission_id, t_extract_ms, t_infer_ms, t_total_ms,))


@app.route("/results/<int:submission_id>", methods=["GET"])
@login_required
def results(submission_id):
    db = get_db()
    
    load = db.execute("SELECT f.file_name, r.inferred_at, r.verdict, r.confidence FROM files f LEFT JOIN results r ON r.submission_id = f.id WHERE f.uploader_id = ? AND f.id = ?", (session["user_id"], submission_id,)).fetchone()
    
    return render_template("results.html", file_name=load["file_name"], inferred_at=load["inferred_at"], prediction=load["verdict"], confidence=load["confidence"])


@app.route("/register", methods=["GET", "POST"])
def register():
    """Create new connection with db"""
    db = get_db()
    """Register user"""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")

        # Ensure username was submitted
        if not username:
            return render_template("register.html", message="Must provide username")

        # Ensure password was submitted
        elif not password:
            return render_template("register.html", message="Must provide password")

        # Ensure confirmation was submitted
        elif not confirmation:
            return render_template("register.html", message="Must provide password confirmation")

        # Ensure password matches confirmation
        elif password != confirmation:
            return render_template("register.html", message="Passwords do not match")

        # Ensure username is unique
        rows = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if rows is not None:
            return render_template("register.html", message="Username already exists")

        # Insert new user into database
        hash = generate_password_hash(password)
        db.execute("INSERT INTO users (username, hash) VALUES (?, ?)", (username, hash))
        db.commit()

        # Redirect user to login page
        return redirect("/login")

    else:
        return render_template("register.html")
    

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""
    session.clear()
    """Create new connection with db"""
    db = get_db()

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Ensure username was submitted
        if not username:
            return render_template("login.html", message="Must provide username")

        # Ensure password was submitted
        elif not password:
            return render_template("login.html", message="Must provide password")

        # Query database for username
        rows = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

        # Ensure username exists and password is correct
        if rows is None or not check_password_hash(rows["hash"], password):
            return render_template("login.html", message="Invalid username and/or password")

        # Remember which user has logged in
        session["user_id"] = rows["id"]

        # Redirect user to home page
        return redirect("/")

    else:
        return render_template("login.html")

    
@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    return redirect("/login")

@app.route("/history", methods=["GET", "POST"])
@login_required
def history():
    db = get_db()
    
    username = db.execute("SELECT username FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    history = db.execute("SELECT f.id, datetime(f.uploaded_at, 'localtime') AS uploaded_at, f.file_name, f.file_type, f.file_size, datetime(r.inferred_at, 'localtime') AS inferred_at , r.verdict, r.confidence FROM files f LEFT JOIN results r ON r.submission_id = f.id WHERE f.uploader_id = ? ORDER BY f.uploaded_at DESC", (session["user_id"],)).fetchall()
   
    return render_template("history.html", rows=history, candidates=username["username"])


@app.route("/statistics", methods=["GET", "POST"])
@login_required
def stats():
    db = get_db()
    
    username = db.execute("SELECT username FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    summary = db.execute("SELECT COUNT(*) AS total_uploads, SUM(CASE WHEN r.verdict = 'FAKE' THEN 1 ELSE 0 END) AS fake_count, SUM(CASE WHEN r.verdict = 'REAL' THEN 1 ELSE 0 END) AS real_count, ROUND(AVG(s.t_total_ms), 2) AS avg_total, ROUND(AVG(s.t_extract_ms), 2) AS avg_extract, ROUND(AVG(s.t_infer_ms), 2) AS avg_infer, ROUND(AVG(r.confidence), 2) AS avg_confidence FROM files f LEFT JOIN results r ON r.submission_id = f.id LEFT JOIN stats s ON s.submission_id = f.id WHERE f.uploader_id = (?)", (session["user_id"],)).fetchone()
    
    return render_template("statistics.html", summary=summary, candidates=username["username"])