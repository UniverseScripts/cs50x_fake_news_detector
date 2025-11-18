# Fake News Detector - CS50x Final Project
#### Video Demo:  <URL HERE>
#### Description: 

## Setup
### Installation
1. **Clone the repository**
    ```bash
    git clone <https://github.com/UniverseScripts/cs50x_fake_news_detector/>
    cd cs50x_fake_news_detector

2. **Create a Python virtual environment**
    python -m venv CS50_Final_Project.venv
    CS50_Final_Project.venv\Scripts\activate (For Windows)
    source CS50_Final_Project.venv/bin/activate (For Linux/Mac)
    **=> While not compulsory, a virtual enviroment is heavily recommended as to avoid file and module obstructions from other languages**

3. **Install dependencies**
    pip install -r requirements.txt
    **IMPORTANT: Your Python must be a prior 3.13 version (3.12.10 recommended) since PyTorch and transformers have not been updated to the latest versions (3.13 & 3.14)**
    **Note: User should reserve roughly _2GB_ of storage since Pytorch and transformer modules are relatively heavy, not to mention one would need minimally _500MB+_ to download a pre-trained AI model**

4. **Run session**
    Type 'flask run' to start session
    Press Ctrl+C to end session

### Troubleshooting
- If installation is halted due to storage issue, kindly check and delete some space for the project. Empirically, if you are having trouble on C disk storage, I'd recommend creating a virtual environment (see "Installation: 2.") on D disk and launch the project there instead.
- If you see "transformers/torch library not found", it is very likely due to your latest Python version installed or set as a system environment variable. Remember to check your Python installation and ensure that the interpreter of your environment (where your project is located at) is a **PRIOR 3.13 version**. (see "Installation: 3.")

## Instructions
### Project structure
#### It is important to check whether the dependent files are sufficient or not. Otherwise, the project may not launch properly.
**Post-installation structure**
    CS50_Final_Project/
    ├── app.py                 # Main Flask application
    ├── model.py              # AI model and prediction logic
    ├── helpers.py            # Utility functions (PDF extraction, auth decorators)
    ├── requirements.txt      # Python dependencies
    ├── candidates.db         # SQLite database (created automatically)
    ├── templates/            # HTML templates
    │   ├── layout.html       # Mainframe layout template
    │   ├── index.html
    │   ├── login.html
    │   ├── register.html
    │   ├── detect.html
    │   ├── results.html
    │   ├── history.html
    │   └── statistics.html
    └── static/                 # CSS/Image assets
    │   └── css/                # CSS assets
    │   │    └── detect.css
    │   │    └── history.css
    │   │    └── layout.css
    │   │    └── session.css
    │   │    └── statistics.css
    │   └── home_logo.png       # Image assets
    └── CCS50_Final_Project.venv/   # Important assets/libraries within the Python environment (Created upon creating a Python enviroment, see "Installation: 2.")

**First-flask-run structure (see "Installation: 4.")**
Post-installation structure
    └── flask_session/   # Records Flask sessions (Automatically created after running a Flask session)
    └── \_\_pycache\_\_/   # Runs, interprets and stores the 3 ".py" files (Automatically created upon Flask launch)

**Note: \_\_pycache\_\_ may or may not be created since this depends on your VSCode settings but by default, it should.**

### Terminal output
#### Whenever you run "flask run" on your terminal, they are lines of output that may be arcane to many users who are not acquainted with similar projects.
**Firstly**, you should see on the default launch prompt notifying the user that the Python enviroment has launched:
Linux's bash terminal:
"Admin@user /d/CS50x 2025/Final_Project
 $ source "D:/CS50x 2025/Final_Project/CS50_Final_Project.venv/Scripts/activate"
 ((CS50_Final_Project.venv) ) "

Windows's cmd terminal:
"Microsoft Windows [Version xx]
(c) Microsoft Corporation. All rights reserved.

D:\CS50x 2025\Final_Project>"D:/CS50x 2025/Final_Project/CS50_Final_Project.venv/Scripts/activate.bat"

(CS50_Final_Project.venv)"

As you can see, the "activate" file was shown, offering a great start to the initiation of the project.

**Secondly**, you should check the terminal output after running "flask run" for any potential issues. Here are 2 important output lines to scrutinize:
1. "Running on https://... " -> This is the link that the project will be running on.
2. "Device set to use cpu/gpu" -> Take good note of this as you will have an option to explicitly set the AI model inherent within the project to run on your CPU or GPU, whichever is more efficient. (see "model.py: 4.")

### The introduction seems adequate, I think. It's time to move onto the next section: understanding the code

# Comprehension
#### In this section, I will be guiding you through the **IMPORTANT** codes that truly define the Fake News Detector project!

## app.py
Description: This is the mainframe application with an agglomeration of codes. Flask will not be able to detect and run the website without this file. I will be using the comments made on the file and "@app.route"s for references. (app.py:linenum)

1. Comment 1 (app.py:12)
These are the configurations to setup the web application. \["TEMPLATES_AUTO_RELOAD"] is enabled to display live changes on HTML, CSS.
Note: Auto-reload **DOES NOT** load changes on the intepreted python code changes such as SQLite3 database manipulation, AI model modifications. You will have to manually "CTRL+C" to cancel and re-launch "flask run" to load the latest changes.

2. Comment 2 (app.py:20) **This must be declared for projects that utilize SQLite3 _outside_ of CS50x's workspace; the latter has this entirely enabled and set up for begineers**
This section defines the get_db() function that does the following **in order**:
- Establishes connection with SQLite via sqlite3.connect
- Returns a format with similar properties as a Python list, but not defined as a "list".
- Enables use of "FOREIGN KEY"s.
- Automatically closes the thread connection with the database via @app.teardown wrapper.
Explanation: In other words, users will have to do "thread-handling" by themselves, manually calling get_db() for each connection they want to establish with the SQLite3's database. This does not handle 'race conditions' just yet. Considering the scope of audience this project would reach, I find it unecessary but one **MUST** handle this on real products.

3. after_request(response) (app.py:45)
Disables caching so pages always reflect the latest state. Sets headers (Cache-Control, Pragma, Expires=0) after every request to prevent browser and proxy caches from serving stale pages. Returns the modified (response). This is important for auth transitions and result pages. (Flask @app.after_request)

4. @app.route("/") → index() (app.py:52) (login_required)  
Fetches the current user via (get_db) and (session["user_id"]), then renders **index.html** with the username as `candidates`. Keeps DB lifecycle safe through (get_db) and teardown (close_db). (SELECT … WHERE id=?), (render_template)

5. @app.route("/detect") → load() (app.py:64) (login_required)  
Prepares the detection workspace. Pulls username using (get_db) and (session["user_id"]) and renders **detect.html**. This route only sets up the UI; actual upload and inference happen elsewhere. (render_template)

6. @app.route("/upload", methods=["GET","POST"]) → upload() (app.py:72) (login_required)  
Validates upload presence/type (PDF). Measures timings with (time.perf_counter). Computes size via (file.stream.seek/tell). Extracts text (extract_text_from_pdf), predicts (detector.predict), and persists run using (record_meta) and (record_stats). Returns JSON (jsonify) with (submission_id, verdict, confidence). Defensive checks: missing file, empty filename, wrong extension, empty text. (try/except)

7. record_meta(file_name, file_type, file_size, prediction, confidence) (app.py:118)  
Creates a submission then its result atomically using (sqlite3.connect via get_db) inside (with db:). Inserts into **files** (uploader_id, file_name, file_type, file_size), captures (cur.lastrowid) as (submission_id), then inserts into **results** (verdict, confidence). Returns (submission_id). (INSERT, lastrowid)

8. record_stats(submission_id, t_extract_ms, t_infer_ms, t_total_ms) (app.py:127)  
Persists timing metrics per submission. Uses (with db:) to (INSERT) into **stats** the extraction, inference, and total milliseconds tied to (submission_id). Keeps performance data decoupled from core verdicts. (INSERT)

9. @app.route("/results/<int:submission_id>") → results(submission_id) (app.py:133) (login_required)  
Loads one run by joining **files** ↔ **results** (LEFT JOIN), scoped to the current user via (session["user_id"]). Renders **results.html** with (file_name, inferred_at, verdict, confidence). Prevents cross-user access by filtering on uploader id. (SELECT … JOIN … WHERE uploader_id=? AND id=?), (render_template)

10. @app.route("/register", methods=["GET","POST"]) → register() (app.py:143)  
Validates inputs (username, password, confirmation). Ensures uniqueness (SELECT). Hashes password with (generate_password_hash) and inserts into **users**. On success, (redirect) to **/login**; otherwise re-renders **register.html** with a message. Uses (db.commit) for durability. (POST/Redirect/GET)

11. @app.route("/login", methods=["GET","POST"]) → login() (app.py:186)  
Clears prior session (session.clear). Looks up user (SELECT) and verifies password using (check_password_hash). On success, sets (session["user_id"]) and (redirects) to **/**; on failure, re-renders **login.html** with an error. (session)

12. @app.route("/logout") → logout() (app.py:222)  
Logs the user out by (session.clear) and redirects to **/login**. Stateless and idempotent; ensures protected routes trigger (login_required) next time. (redirect)

13. @app.route("/history") → history() (app.py:227) (login_required)  
Shows the user’s full ledger of uploads and inferences. Executes (LEFT JOIN) across **files** and **results**, localizes timestamps with (datetime(..., 'localtime')), sorts newest first, and renders **history.html** with rows. Scopes to the current user via (session["user_id"]). (ORDER BY DESC)

14. @app.route("/statistics") → stats() (app.py:238) (login_required)  
Computes aggregates per user: total uploads, FAKE/REAL counts, mean total/extract/infer times, and average confidence. Uses (COUNT, SUM, CASE, AVG, ROUND) across **files**, **results**, **stats** with (LEFT JOIN). Renders **statistics.html** with `summary` plus username. (SELECT aggregate)

Notes on DB lifecycle (already introduced):  
- **get_db() / close_db()**: one connection per request via (sqlite3.connect), sets (sqlite3.Row), enables (PRAGMA foreign_keys=ON), and closes at teardown.  
- **Tables creation block**: runs within (app.app_context), creating **users**, **files**, **results**, **stats** using (CREATE TABLE IF NOT EXISTS) and (FOREIGN KEY … ON DELETE CASCADE).

## helpers.py
Description: Utility helpers for auth guarding and robust PDF → text extraction/cleanup. (helpers.py:linenum)

1. login_required(f) (helpers.py:6)  
Decorator that blocks access unless a user is authenticated. Uses (functools.wraps) to preserve metadata, checks (session.get("user_id")), and redirects unauthenticated users to **/login** via (redirect). Keeps route code clean while centralizing access control.

2. _normalize_text(s) (helpers.py:18)  
Returns safe, clean `str` for the model. Decodes (bytes/bytearray) as UTF‑8 with `errors="ignore"`, collapses excess whitespace/tabs via (re.sub), caps consecutive newlines (re.sub `\n{3,}` → `\n\n`), and (strip)s ends. Prevents odd Unicode and layout artifacts from derailing inference.

3. extract_text_from_pdf(file_obj) (helpers.py:30)  
Tries (pdfplumber.open) first, concatenating page text with trailing `\n`; on failure, falls back to (PyPDF2.PdfReader). Resets stream pointer (file_obj.stream.seek(0)) between attempts. Returns normalized text via (_normalize_text). If both fail, returns empty string for graceful handling upstream.


## model.py
Description: Loads the Fake‑News classifier and standardizes predictions. Also provides chunking so long texts work with BERT’s token limits. (model.py)

1. _to_int_percent(p: float) → int (model.py:7)  
Converts probability `[0,1]` into integer percent `[0..100]`. Guards (None) and (math.isnan), then clamps with (max/min) and (round). Keeps UI consistent and prevents out‑of‑range values.
2. postprocess_predictions(all_scores) → (verdict, confidence) (model.py:14)  
Accepts list of `{label, score}` dicts. Picks highest (max with key="score"), normalizes label, maps common variants (fake/real/true/false), handles label sets like `{"label_0","label_1"}` or `{"negative","positive"}`, and defaults to “REAL”. Returns `(“REAL”|“FAKE”, int[0..100])` using (_to_int_percent).

3. chunks(text, tokenizer, max_len=512, stride=256) → List[str] (model.py:58)  
Tokenizes with (tokenizer), slices overlapping windows by token indices (pointer/stride), and decodes back to text (tokenizer.decode, skip_special_tokens=True). Returns `[]` for empty input. Enables safe processing of texts longer than BERT’s 512‑token limit.

4. class FakeNewsDetector.__init__(self) (model.py:91)  
Initializes a (transformers.pipeline) for `"text-classification"` using `model="Pulk17/Fake-News-Detection"`, with (top_k=None) to get full label distributions. Sets `device=-1` (CPU); users with CUDA can switch to `0` for GPU.

5. FakeNewsDetector.predict(self, text) → (verdict, confidence) (model.py:97)  
Short‑circuits empty input to (“REAL”, 0). Gets (tokenizer), relaxes `model_max_length`, builds windows via (chunks). Runs batched inference with (pipeline) (`truncation=True`, `max_length=512`, `padding=False`). Aggregates per‑label scores across chunks, normalizes, then calls (postprocess_predictions). Returns bounded `(“REAL”|“FAKE”, 0..100)`; on exception, returns (“Error: …”, 0).

6. detector = FakeNewsDetector() (model.py:155)  
Declares the model as a global variable that can be called across Python files. In "app.py", it is called when importing "model.py". (app.py:6)

## templates/
Description: This is the front-end sector of the project, powered by HTML, styled with CSS and constructed with Jinja.

1. Abstraction
The "layout.html" itself is the abstraction of the website and includes:
- 5 CSS sheets in "/static/css/".
- 2 layouts: one for "register" and "login"; another for the post-login format, aka @login-required in "app.py".
Note that I have used Jinja's condition calls to achieve the layout.

**Important notice: Every single file _EXCEPT_ for "detect.html" has a similar format, of which I will abstract to save some reading time.**

2. "The similar format"
Notice that HTML files in these catagories all do the same thing, which is to display either a form with "POST" as the API call method to their respective routers in "app.py" **OR** retrieving elements via Jinja's {{variable_name}} post render_template, also in "app.py".
Here is the general consensus:
- {% extends "layout.html"%} -> This initiates the branching from the "parent" template, aka the abstraction (see "templates/: 1.").
- {% block title %} Page title {% endblock %} -> Extends and inserts content for the page title in <title> tags within the <head> section of "layout.html".
- {% block body %} Html input {% endblock %} -> Extends and inserts content for the page title in <main> tags within the <body> section of "layout.html".

3. JavaScript's API call ("detect.html")
This section handles the asynchronous upload and inference request. The script binds to the “Analyze” button, captures the selected PDF via (FormData), and sends it to the Flask endpoint **/upload** using (fetch) with `method: "POST"`.  
Key steps:  
- Validate file presence before sending.  
- Display a loading spinner while awaiting response.  
- Parse JSON response from Flask: `{success, submission_id, verdict, confidence}`.  
- Dynamically update the DOM to show verdict and confidence without page reload.  
Error handling: If response includes `"error"`, display it in a styled alert box.  

Security note: CSRF tokens are unnecessary here since the session is server-side managed.
