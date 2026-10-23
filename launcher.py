"""Graphical launcher for the face detection project.

The launcher starts the existing scripts (live camera, batch image detection,
enrolment, people list) as child processes and streams their output into a
log panel. Packaged with PyInstaller it becomes a single Windows executable:

    pyinstaller --onefile --windowed --name FaceDetectionLauncher launcher.py
"""

from __future__ import annotations

import os
import queue
import runpy
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

APP_TITLE = "Face Detection Launcher"
WORKER_FLAG = "--worker"

SOURCE_DIR = "src"
MODEL_DIR = "models"
DATA_DIR = "data"
OUTPUT_DIR = "outputs"

ENTRY_SCRIPTS = ("detect_camera.py", "detect_image.py", "detect_live_combined.py", "enroll.py")

ACCENT = "#00A84F"
ACCENT_HOVER = "#008F43"
DANGER = "#C74343"
BG = "#F1F4F8"
CARD = "#FFFFFF"
BORDER = "#DCE3EA"
TEXT = "#1F2933"
MUTED = "#69798A"
LOG_BG = "#141A21"
LOG_FG = "#D5DEE9"

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

PROJECT_ROOT = ""
BUNDLED_DIR = ""


# --------------------------------------------------------------------------
# project discovery
# --------------------------------------------------------------------------
def _is_project(path: str) -> bool:
    if not path:
        return False
    return os.path.isfile(os.path.join(path, SOURCE_DIR, "detect_camera.py"))


def _root_candidates() -> list[str]:
    candidates: list[str] = []
    env_root = os.environ.get("FACE_APP_ROOT")
    if env_root:
        candidates.append(env_root)
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        candidates.append(os.path.join(exe_dir, "FaceDetectionApp"))
        current = exe_dir
        for _ in range(3):
            candidates.append(current)
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
    else:
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        candidates.append(os.getcwd())
    except OSError:
        pass
    return candidates


def find_project_root() -> str:
    for candidate in _root_candidates():
        if _is_project(candidate):
            return os.path.abspath(candidate)
    return ""


def prepare_workspace() -> str:
    """Copy the bundled app into a per-user folder when no project is found."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    workspace = os.path.join(base, "FaceDetection")
    for folder in (SOURCE_DIR, MODEL_DIR, DATA_DIR, OUTPUT_DIR):
        os.makedirs(os.path.join(workspace, folder), exist_ok=True)
    if not os.path.isdir(BUNDLED_DIR):
        return workspace
    for folder in (SOURCE_DIR, MODEL_DIR):
        source = os.path.join(BUNDLED_DIR, folder)
        if not os.path.isdir(source):
            continue
        for name in os.listdir(source):
            src = os.path.join(source, name)
            dst = os.path.join(workspace, folder, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=_copy_if_missing)
            elif os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
    return workspace


def _copy_if_missing(source: str, destination: str) -> str:
    """Copy a file unless the workspace already has one."""
    if os.path.exists(destination):
        return destination
    return shutil.copy2(source, destination)


# --------------------------------------------------------------------------
# worker mode (the frozen exe re-runs itself to run a script)
# --------------------------------------------------------------------------
def attach_stdio() -> None:
    """Reconnect stdout/stderr in windowed builds so output reaches the pipe."""
    for name, fd in (("stdout", 1), ("stderr", 2)):
        if getattr(sys, name, None) is not None:
            continue
        try:
            stream = os.fdopen(
                os.dup(fd), "w", encoding="utf-8", errors="replace", buffering=1
            )
        except (OSError, ValueError):
            continue
        setattr(sys, name, stream)


def run_worker(script: str, args: list[str]) -> int:
    attach_stdio()
    if not script.lower().endswith(".py"):
        script = f"{script}.py"
    script_path = os.path.join(PROJECT_ROOT, SOURCE_DIR, script)
    if not os.path.isfile(script_path):
        print(f"Missing script: {script_path}")
        return 1
    sys.path.insert(0, os.path.dirname(script_path))
    sys.argv = [script_path] + list(args)
    runpy.run_path(script_path, run_name="__main__")
    return 0


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
class Launcher:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.process: subprocess.Popen | None = None
        self.lines: queue.Queue = queue.Queue()
        self.buttons: list[ttk.Button] = []
        self.stop_button: ttk.Button | None = None
        self.status = tk.StringVar(value="Ready")
        self.location = tk.StringVar(value=PROJECT_ROOT)

        self._build_style()
        self._build_ui()
        self.root.after(100, self._drain)

    # -- styling ----------------------------------------------------------
    def _build_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Accent.TButton",
            font=("Segoe UI", 11, "bold"),
            foreground="#FFFFFF",
            background=ACCENT,
            borderwidth=0,
            focusthickness=0,
            padding=(14, 9),
        )
        style.map(
            "Accent.TButton",
            background=[("active", ACCENT_HOVER), ("disabled", "#BFC9D2")],
        )
        style.configure(
            "Quiet.TButton",
            font=("Segoe UI", 10),
            foreground=TEXT,
            background=CARD,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            borderwidth=1,
            relief="solid",
            focusthickness=0,
            padding=(14, 8),
        )
        style.map(
            "Quiet.TButton",
            background=[("active", "#E7ECF2")],
            bordercolor=[("active", "#C3CDD8")],
        )
        style.configure(
            "Danger.TButton",
            font=("Segoe UI", 10, "bold"),
            foreground="#FFFFFF",
            background=DANGER,
            borderwidth=0,
            relief="flat",
            focusthickness=0,
            padding=(14, 8),
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#A93333"), ("disabled", "#C4CBD2")],
        )
        style.configure("TLabel", background=CARD, foreground=TEXT)

    # -- layout -----------------------------------------------------------
    def _build_ui(self) -> None:
        self.root.title(APP_TITLE)
        self.root.geometry("860x640")
        self.root.minsize(760, 560)
        self.root.configure(bg=BG)

        header = tk.Frame(self.root, bg=BG)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=24, pady=(20, 12))
        tk.Label(
            header,
            text="Face Detection",
            font=("Segoe UI", 20, "bold"),
            bg=BG,
            fg=TEXT,
        ).pack(anchor="w")
        tk.Label(
            header,
            text="YuNet detection and SFace recognition - camera, images and enrolment.",
            font=("Segoe UI", 10),
            bg=BG,
            fg=MUTED,
        ).pack(anchor="w", pady=(2, 0))

        body = tk.Frame(self.root, bg=BG)
        body.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=24)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=0, minsize=330)
        self.root.grid_columnconfigure(1, weight=1)

        self._build_actions(body)
        self._build_log(body)
        self._build_status_bar()

    def _card(self, parent: tk.Misc) -> tk.Frame:
        card = tk.Frame(
            parent,
            bg=CARD,
            highlightthickness=1,
            highlightbackground=BORDER,
            bd=0,
        )
        return card

    def _build_actions(self, parent: tk.Misc) -> None:
        card = self._card(parent)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        parent.grid_rowconfigure(0, weight=1)

        tk.Label(
            card,
            text="ACTIONS",
            font=("Segoe UI", 9, "bold"),
            bg=CARD,
            fg=MUTED,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 10))

        rows = [
            ("Live camera detection", "Open the webcam and label every face in real time.", "detect_camera", "primary"),
            ("Live detection + liveness", "Webcam with face recognition and the blink/head-movement liveness check.", "detect_live_combined", "primary"),
            ("Detect faces in images", "Process every picture in the data folder.", "detect_image", "primary"),
            ("Enrol from camera", "Add a new person using webcam samples.", "enroll", "quiet"),
            ("Enrol from images", "Add a new person from a folder of pictures.", "enroll_images", "quiet"),
            ("List known people", "Show everyone stored in the face database.", "list", "quiet"),
        ]

        row = 1
        for title, description, action, kind in rows:
            button = ttk.Button(
                card,
                text=title,
                style="Accent.TButton" if kind == "primary" else "Quiet.TButton",
                command=lambda a=action: self._start(a),
            )
            button.grid(row=row, column=0, sticky="ew", padx=(18, 8), pady=(0, 2))
            tk.Label(
                card,
                text=description,
                font=("Segoe UI", 9),
                bg=CARD,
                fg=MUTED,
                wraplength=260,
                justify="left",
            ).grid(row=row, column=1, sticky="w", padx=(0, 18), pady=(2, 12))
            self.buttons.append(button)
            row += 1

        card.grid_columnconfigure(0, weight=1)

        stop = ttk.Button(
            card,
            text="Stop running task",
            style="Danger.TButton",
            command=self._stop,
            state="disabled",
        )
        stop.grid(row=row, column=0, columnspan=2, sticky="ew", padx=18, pady=(8, 14))
        self.stop_button = stop

        folders = tk.Frame(card, bg=CARD)
        folders.grid(row=row + 1, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 16))
        for index, (label, folder) in enumerate(
            (("data", DATA_DIR), ("outputs", OUTPUT_DIR), ("project", ""))
        ):
            ttk.Button(
                folders,
                text=f"Open {label} folder",
                style="Quiet.TButton",
                command=lambda f=folder: self._open_folder(f),
            ).grid(row=0, column=index, sticky="ew", padx=(0, 6) if index < 2 else 0)
        folders.grid_columnconfigure(0, weight=1)
        folders.grid_columnconfigure(1, weight=1)
        folders.grid_columnconfigure(2, weight=1)

    def _build_log(self, parent: tk.Misc) -> None:
        card = self._card(parent)
        card.grid(row=0, column=1, sticky="nsew")
        parent.grid_columnconfigure(1, weight=1)

        bar = tk.Frame(card, bg=CARD)
        bar.grid(row=0, column=0, sticky="ew")
        tk.Label(
            bar,
            text="OUTPUT",
            font=("Segoe UI", 9, "bold"),
            bg=CARD,
            fg=MUTED,
        ).pack(side="left", padx=18, pady=(16, 8))
        ttk.Button(
            bar,
            text="Clear",
            style="Quiet.TButton",
            command=self._clear_log,
        ).pack(side="right", padx=18, pady=(10, 6))

        self.log = ScrolledText(
            card,
            wrap="word",
            width=40,
            height=14,
            bg=LOG_BG,
            fg=LOG_FG,
            insertbackground=LOG_FG,
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=14,
            pady=12,
            font=("Consolas", 10),
        )
        self.log.grid(row=1, column=0, sticky="nsew")
        card.grid_rowconfigure(1, weight=1)
        card.grid_columnconfigure(0, weight=1)
        self.log.configure(state="disabled")
        self._append(
            f"Project folder: {PROJECT_ROOT}\n"
            "Pick an action on the left to start. Output from the task shows up here."
        )

    def _build_status_bar(self) -> None:
        bar = tk.Frame(self.root, bg=BG)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=24, pady=(10, 16))
        tk.Label(
            bar,
            textvariable=self.status,
            font=("Segoe UI", 9, "bold"),
            bg=BG,
            fg=TEXT,
        ).pack(side="left")
        tk.Label(
            bar,
            textvariable=self.location,
            font=("Segoe UI", 9),
            bg=BG,
            fg=MUTED,
        ).pack(side="right")

    # -- logging ----------------------------------------------------------
    def _append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _drain(self) -> None:
        finished = False
        while True:
            try:
                item = self.lines.get_nowait()
            except queue.Empty:
                break
            if item is None:
                finished = True
            else:
                self._append(item)
        if finished:
            self._set_busy(False)
            self._append("Task finished.\n")
        self.root.after(100, self._drain)

    # -- tasks ------------------------------------------------------------
    def _set_busy(self, busy: bool) -> None:
        for button in self.buttons:
            button.configure(state="disabled" if busy else "normal")
        if self.stop_button is not None:
            self.stop_button.configure(state="normal" if busy else "disabled")
        self.status.set("Running" if busy else "Ready")

    def _python_command(self, script: str, args: list[str]) -> list[str]:
        if getattr(sys, "frozen", False):
            return [sys.executable, WORKER_FLAG, script, *args]
        venv = os.path.join(PROJECT_ROOT, "face_env", "Scripts", "python.exe")
        interpreter = venv if os.path.isfile(venv) else sys.executable
        return [interpreter, os.path.join(PROJECT_ROOT, SOURCE_DIR, script), *args]

    def _start(self, action: str) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showinfo("Busy", "A task is already running. Stop it first.")
            return

        if action == "enroll":
            name = simpledialog.askstring(
                "Enrol a person",
                "Name of the person:",
                parent=self.root,
            )
            if not name or not name.strip():
                return
            script, extra = "enroll.py", ["--name", name.strip()]
            label = f"Enrol {name.strip()} (camera)"
        elif action == "enroll_images":
            name = simpledialog.askstring(
                "Enrol a person",
                "Name of the person:",
                parent=self.root,
            )
            if not name or not name.strip():
                return
            folder = filedialog.askdirectory(
                title="Folder with pictures of the person", parent=self.root
            )
            if not folder:
                return
            script = "enroll.py"
            extra = ["--name", name.strip(), "--images", folder]
            label = f"Enrol {name.strip()} (images)"
        elif action == "detect_live_combined":
            script, extra, label = "detect_live_combined.py", [], "Live detection + liveness"
        elif action == "detect_camera":
            script, extra, label = "detect_camera.py", [], "Live camera detection"
        elif action == "detect_image":
            script, extra, label = "detect_image.py", [], "Detect faces in images"
        else:
            script, extra, label = "enroll.py", ["--list"], "List known people"

        command = self._python_command(script, extra)
        self._append(f"\n=== {label} ===\n")
        self.status.set(label)
        try:
            self.process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=CREATE_NO_WINDOW,
            )
        except OSError as error:
            self._append(f"Could not start the task: {error}\n")
            self.status.set("Ready")
            return

        self._set_busy(True)
        threading.Thread(target=self._pump, args=(self.process,), daemon=True).start()

    def _pump(self, process: subprocess.Popen) -> None:
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    self.lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self.lines.put(None)

    def _stop(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        self._append("Stopping...\n")

        def _terminate() -> None:
            try:
                process.terminate()
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception:
                pass

        threading.Thread(target=_terminate, daemon=True).start()

    # -- helpers ----------------------------------------------------------
    def _open_folder(self, folder: str) -> None:
        path = os.path.join(PROJECT_ROOT, folder) if folder else PROJECT_ROOT
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except (OSError, AttributeError) as error:
            messagebox.showerror("Could not open folder", f"{path}\n{error}")


def main() -> int:
    global PROJECT_ROOT, BUNDLED_DIR

    if getattr(sys, "frozen", False):
        BUNDLED_DIR = os.path.join(sys._MEIPASS, "FaceDetectionApp")  # type: ignore[attr-defined]

    if len(sys.argv) > 2 and sys.argv[1] == WORKER_FLAG:
        PROJECT_ROOT = find_project_root() or prepare_workspace()
        return run_worker(sys.argv[2], sys.argv[3:])

    PROJECT_ROOT = find_project_root() or prepare_workspace()
    root = tk.Tk()
    Launcher(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
