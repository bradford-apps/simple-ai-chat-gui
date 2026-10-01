#!/usr/bin/env python3
"""
github.com/bradford-apps/simple-ai-chat-gui v1.00.0
===================================================
OpenAI-API-compatible Chat GUI (dark theme).

Works with any endpoint that implements the OpenAI "chat completions" API
(OpenAI, OpenRouter, LM Studio, Ollama, DeepSeek, llama.cpp, ...).

Requires:  pip install requests

Data layout (created automatically in subfolders of this script's folder):
    data/providers.json    providers + their models
    data/assistants.json   assistants / characters (system prompts)
    data/settings.json     last selections (provider/model/assistant/temp/top_p)
    chats/*.json           one JSON file per chat
"""

import base64
import datetime
import json
import mimetypes
import os
import queue
import shutil
import threading
import uuid

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import requests
except ImportError:
    requests = None

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
CHATS_DIR = os.path.join(APP_DIR, "chats")
PROVIDERS_FILE = os.path.join(DATA_DIR, "providers.json")
ASSISTANTS_FILE = os.path.join(DATA_DIR, "assistants.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")

APP_TITLE = "github.com/bradford-apps/simple-ai-chat-gui v1.00.0"

# dark palette
BG        = "#1e1e1e"
BG_PANEL  = "#252526"
BG_WIDGET = "#2d2d30"
FG        = "#e0e0e0"
FG_DIM    = "#9a9a9a"
ACCENT    = "#0e639c"
CHAT_BG   = "#000000"   # main chat area: black
USER_BG   = "#d3d3d3"   # user text: light gray background
USER_FG   = "#000000"   #            black text
THINK_BG  = "#404040"   # thinking: dark grey background
ERR_FG    = "#ff6b6b"

# send / stop button colors
SEND_BG        = "#2e7d32"   # green when idle
SEND_BG_ACTIVE = "#388e3c"
STOP_BG        = "#c62828"   # red when a request is active
STOP_BG_ACTIVE = "#e53935"
BTN_OFF_BG     = "#4a4a4a"   # greyed out
BTN_OFF_FG     = "#8a8a8a"

DEFAULT_ASSISTANT = {"name": "assistant", "system": "You are a helpful assistant."}
NONE_ASSISTANT = "none"          # fixed dropdown entry: sends no system message
PROVIDER_PLACEHOLDER = "click add provider"
MODEL_PLACEHOLDER = "click add model"
MONO = ("Consolas", 10)


def ascii_only(s):
    """The chat display is plain-text ASCII only."""
    if s is None:
        return ""
    return str(s).encode("ascii", "replace").decode("ascii")


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Small dark-widget helpers
# ---------------------------------------------------------------------------
def dark_button(parent, text, command):
    return tk.Button(parent, text=text, command=command, bg=BG_WIDGET, fg=FG,
                     activebackground=ACCENT, activeforeground="#ffffff",
                     relief="flat", padx=8, pady=2)


def dark_label(parent, text):
    return tk.Label(parent, text=text, bg=BG, fg=FG)


def dark_entry(parent, width=40):
    return tk.Entry(parent, bg=BG_WIDGET, fg=FG, insertbackground=FG,
                    relief="flat", width=width,
                    disabledbackground="#232323", disabledforeground="#6f6f6f")


def dark_listbox(parent, width=None, height=None):
    kw = dict(bg=BG_WIDGET, fg=FG, selectbackground=ACCENT,
              selectforeground="#ffffff", relief="flat",
              highlightthickness=0, exportselection=False, activestyle="none")
    if width:
        kw["width"] = width
    if height:
        kw["height"] = height
    return tk.Listbox(parent, **kw)


# ---------------------------------------------------------------------------
# Streaming think-tag parser
# ---------------------------------------------------------------------------
class ThinkParser:
    """Split a streamed model response into (kind, text) events.

    Some reasoning models embed their chain-of-thought directly in the
    content stream, wrapped in a pair of "think" tags. The exact tag
    strings are the OPEN and CLOSE class attributes below; they are
    built by concatenation so that no literal tag text ever appears in
    this source file.

    Because streaming delivers arbitrary chunk boundaries, a tag can be
    split across two chunks (the first chunk may end with only part of
    OPEN, the remainder arriving in the next chunk). To handle that,
    feed() keeps a short tail of un-emitted text in the buffer -- at
    most len(tag) - 1 characters -- until the next chunk arrives.
    Call flush() at end-of-stream to emit whatever remains buffered.
    """

    # Concatenated so no literal tag string appears in the source.
    OPEN = "<" + "think" + ">"
    CLOSE = "<" + "/" + "think" + ">"

    def __init__(self):
        self.in_think = False
        self.buf = ""

    def feed(self, chunk):
        self.buf += chunk
        events = []
        while True:
            tag = self.CLOSE if self.in_think else self.OPEN
            kind = "reasoning" if self.in_think else "content"
            idx = self.buf.find(tag)
            if idx == -1:
                # No complete tag in the buffer; emit everything except a
                # tail that could be the start of a tag split across chunks.
                safe = len(self.buf) - (len(tag) - 1)
                if safe > 0:
                    events.append((kind, self.buf[:safe]))
                    self.buf = self.buf[safe:]
                break
            if idx > 0:
                events.append((kind, self.buf[:idx]))
            self.buf = self.buf[idx + len(tag):]
            self.in_think = not self.in_think
        return events

    def flush(self):
        if not self.buf:
            return []
        kind = "reasoning" if self.in_think else "content"
        out, self.buf = [(kind, self.buf)], ""
        return out


# ---------------------------------------------------------------------------
# API worker (runs in a background thread)
# ---------------------------------------------------------------------------
def _delta_parts(delta):
    reasoning = ""
    for key in ("reasoning_content", "reasoning", "thinking"):
        val = delta.get(key)
        if isinstance(val, str) and val:
            reasoning += val
    content = delta.get("content") or ""
    return reasoning, content


def api_worker(url, api_key, payload, q, cancel, ctrl):
    """POSTs to /chat/completions with stream=True and pushes
    ('reasoning'|'content'|'error'|'stopped'|'done', text) events onto q.
    Setting the `cancel` event aborts the request; any partial progress
    is simply whatever was already emitted."""
    try:
        if requests is None:
            q.put(("error", "The 'requests' package is not installed. Run: pip install requests"))
            return
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        with requests.post(url, headers=headers, json=payload,
                           stream=True, timeout=(15, 600)) as resp:
            ctrl["response"] = resp
            ctype = resp.headers.get("Content-Type", "")
            if resp.status_code != 200:
                q.put(("error", f"HTTP {resp.status_code}: {resp.text[:800]}"))
                return
            if "text/event-stream" not in ctype:
                # provider ignored stream=True and returned one JSON blob
                obj = resp.json()
                msg = (obj.get("choices") or [{}])[0].get("message") or {}
                r, c = _delta_parts(msg)
                if r:
                    q.put(("reasoning", r))
                if c:
                    q.put(("content", c))
                q.put(("stopped" if cancel.is_set() else "done", None))
                return
            for raw in resp.iter_lines(decode_unicode=True):
                if cancel.is_set():
                    q.put(("stopped", None))
                    return
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or choices[0].get("message") or {}
                r, c = _delta_parts(delta)
                if r:
                    q.put(("reasoning", r))
                if c:
                    q.put(("content", c))
            q.put(("stopped" if cancel.is_set() else "done", None))
    except Exception as exc:
        if cancel.is_set():
            q.put(("stopped", None))
        else:
            q.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        ctrl["response"] = None


# ---------------------------------------------------------------------------
# Message building (OpenAI format)
# ---------------------------------------------------------------------------
def text_attachment_block(attachments):
    blocks = []
    for a in attachments:
        if a.get("type") == "text":
            blocks.append(f"[File: {a.get('name', 'file')}]\n{a.get('text', '')}")
    return "\n\n".join(blocks)


def media_attachment_parts(attachments):
    parts = []
    for a in attachments:
        t = a.get("type")
        if t == "image":
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:{a.get('mime', 'image/png')};base64,{a['data_b64']}"}})
        elif t == "audio":
            parts.append({"type": "input_audio",
                          "input_audio": {"data": a["data_b64"],
                                          "format": a.get("audio_format", "wav")}})
        elif t == "video":
            # OpenAI has no official video part; "video_url" is used by several
            # OpenAI-compatible providers (OpenRouter, Qwen, ...). Support varies.
            parts.append({"type": "video_url",
                          "video_url": {"url": f"data:{a.get('mime', 'video/mp4')};base64,{a['data_b64']}"}})
    return parts


def build_user_message(text, attachments):
    """Text attachments go after the system message but before the typed text."""
    file_text = text_attachment_block(attachments)
    full_text = (file_text + "\n\n" + text).strip() if file_text else text
    media = media_attachment_parts(attachments)
    if media:
        parts = list(media)
        if full_text:
            parts.append({"type": "text", "text": full_text})
        return {"role": "user", "content": parts}
    return {"role": "user", "content": full_text}


def build_api_messages(chat, include_system):
    """System message is included only once per chat (to save context)."""
    msgs = []
    if include_system and chat.get("system_prompt"):
        msgs.append({"role": "system", "content": chat["system_prompt"]})
    for m in chat.get("messages", []):
        if m["role"] == "user":
            msgs.append(build_user_message(m.get("content", ""), m.get("attachments", [])))
        elif m["role"] == "assistant":
            msgs.append({"role": "assistant", "content": m.get("content", "")})
    return msgs


# ---------------------------------------------------------------------------
# Attachment type chooser dialog
# ---------------------------------------------------------------------------
def ask_file_type(parent):
    win = tk.Toplevel(parent)
    win.title("Attachment type")
    win.configure(bg=BG)
    win.resizable(False, False)
    result = {"value": None}
    tk.Label(win, text="What kind of file(s) are you attaching?",
             bg=BG, fg=FG, padx=20, pady=12).pack()
    row = tk.Frame(win, bg=BG)
    row.pack(padx=12, pady=(0, 12))

    def choose(v):
        result["value"] = v
        win.destroy()

    for label, val in (("Text", "text"), ("Image", "image"),
                       ("Video", "video"), ("Audio", "audio")):
        dark_button(row, label, lambda v=val: choose(v)).pack(side="left", padx=4)
    win.transient(parent.winfo_toplevel())
    win.grab_set()
    win.protocol("WM_DELETE_WINDOW", win.destroy)
    parent.winfo_toplevel().wait_window(win)
    return result["value"]


# ---------------------------------------------------------------------------
# Setup screen: Providers & Models tab
# ---------------------------------------------------------------------------
class ProvidersTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=8, pady=8)

        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="y")
        dark_label(left, "Providers").pack(anchor="w")
        self.plist = dark_listbox(left, width=28)
        self.plist.pack(fill="y", expand=True, pady=4)
        self.plist.bind("<<ListboxSelect>>", self.on_provider_select)
        btns = tk.Frame(left, bg=BG)
        btns.pack(fill="x")
        dark_button(btns, "Add", self.add_provider).pack(side="left", expand=True, fill="x", padx=1)
        dark_button(btns, "Save", self.save_provider).pack(side="left", expand=True, fill="x", padx=1)
        dark_button(btns, "Delete", self.delete_provider).pack(side="left", expand=True, fill="x", padx=1)

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        form = tk.Frame(right, bg=BG)
        form.pack(fill="x")
        dark_label(form, "Friendly name:").grid(row=0, column=0, sticky="w")
        self.p_name = dark_entry(form, 46)
        self.p_name.grid(row=0, column=1, sticky="w", pady=2)
        dark_label(form, "Base URL (e.g. https://api.openai.com/v1):").grid(row=1, column=0, sticky="w")
        self.p_url = dark_entry(form, 46)
        self.p_url.grid(row=1, column=1, sticky="w", pady=2)
        dark_label(form, "API key (optional):").grid(row=2, column=0, sticky="w")
        self.p_key = dark_entry(form, 46)
        self.p_key.grid(row=2, column=1, sticky="w", pady=2)

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)
        dark_label(right, "Models for selected provider").pack(anchor="w")
        mid = tk.Frame(right, bg=BG)
        mid.pack(fill="both", expand=True)
        self.mlist = dark_listbox(mid, width=36)
        self.mlist.pack(side="left", fill="both", expand=True, pady=4)
        self.mlist.bind("<<ListboxSelect>>", self.on_model_select)
        mform = tk.Frame(mid, bg=BG)
        mform.pack(side="left", fill="y", padx=(12, 0))
        dark_label(mform, "Friendly name:").pack(anchor="w")
        self.m_name = dark_entry(mform, 34)
        self.m_name.pack(anchor="w", pady=2)
        dark_label(mform, "Model identifier (sent to the API; may be blank):").pack(anchor="w")
        self.m_id = dark_entry(mform, 34)
        self.m_id.pack(anchor="w", pady=2)
        dark_button(mform, "Add model", self.add_model).pack(fill="x", pady=(10, 2))
        dark_button(mform, "Save model", self.save_model).pack(fill="x", pady=2)
        dark_button(mform, "Delete model", self.delete_model).pack(fill="x", pady=2)

        # everything starts greyed out until the user picks/adds things
        self._set_provider_form(False)
        self._set_model_form(False)
        self.mlist.configure(state="disabled")

    # ---- enable/disable helpers (greyed-out placeholders) ----
    @staticmethod
    def _set_entry(entry, enabled, placeholder=""):
        entry.configure(state="normal")
        entry.delete(0, "end")
        if not enabled:
            if placeholder:
                entry.insert(0, placeholder)
            entry.configure(state="disabled")

    def _set_provider_form(self, enabled):
        self._set_entry(self.p_name, enabled, PROVIDER_PLACEHOLDER)
        self._set_entry(self.p_url, enabled)
        self._set_entry(self.p_key, enabled)

    def _set_model_form(self, enabled):
        self._set_entry(self.m_name, enabled, MODEL_PLACEHOLDER)
        self._set_entry(self.m_id, enabled)

    # ---- providers ----
    def refresh(self):
        self.plist.delete(0, "end")
        for p in self.app.providers["providers"]:
            self.plist.insert("end", p.get("name", "(unnamed)"))
        self.mlist.delete(0, "end")
        self.mlist.configure(state="disabled")
        self._set_provider_form(False)
        self._set_model_form(False)

    def _pidx(self):
        sel = self.plist.curselection()
        return sel[0] if sel else None

    def on_provider_select(self, _=None):
        i = self._pidx()
        if i is None:
            return
        p = self.app.providers["providers"][i]
        self._set_provider_form(True)
        for entry, key in ((self.p_name, "name"), (self.p_url, "url"), (self.p_key, "api_key")):
            entry.delete(0, "end")
            entry.insert(0, p.get(key, ""))
        self.mlist.configure(state="normal")
        self.mlist.delete(0, "end")
        for m in p.get("models", []):
            self.mlist.insert("end", model_display(m))
        self.mlist.selection_clear(0, "end")
        self._set_model_form(False)

    def add_provider(self):
        self.app.providers["providers"].append(
            {"name": "New Provider", "url": "", "api_key": "", "models": []})
        self.app.save_providers()
        self.refresh()
        self.plist.selection_set("end")
        self.on_provider_select()

    def save_provider(self):
        i = self._pidx()
        if i is None:
            messagebox.showinfo("Providers", "Select a provider first.")
            return
        p = self.app.providers["providers"][i]
        p["name"] = self.p_name.get().strip() or p.get("name", "")
        p["url"] = self.p_url.get().strip()
        p["api_key"] = self.p_key.get().strip()
        self.app.save_providers()
        self.refresh()
        self.plist.selection_set(i)
        self.on_provider_select()

    def delete_provider(self):
        i = self._pidx()
        if i is None:
            return
        p = self.app.providers["providers"][i]
        if messagebox.askyesno("Providers", f"Delete provider '{p.get('name')}' and its models?"):
            del self.app.providers["providers"][i]
            self.app.save_providers()
            self.refresh()

    # ---- models ----
    def _midx(self):
        sel = self.mlist.curselection()
        return sel[0] if sel else None

    def on_model_select(self, _=None):
        pi, mi = self._pidx(), self._midx()
        if pi is None or mi is None:
            return
        m = self.app.providers["providers"][pi].get("models", [])[mi]
        self._set_model_form(True)
        self.m_name.delete(0, "end")
        self.m_name.insert(0, m.get("name", ""))
        self.m_id.delete(0, "end")
        self.m_id.insert(0, m.get("model_id", ""))

    def add_model(self):
        pi = self._pidx()
        if pi is None:
            messagebox.showinfo("Models", "Select a provider first.")
            return
        self.app.providers["providers"][pi].setdefault("models", []).append(
            {"name": "New Model", "model_id": ""})
        self.app.save_providers()
        self.on_provider_select()
        self.mlist.selection_set("end")
        self.on_model_select()

    def save_model(self):
        pi, mi = self._pidx(), self._midx()
        if pi is None or mi is None:
            messagebox.showinfo("Models", "Select a model first.")
            return
        m = self.app.providers["providers"][pi]["models"][mi]
        m["name"] = self.m_name.get().strip() or m.get("name", "")
        m["model_id"] = self.m_id.get().strip()   # blank identifier is allowed
        self.app.save_providers()
        self.on_provider_select()
        self.mlist.selection_set(mi)
        self.on_model_select()

    def delete_model(self):
        pi, mi = self._pidx(), self._midx()
        if pi is None or mi is None:
            return
        del self.app.providers["providers"][pi]["models"][mi]
        self.app.save_providers()
        self.on_provider_select()


def model_display(m):
    """How a model appears in lists/dropdowns: friendly name, with the
    identifier appended in parentheses only when it is not blank."""
    name = m.get("name", "")
    mid = m.get("model_id", "")
    return f"{name}  ({mid})" if mid else name


# ---------------------------------------------------------------------------
# Setup screen: Assistants tab
# ---------------------------------------------------------------------------
class AssistantsTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=8, pady=8)

        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="y")
        dark_label(left, "Assistants / Characters").pack(anchor="w")
        self.alist = dark_listbox(left, width=28)
        self.alist.pack(fill="y", expand=True, pady=4)
        self.alist.bind("<<ListboxSelect>>", self.on_select)
        btns = tk.Frame(left, bg=BG)
        btns.pack(fill="x")
        dark_button(btns, "Add", self.add).pack(side="left", expand=True, fill="x", padx=1)
        dark_button(btns, "Save", self.save).pack(side="left", expand=True, fill="x", padx=1)
        dark_button(btns, "Delete", self.delete).pack(side="left", expand=True, fill="x", padx=1)

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        dark_label(right, "Name:").pack(anchor="w")
        self.a_name = dark_entry(right, 44)
        self.a_name.pack(anchor="w", pady=2)
        dark_label(right, "System message:").pack(anchor="w", pady=(8, 0))
        self.a_sys = tk.Text(right, wrap="word", bg=BG_WIDGET, fg=FG,
                             insertbackground=FG, relief="flat", height=12, font=MONO)
        self.a_sys.pack(fill="both", expand=True, pady=2)

    def refresh(self):
        self.alist.delete(0, "end")
        for a in self.app.assistants["assistants"]:
            self.alist.insert("end", a.get("name", ""))

    def _idx(self):
        sel = self.alist.curselection()
        return sel[0] if sel else None

    def on_select(self, _=None):
        i = self._idx()
        if i is None:
            return
        a = self.app.assistants["assistants"][i]
        self.a_name.delete(0, "end")
        self.a_name.insert(0, a.get("name", ""))
        self.a_sys.delete("1.0", "end")
        self.a_sys.insert("1.0", a.get("system", ""))

    def add(self):
        self.app.assistants["assistants"].append({"name": "New Assistant", "system": ""})
        self.app.save_assistants()
        self.refresh()
        self.alist.selection_set("end")
        self.on_select()

    def save(self):
        i = self._idx()
        if i is None:
            messagebox.showinfo("Assistants", "Select an assistant first.")
            return
        a = self.app.assistants["assistants"][i]
        new_name = self.a_name.get().strip()
        if a.get("name") == "assistant" and new_name != "assistant":
            messagebox.showinfo("Assistants", "The default 'assistant' cannot be renamed.")
            new_name = "assistant"
        if not new_name:
            messagebox.showinfo("Assistants", "Name cannot be empty.")
            return
        a["name"] = new_name
        a["system"] = self.a_sys.get("1.0", "end-1c")
        self.app.save_assistants()
        self.refresh()
        self.alist.selection_set(i)

    def delete(self):
        i = self._idx()
        if i is None:
            return
        a = self.app.assistants["assistants"][i]
        if a.get("name") == "assistant":
            messagebox.showinfo("Assistants", "The default 'assistant' cannot be deleted.")
            return
        if messagebox.askyesno("Assistants", f"Delete '{a.get('name')}'?"):
            del self.app.assistants["assistants"][i]
            self.app.save_assistants()
            self.refresh()


class SetupFrame(tk.Frame):
    def __init__(self, app):
        super().__init__(app, bg=BG)
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=8, pady=6)
        tk.Label(top, text="Setup", bg=BG, fg=FG,
                 font=("Segoe UI", 14, "bold")).pack(side="left")
        dark_button(top, "Back to Chat", app.show_chat).pack(side="right")
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=6)
        self.providers_tab = ProvidersTab(nb, app)
        self.assistants_tab = AssistantsTab(nb, app)
        nb.add(self.providers_tab, text=" Providers && Models ")
        nb.add(self.assistants_tab, text=" Assistants / Characters ")

    def refresh_all(self):
        self.providers_tab.refresh()
        self.assistants_tab.refresh()


# ---------------------------------------------------------------------------
# Chat screen
# ---------------------------------------------------------------------------
class ChatFrame(tk.Frame):
    def __init__(self, app):
        super().__init__(app, bg=BG)
        self.app = app
        self.chat = None            # currently open chat dict (None = fresh, unsaved)
        self.chat_file = None       # path of current chat file
        self.attachments = []       # pending attachments for the next send
        self.q = queue.Queue()
        self._busy = False
        self._busy_dots = 0
        self._busy_job = None
        self._busy_line = None      # busy mark active while waiting
        self._think_ph_active = False   # "Thinking" placeholder (Hide Thinking mode)
        self._think_ph_dots = 0
        self._think_ph_job = None
        self._cur_reasoning = ""
        self._cur_content = ""
        self._think_parser = None
        self._include_system = False
        self._cancel = threading.Event()
        self._ctrl = {"response": None}
        self._list_paths = []       # listbox index -> chat file path

        self._build_widgets()
        self.refresh_selectors()
        self.refresh_chat_list()
        self.show_new_chat()

    # ---------------- UI construction ----------------
    def _build_widgets(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ---- top bar ----
        top = tk.Frame(self, bg=BG_PANEL)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        dark_button(top, "Setup", self.app.show_setup).pack(side="left", padx=4, pady=4)

        def lab(txt):
            tk.Label(top, text=txt, bg=BG_PANEL, fg=FG).pack(side="left", padx=(6, 2))

        lab("Provider:")
        self.provider_var = tk.StringVar()
        self.provider_cb = ttk.Combobox(top, textvariable=self.provider_var,
                                        state="readonly", width=14)
        self.provider_cb.pack(side="left")
        self.provider_cb.bind("<<ComboboxSelected>>", self.on_provider_change)

        lab("Model:")
        self.model_var = tk.StringVar()
        self.model_cb = ttk.Combobox(top, textvariable=self.model_var,
                                     state="readonly", width=24)
        self.model_cb.pack(side="left")
        self.model_cb.bind("<<ComboboxSelected>>", lambda e: self.persist_settings())

        lab("Assistant:")
        self.assistant_var = tk.StringVar()
        self.assistant_cb = ttk.Combobox(top, textvariable=self.assistant_var,
                                         state="readonly", width=14)
        self.assistant_cb.pack(side="left")
        self.assistant_cb.bind("<<ComboboxSelected>>", lambda e: self.persist_settings())

        lab("Temp:")
        self.temp_var = tk.StringVar(value="1.0")
        tk.Entry(top, textvariable=self.temp_var, width=5, bg=BG_WIDGET, fg=FG,
                 insertbackground=FG, relief="flat").pack(side="left")
        lab("Top P:")
        self.topp_var = tk.StringVar(value="1.0")
        tk.Entry(top, textvariable=self.topp_var, width=5, bg=BG_WIDGET, fg=FG,
                 insertbackground=FG, relief="flat").pack(side="left")

        self.hide_thinking_var = tk.BooleanVar(value=False)
        self.hide_thinking_cb = tk.Checkbutton(
            top, text="Hide Thinking", variable=self.hide_thinking_var,
            bg=BG_PANEL, fg=FG, selectcolor=BG_WIDGET,
            activebackground=BG_PANEL, activeforeground=FG,
            command=self.on_hide_thinking_toggle)
        self.hide_thinking_cb.pack(side="left", padx=(12, 2))

        # ---- left sidebar ----
        side = tk.Frame(self, bg=BG_PANEL, width=250)
        side.grid(row=1, column=0, sticky="ns", padx=(4, 2), pady=(0, 4))
        side.grid_propagate(False)

        srow = tk.Frame(side, bg=BG_PANEL)
        srow.pack(fill="x", padx=4, pady=(4, 0))
        self.search_var = tk.StringVar()
        se = tk.Entry(srow, textvariable=self.search_var, bg=BG_WIDGET, fg=FG,
                      insertbackground=FG, relief="flat")
        se.pack(side="left", fill="x", expand=True)
        se.bind("<Return>", lambda e: self.search_chats())
        dark_button(srow, "Search", self.search_chats).pack(side="left", padx=(2, 0))
        dark_button(side, "Show all chats", self.refresh_chat_list).pack(fill="x", padx=4, pady=2)

        self.chat_list = dark_listbox(side)
        self.chat_list.pack(fill="both", expand=True, padx=4, pady=4)
        self.chat_list.bind("<<ListboxSelect>>", self.on_chat_select)

        b1 = tk.Frame(side, bg=BG_PANEL)
        b1.pack(fill="x", padx=4, pady=(0, 2))
        dark_button(b1, "New chat", self.show_new_chat).pack(side="left", expand=True, fill="x", padx=1)
        dark_button(b1, "Archive...", self.archive_chat).pack(side="left", expand=True, fill="x", padx=1)
        b2 = tk.Frame(side, bg=BG_PANEL)
        b2.pack(fill="x", padx=4, pady=(0, 4))
        dark_button(b2, "Import...", self.import_chat).pack(side="left", expand=True, fill="x", padx=1)
        dark_button(b2, "Delete", self.delete_chat).pack(side="left", expand=True, fill="x", padx=1)

        # ---- main column ----
        main = tk.Frame(self, bg=BG)
        main.grid(row=1, column=1, sticky="nsew", padx=(2, 4), pady=(0, 4))
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        holder = tk.Frame(main, bg=BG)
        holder.grid(row=0, column=0, sticky="nsew")
        self.chat_text = tk.Text(holder, wrap="word", bg=CHAT_BG, fg=FG,
                                 insertbackground=FG, relief="flat",
                                 state="disabled", font=MONO,
                                 spacing1=2, spacing3=2)
        sb = tk.Scrollbar(holder, command=self.chat_text.yview)
        self.chat_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.chat_text.pack(side="left", fill="both", expand=True)
        self.chat_text.tag_configure("user", background=USER_BG, foreground=USER_FG)
        self.chat_text.tag_configure("user_hdr", background=USER_BG, foreground=USER_FG,
                                     font=(MONO[0], MONO[1], "bold"), spacing1=10)
        self.chat_text.tag_configure("assistant_hdr", foreground="#7cc7ff",
                                     font=(MONO[0], MONO[1], "bold"), spacing1=8)
        self.chat_text.tag_configure("think", background=THINK_BG, foreground="#d8d8d8",
                                     font=(MONO[0], MONO[1], "italic"))
        self.chat_text.tag_configure("info", foreground=FG_DIM,
                                     font=(MONO[0], 9, "italic"))
        self.chat_text.tag_configure("error", foreground=ERR_FG)
        self.chat_text.tag_configure("busy", foreground=FG_DIM,
                                     font=(MONO[0], MONO[1], "italic"))

        bottom = tk.Frame(main, bg=BG_PANEL)
        bottom.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.input_text = tk.Text(bottom, height=5, wrap="word", bg=BG_WIDGET, fg=FG,
                                  insertbackground=FG, relief="flat", font=MONO)
        self.input_text.pack(fill="x", padx=4, pady=(4, 2))
        arow = tk.Frame(bottom, bg=BG_PANEL)
        arow.pack(fill="x", padx=4)
        # Stop is packed first (side="right") so it ends up to the RIGHT of Send.
        self.stop_btn = tk.Button(arow, text="Stop", command=self.on_stop,
                                  bg=BTN_OFF_BG, fg="#ffffff",
                                  activebackground=STOP_BG_ACTIVE, activeforeground="#ffffff",
                                  disabledforeground=BTN_OFF_FG, relief="flat",
                                  padx=10, pady=2, state="disabled")
        self.stop_btn.pack(side="right", padx=2)
        self.send_btn = tk.Button(arow, text="Send", command=self.on_send,
                                  bg=SEND_BG, fg="#ffffff",
                                  activebackground=SEND_BG_ACTIVE, activeforeground="#ffffff",
                                  disabledforeground=BTN_OFF_FG, relief="flat",
                                  padx=10, pady=2)
        self.send_btn.pack(side="right", padx=2)
        dark_button(arow, "Attach files...", self.on_attach).pack(side="left", padx=2)
        dark_button(arow, "Remove attachment", self.on_remove_attachment).pack(side="left", padx=2)
        self.att_list = dark_listbox(bottom, height=3)
        self.att_list.pack(fill="x", padx=4, pady=(2, 4))

    def _set_busy_buttons(self, busy):
        """Send is green when idle and greyed while busy; Stop is the opposite."""
        if busy:
            self.send_btn.configure(state="disabled", bg=BTN_OFF_BG)
            self.stop_btn.configure(state="normal", bg=STOP_BG)
        else:
            self.send_btn.configure(state="normal", bg=SEND_BG)
            self.stop_btn.configure(state="disabled", bg=BTN_OFF_BG)

    # ---------------- selectors ----------------
    def refresh_selectors(self):
        providers = self.app.providers["providers"]
        names = [p.get("name", "") for p in providers]
        self.provider_cb["values"] = names
        s = self.app.settings
        pname = s.get("provider", "")
        if pname not in names:
            pname = names[0] if names else ""
        self.provider_var.set(pname)
        self._refresh_models(select_id=s.get("model", ""))
        # assistant dropdown: fixed "none" entry first (not in the setup list)
        anames = [NONE_ASSISTANT] + [a.get("name", "")
                                     for a in self.app.assistants["assistants"]
                                     if a.get("name") != NONE_ASSISTANT]
        self.assistant_cb["values"] = anames
        aname = s.get("assistant", NONE_ASSISTANT)
        if aname not in anames:
            aname = NONE_ASSISTANT
        self.assistant_var.set(aname)
        self.temp_var.set(str(s.get("temperature", "1.0")))
        self.topp_var.set(str(s.get("top_p", "1.0")))
        self.hide_thinking_var.set(bool(s.get("hide_thinking", False)))

    def _current_provider(self):
        for p in self.app.providers["providers"]:
            if p.get("name") == self.provider_var.get():
                return p
        return None

    def _refresh_models(self, select_id=None):
        p = self._current_provider()
        models = p.get("models", []) if p else []
        self.model_cb["values"] = [model_display(m) for m in models]
        chosen = ""
        if select_id:
            for m in models:
                if m.get("model_id") == select_id:
                    chosen = model_display(m)
                    break
        if not chosen and models:
            chosen = model_display(models[0])
        self.model_var.set(chosen)

    def _current_model(self):
        p = self._current_provider()
        if not p:
            return None
        idx = self.model_cb.current()
        models = p.get("models", [])
        if 0 <= idx < len(models):
            return models[idx]
        return None

    def _current_assistant(self):
        """Returns None for the fixed 'none' entry (no system message)."""
        name = self.assistant_var.get()
        if name == NONE_ASSISTANT:
            return None
        for a in self.app.assistants["assistants"]:
            if a.get("name") == name:
                return a
        return None

    def on_provider_change(self, _=None):
        self._refresh_models()
        self.persist_settings()

    def on_hide_thinking_toggle(self):
        self.persist_settings()
        # if thinking was being hidden mid-stream, close out the placeholder
        if not self.hide_thinking_var.get() and self._think_ph_active:
            self._finalize_thinking_placeholder()

    def persist_settings(self):
        m = self._current_model()
        self.app.settings.update({
            "provider": self.provider_var.get(),
            "model": m.get("model_id") if m else "",
            "assistant": self.assistant_var.get(),
            "temperature": self.temp_var.get(),
            "top_p": self.topp_var.get(),
            "hide_thinking": bool(self.hide_thinking_var.get()),
        })
        self.app.save_settings()

    # ---------------- chat list ----------------
    def refresh_chat_list(self):
        self.chat_list.delete(0, "end")
        self._list_paths = []
        files = []
        try:
            for fn in os.listdir(CHATS_DIR):
                if fn.lower().endswith(".json"):
                    path = os.path.join(CHATS_DIR, fn)
                    files.append((os.path.getmtime(path), path))
        except FileNotFoundError:
            pass
        files.sort(reverse=True)
        for _, path in files:
            data = load_json(path, None)
            if not isinstance(data, dict):
                continue
            self._list_paths.append(path)
            self.chat_list.insert("end", ascii_only(data.get("title") or os.path.basename(path)))
        if self.chat_file and self.chat_file in self._list_paths:
            i = self._list_paths.index(self.chat_file)
            self.chat_list.selection_set(i)
            self.chat_list.see(i)

    def search_chats(self):
        query = self.search_var.get().strip().lower()
        if not query:
            self.refresh_chat_list()
            return
        self.chat_list.delete(0, "end")
        self._list_paths = []
        try:
            fns = [fn for fn in os.listdir(CHATS_DIR) if fn.lower().endswith(".json")]
        except FileNotFoundError:
            fns = []
        paths = sorted((os.path.join(CHATS_DIR, fn) for fn in fns),
                       key=lambda p: os.path.getmtime(p), reverse=True)
        for path in paths:
            data = load_json(path, None)
            if not isinstance(data, dict):
                continue
            hay = [str(data.get("title", ""))]
            for m in data.get("messages", []):
                hay.append(str(m.get("content", "")))
                hay.append(str(m.get("reasoning", "")))
                for a in m.get("attachments", []):
                    hay.append(a.get("name", ""))
                    if a.get("type") == "text":
                        hay.append(a.get("text", ""))
            if query in "\n".join(hay).lower():
                self._list_paths.append(path)
                self.chat_list.insert("end", ascii_only(data.get("title") or os.path.basename(path)))

    def _selected_path(self):
        sel = self.chat_list.curselection()
        return self._list_paths[sel[0]] if sel else None

    # ---------------- chat open / new ----------------
    def show_new_chat(self):
        if self._busy:
            messagebox.showinfo("Busy", "Please wait for the current response to finish.")
            return
        self.chat = None
        self.chat_file = None
        self._clear_display()
        self._insert_line("New chat. Pick a provider, model and assistant above, then type below.",
                          "info")

    def on_chat_select(self, _=None):
        if self._busy:
            messagebox.showinfo("Busy", "Please wait for the current response to finish.")
            return
        path = self._selected_path()
        if not path:
            return
        data = load_json(path, None)
        if not isinstance(data, dict):
            messagebox.showerror("Chat", "Could not read that chat file.")
            return
        self.chat = data
        self.chat.setdefault("messages", [])
        self.chat_file = path
        self.render_chat()

    def render_chat(self):
        self._clear_display()
        model_disp = self.chat.get("model_name") or self.chat.get("model", "")
        self._insert_line(f"Chat: {self.chat.get('title', 'Chat')}    "
                          f"[model: {model_disp}]", "info")
        # note: system messages are never shown in the display area
        for m in self.chat.get("messages", []):
            self.render_message(m)

    # ---------------- rendering helpers ----------------
    def _clear_display(self):
        self.chat_text.configure(state="normal")
        self.chat_text.delete("1.0", "end")
        self.chat_text.configure(state="disabled")

    def _is_at_bottom(self):
        """True when the user is already viewing the last few lines; only
        then do we auto-scroll during streaming (so the user can scroll up
        and read while the response grows)."""
        try:
            if self.chat_text.yview()[1] >= 0.99:
                return True
            return self.chat_text.dlineinfo("end-1c") is not None
        except tk.TclError:
            return True

    def _insert_line(self, text, tag=None):
        self._append(text + "\n", tag)

    def _append(self, text, tag=None, guard_scroll=False):
        """Append display text (ASCII only). With guard_scroll=True the view
        only follows the text if the user is already at the bottom."""
        follow = (not guard_scroll) or self._is_at_bottom()
        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", ascii_only(text), (tag,) if tag else ())
        if follow:
            self.chat_text.see("end")
        self.chat_text.configure(state="disabled")

    def _attachment_note(self, attachments):
        if not attachments:
            return ""
        return "[Attached: " + ", ".join(
            f"{a.get('name', 'file')} ({a.get('type', 'file')})" for a in attachments) + "]"

    def render_message(self, m):
        if m.get("role") == "user":
            self._append("You:\n", "user_hdr")
            note = self._attachment_note(m.get("attachments", []))
            if note:
                self._append(note + "\n", "user")
            self._append(m.get("content", "") + "\n\n", "user")
        elif m.get("role") == "assistant":
            self._append("Assistant:\n", "assistant_hdr")
            reasoning = m.get("reasoning", "")
            if reasoning:
                if self.hide_thinking_var.get():
                    self._append("Hidden Thinking Block\n", "think")
                else:
                    self._append(reasoning + "\n", "think")
            self._append(m.get("content", "") + "\n\n", None)

    # ---------------- archive / import / delete ----------------
    def archive_chat(self):
        if self._busy:
            messagebox.showinfo("Busy", "Please wait for the current response to finish.")
            return
        path = self._selected_path()
        if not path:
            messagebox.showinfo("Archive", "Select a chat in the list first.")
            return
        folder = filedialog.askdirectory(title="Choose a folder to move this chat into")
        if not folder:
            return
        try:
            dest = os.path.join(folder, os.path.basename(path))
            if os.path.exists(dest):
                dest = os.path.join(folder, uuid.uuid4().hex[:6] + "_" + os.path.basename(path))
            shutil.move(path, dest)
        except OSError as exc:
            messagebox.showerror("Archive", str(exc))
            return
        if self.chat_file == path:
            self.chat = None
            self.chat_file = None
            self._clear_display()
        self.refresh_chat_list()

    def import_chat(self):
        if self._busy:
            messagebox.showinfo("Busy", "Please wait for the current response to finish.")
            return
        path = filedialog.askopenfilename(
            title="Import chat file",
            filetypes=[("Chat files", "*.json"), ("All files", "*.*")])
        if not path:
            return
        data = load_json(path, None)
        if not isinstance(data, dict) or "messages" not in data:
            messagebox.showerror("Import", "That file does not look like a saved chat.")
            return
        dest = os.path.join(CHATS_DIR, os.path.basename(path))
        if os.path.abspath(dest) != os.path.abspath(path):
            if os.path.exists(dest):
                dest = os.path.join(CHATS_DIR, uuid.uuid4().hex[:6] + "_" + os.path.basename(path))
            try:
                shutil.copy(path, dest)
            except OSError as exc:
                messagebox.showerror("Import", str(exc))
                return
        self.refresh_chat_list()
        if dest in self._list_paths:
            i = self._list_paths.index(dest)
            self.chat_list.selection_clear(0, "end")
            self.chat_list.selection_set(i)
            self.on_chat_select()

    def delete_chat(self):
        if self._busy:
            messagebox.showinfo("Busy", "Please wait for the current response to finish.")
            return
        path = self._selected_path()
        if not path:
            messagebox.showinfo("Delete", "Select a chat in the list first.")
            return
        if not messagebox.askyesno("Delete", f"Permanently delete '{os.path.basename(path)}'?"):
            return
        try:
            os.remove(path)
        except OSError as exc:
            messagebox.showerror("Delete", str(exc))
            return
        if self.chat_file == path:
            self.chat = None
            self.chat_file = None
            self._clear_display()
        self.refresh_chat_list()

    # ---------------- attachments ----------------
    def on_attach(self):
        paths = filedialog.askopenfilenames(title="Select one or more files")
        if not paths:
            return
        ftype = ask_file_type(self)
        if not ftype:
            return
        for path in paths:
            name = os.path.basename(path)
            try:
                if ftype == "text":
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        content = fh.read()
                    att = {"type": "text", "name": name, "text": content}
                    size = len(content)
                else:
                    with open(path, "rb") as fh:
                        raw = fh.read()
                    att = {"type": ftype, "name": name,
                           "mime": mimetypes.guess_type(path)[0] or "application/octet-stream",
                           "data_b64": base64.b64encode(raw).decode("ascii")}
                    if ftype == "audio":
                        att["audio_format"] = os.path.splitext(name)[1].lstrip(".").lower() or "wav"
                    size = len(raw)
            except OSError as exc:
                messagebox.showerror("Attach", f"{name}: {exc}")
                continue
            self.attachments.append(att)
            self.att_list.insert("end", f"{name}  [{ftype}]  ({size / 1024:.1f} KB)")

    def on_remove_attachment(self):
        for i in reversed(self.att_list.curselection()):
            self.att_list.delete(i)
            del self.attachments[i]

    # ---------------- send / streaming ----------------
    def on_send(self):
        if self._busy:
            return
        text = self.input_text.get("1.0", "end-1c")
        if not text.strip() and not self.attachments:
            return
        provider = self._current_provider()
        model = self._current_model()
        assistant = self._current_assistant()
        if not provider or not provider.get("url"):
            messagebox.showwarning("Send", "Please configure and select a provider (Setup screen).")
            return
        if model is None:
            messagebox.showwarning("Send", "Please select a model (Setup screen).")
            return
        model_id = model.get("model_id", "") or ""   # blank identifier is allowed
        try:
            temperature = float(self.temp_var.get())
            top_p = float(self.topp_var.get())
        except ValueError:
            messagebox.showwarning("Send", "Temp and Top P must be numbers.")
            return

        # create the chat on first send
        if self.chat is None:
            now = datetime.datetime.now()
            self.chat = {"id": uuid.uuid4().hex[:8],
                         "title": "New Chat",
                         "created": now.isoformat(timespec="seconds"),
                         "provider": "", "model": "", "model_name": "",
                         "assistant": "",
                         "system_prompt": "", "system_sent": False,
                         "messages": []}
            self.chat_file = os.path.join(
                CHATS_DIR, f"chat_{now:%Y%m%d_%H%M%S}_{self.chat['id']}.json")

        chat = self.chat
        asst_name = assistant.get("name", "") if assistant else NONE_ASSISTANT
        sys_prompt = assistant.get("system", "") if assistant else ""
        # system prompt: sent once per chat (and once again if the assistant changes)
        if chat.get("assistant") != asst_name:
            chat["assistant"] = asst_name
            chat["system_prompt"] = sys_prompt
            chat["system_sent"] = False
        elif not chat.get("system_prompt"):
            chat["system_prompt"] = sys_prompt
        self._include_system = bool(sys_prompt) and (not chat.get("system_sent", False))

        user_msg = {"role": "user", "content": text, "attachments": self.attachments}
        chat["messages"].append(user_msg)
        chat["provider"] = provider.get("name", "")
        chat["model"] = model_id
        chat["model_name"] = model.get("name", "")
        if chat["title"] == "New Chat":
            base = text.strip() or (self.attachments[0]["name"] if self.attachments else "Chat")
            chat["title"] = " ".join(base.split())[:48] or "Chat"

        payload = {
            "model": model_id,
            "messages": build_api_messages(chat, self._include_system),
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        url = provider["url"].rstrip("/") + "/chat/completions"

        # Save immediately so the new chat shows up in the list on the left,
        # then render the user's message.
        self.save_current_chat()
        self.refresh_chat_list()
        self.render_message(user_msg)
        self.input_text.delete("1.0", "end")
        self.attachments = []
        self.att_list.delete(0, "end")
        self.persist_settings()

        self._cur_reasoning = ""
        self._cur_content = ""
        self._think_parser = ThinkParser()
        self._think_ph_active = False
        self._think_ph_job = None
        self._cancel = threading.Event()
        self._ctrl = {"response": None}
        self._start_busy()

        self.q = queue.Queue()
        threading.Thread(target=api_worker,
                         args=(url, provider.get("api_key", ""), payload,
                               self.q, self._cancel, self._ctrl),
                         daemon=True).start()
        self.after(80, self._poll_queue)

    def on_stop(self):
        """Abort the active request; partial progress is kept and saved."""
        if not self._busy:
            return
        self._cancel.set()
        resp = self._ctrl.get("response")
        if resp is not None:
            try:
                resp.close()   # unblocks the worker's stream read
            except Exception:
                pass

    # ---- busy indicator (in the main chat area) ----
    def _start_busy(self):
        self._busy = True
        self._busy_dots = 0
        self._set_busy_buttons(True)
        follow = self._is_at_bottom()
        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", "Assistant:\n", ("assistant_hdr",))
        self.chat_text.mark_set("busy_start", "end-1c")
        self.chat_text.mark_gravity("busy_start", "left")
        self.chat_text.insert("end", "[working]\n", ("busy",))
        if follow:
            self.chat_text.see("end")
        self.chat_text.configure(state="disabled")
        self._busy_line = "busy_start"
        self._animate_busy()

    def _animate_busy(self):
        if not self._busy:
            return
        self._busy_dots = (self._busy_dots + 1) % 4
        if self._busy_line:
            follow = self._is_at_bottom()
            self.chat_text.configure(state="normal")
            try:
                self.chat_text.delete("busy_start linestart", "busy_start lineend")
                self.chat_text.insert("busy_start linestart",
                                      "[working" + "." * self._busy_dots + "]", ("busy",))
            except tk.TclError:
                pass
            if follow:
                self.chat_text.see("end")
            self.chat_text.configure(state="disabled")
        self._busy_job = self.after(400, self._animate_busy)

    def _clear_busy_line(self):
        if not self._busy_line:
            return
        self.chat_text.configure(state="normal")
        try:
            self.chat_text.delete("busy_start linestart", "busy_start lineend+1c")
        except tk.TclError:
            pass
        self.chat_text.configure(state="disabled")
        self._busy_line = None

    # ---- "Hide Thinking" streaming placeholder ----
    def _display_reasoning_chunk(self, txt):
        if self.hide_thinking_var.get():
            if not self._think_ph_active:
                self._show_thinking_placeholder()
        else:
            self._append(txt, "think", guard_scroll=True)

    def _display_content_chunk(self, txt):
        if self._think_ph_active:
            self._finalize_thinking_placeholder()
        self._append(txt, None, guard_scroll=True)

    def _show_thinking_placeholder(self):
        follow = self._is_at_bottom()
        self.chat_text.configure(state="normal")
        self.chat_text.mark_set("think_ph", "end-1c")
        self.chat_text.mark_gravity("think_ph", "left")
        self.chat_text.insert("end", "Thinking\n", ("think",))
        if follow:
            self.chat_text.see("end")
        self.chat_text.configure(state="disabled")
        self._think_ph_active = True
        self._think_ph_dots = 0
        self._animate_thinking_ph()

    def _animate_thinking_ph(self):
        if not self._think_ph_active:
            return
        self._think_ph_dots = (self._think_ph_dots + 1) % 4
        follow = self._is_at_bottom()
        self.chat_text.configure(state="normal")
        try:
            self.chat_text.delete("think_ph linestart", "think_ph lineend")
            self.chat_text.insert("think_ph linestart",
                                  "Thinking" + "." * self._think_ph_dots, ("think",))
        except tk.TclError:
            pass
        if follow:
            self.chat_text.see("end")
        self.chat_text.configure(state="disabled")
        self._think_ph_job = self.after(400, self._animate_thinking_ph)

    def _finalize_thinking_placeholder(self):
        """Replace the animated 'Thinking' line with 'Hidden Thinking Block'."""
        if not self._think_ph_active:
            return
        self._think_ph_active = False
        if self._think_ph_job:
            try:
                self.after_cancel(self._think_ph_job)
            except Exception:
                pass
            self._think_ph_job = None
        self.chat_text.configure(state="normal")
        try:
            self.chat_text.delete("think_ph linestart", "think_ph lineend")
            self.chat_text.insert("think_ph linestart", "Hidden Thinking Block", ("think",))
        except tk.TclError:
            pass
        self.chat_text.configure(state="disabled")

    # ---- stream polling ----
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                self._handle_stream_event(kind, payload)
        except queue.Empty:
            pass
        if self._busy:
            self.after(80, self._poll_queue)

    def _drain_parser(self):
        if not self._think_parser:
            return
        for k2, txt in self._think_parser.flush():
            if k2 == "reasoning":
                self._cur_reasoning += txt
                self._display_reasoning_chunk(txt)
            else:
                self._cur_content += txt
                self._display_content_chunk(txt)

    def _handle_stream_event(self, kind, payload):
        if kind in ("reasoning", "content"):
            if self._busy_line:
                self._clear_busy_line()
            if kind == "reasoning":
                self._cur_reasoning += payload
                self._display_reasoning_chunk(payload)
            else:
                for k2, txt in self._think_parser.feed(payload):
                    if k2 == "reasoning":
                        self._cur_reasoning += txt
                        self._display_reasoning_chunk(txt)
                    else:
                        self._cur_content += txt
                        self._display_content_chunk(txt)
        elif kind == "error":
            self._clear_busy_line()
            self._finalize_thinking_placeholder()
            self._append("\n[Error] " + payload + "\n\n", "error", guard_scroll=True)
            self._finish_response(success=False)
        elif kind == "stopped":
            self._drain_parser()
            self._finalize_thinking_placeholder()
            self._append("\n[Stopped by user]\n", "info", guard_scroll=True)
            self._finish_response(success=True, stopped=True)
        elif kind == "done":
            self._drain_parser()
            self._finish_response(success=True)

    def _finish_response(self, success, stopped=False):
        self._clear_busy_line()
        self._finalize_thinking_placeholder()
        if self._busy_job:
            try:
                self.after_cancel(self._busy_job)
            except Exception:
                pass
            self._busy_job = None
        self._busy = False
        self._set_busy_buttons(False)
        if self.chat is not None:
            if success and (self._cur_content or self._cur_reasoning):
                # saves partial progress too (stopped=True arrives here as well)
                self.chat["messages"].append({
                    "role": "assistant",
                    "content": self._cur_content,
                    "reasoning": self._cur_reasoning,
                })
                if self._include_system:
                    self.chat["system_sent"] = True
                self._append("\n", None, guard_scroll=True)
            elif stopped and self._include_system:
                # the system prompt was transmitted even though no reply arrived
                self.chat["system_sent"] = True
            self.save_current_chat()
        self.refresh_chat_list()

    def save_current_chat(self):
        if self.chat and self.chat_file:
            save_json(self.chat_file, self.chat)


# ---------------------------------------------------------------------------
# Application shell
# ---------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1200x780")
        self.minsize(900, 600)
        self.configure(bg=BG)
        self._init_style()

        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(CHATS_DIR, exist_ok=True)

        self.providers = load_json(PROVIDERS_FILE, {"providers": []})
        if "providers" not in self.providers or not isinstance(self.providers["providers"], list):
            self.providers = {"providers": []}
        self.assistants = load_json(ASSISTANTS_FILE, {"assistants": []})
        if "assistants" not in self.assistants or not isinstance(self.assistants["assistants"], list):
            self.assistants = {"assistants": []}
        if not any(a.get("name") == "assistant" for a in self.assistants["assistants"]):
            self.assistants["assistants"].insert(0, dict(DEFAULT_ASSISTANT))
            save_json(ASSISTANTS_FILE, self.assistants)
        self.settings = load_json(SETTINGS_FILE, {})

        self.setup_frame = SetupFrame(self)
        self.chat_frame = ChatFrame(self)

        if self.providers["providers"]:
            self.show_chat()
        else:
            self.show_setup()   # first run: go straight to the setup screen

        if requests is None:
            messagebox.showwarning(
                "Missing dependency",
                "The 'requests' package is not installed.\nRun:  pip install requests")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _init_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TCombobox", fieldbackground=BG_WIDGET, background=BG_WIDGET,
                        foreground=FG, arrowcolor=FG)
        style.map("TCombobox", fieldbackground=[("readonly", BG_WIDGET)],
                  foreground=[("readonly", FG)], selectbackground=[("readonly", BG_WIDGET)])
        self.option_add("*TCombobox*Listbox*Background", BG_WIDGET)
        self.option_add("*TCombobox*Listbox*Foreground", FG)
        self.option_add("*TCombobox*Listbox*selectBackground", ACCENT)
        style.configure("TNotebook", background=BG)
        style.configure("TNotebook.Tab", background=BG_WIDGET, foreground=FG, padding=(10, 4))
        style.map("TNotebook.Tab", background=[("selected", ACCENT)],
                  foreground=[("selected", "#ffffff")])

    def save_providers(self):
        save_json(PROVIDERS_FILE, self.providers)

    def save_assistants(self):
        save_json(ASSISTANTS_FILE, self.assistants)

    def save_settings(self):
        save_json(SETTINGS_FILE, self.settings)

    def show_setup(self):
        self.chat_frame.pack_forget()
        self.setup_frame.refresh_all()
        self.setup_frame.pack(fill="both", expand=True)

    def show_chat(self):
        self.setup_frame.pack_forget()
        self.chat_frame.pack(fill="both", expand=True)
        self.chat_frame.refresh_selectors()
        self.chat_frame.refresh_chat_list()

    def on_close(self):
        try:
            self.chat_frame.persist_settings()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
