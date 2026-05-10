"""
iSee — a local screen-aware chat assistant.

A focused desktop app that runs Qwen 3.5:9b through Ollama with one
distinctive feature: a persistent 📷 toggle that lets you frame any
region of your screen — a window, a chart, a video game, an obscure
piece of software — and ask the assistant about it conversationally.

What this app does:
  * Local chat with Qwen 3.5:9b (no cloud, no API keys required)
  * Screen capture toggle: full screen / region / window
  * Custom prompts (saved presets, switch via dropdown)
  * Four themes (terminal / gold / cyan / blood)

What this app deliberately doesn't do:
  * Web search → use Perplexity, ChatGPT, etc. They're better at it.
  * Browser automation → use Operator, Browser Use, etc.
  * Anything cloud-coupled by default → screenshots and prompts
    stay on your machine.

Phase 6 will add an optional Anthropic API key + "Ask Claude" button
for one-click escalation to a frontier model when Qwen falls short.
For now: single brain, local only.

Qwen tuning:
  * /api/chat (NOT /api/generate — Ollama #14793 silently ignores
    `think: false` on /generate for qwen3.5:9b)
  * `think: false` at TOP level of payload, NOT inside options
  * temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5
  * num_ctx=8192 normally, 16384 when an image is attached
  * 600s hard timeout
  * <think>...</think> tags and ``` fences stripped on response

Run:    python ai_monitor.py
Deps:   pip install requests mss Pillow pygetwindow
Optional: pip install tkinterdnd2  (drag-and-drop image support)
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import scrolledtext, messagebox

try:
    import requests
except ImportError:
    print(
        "iSee requires the 'requests' package. "
        "Install with: pip install requests",
        file=sys.stderr,
    )
    sys.exit(2)

# Screen-capture deps. The app degrades gracefully when any are
# missing — capture features are disabled and a system line tells
# the user how to install them.
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None     # type: ignore
    HAS_PIL = False

try:
    import mss
    HAS_MSS = True
except ImportError:
    mss = None       # type: ignore
    HAS_MSS = False

try:
    import pygetwindow as gw
    HAS_PYGETWINDOW = True
except ImportError:
    gw = None        # type: ignore
    HAS_PYGETWINDOW = False

# Drag-and-drop support — optional. When installed, users can drop
# image files (PNG/JPG) onto the chat to ask the assistant about
# them. Gracefully degrades if missing.
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except ImportError:
    TkinterDnD = None    # type: ignore
    DND_FILES  = None    # type: ignore
    HAS_DND    = False

# Capture is available when we can both grab pixels (mss) AND encode
# JPEG (Pillow). Window picking additionally needs pygetwindow but
# region/full-screen capture work without it.
CAPTURE_AVAILABLE = HAS_PIL and HAS_MSS


# ═══════════════════════════════════════════════════════════════════════
#  PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

SCRIPT_DIR    = Path(__file__).resolve().parent
SETTINGS_FILE = SCRIPT_DIR / "app_settings.json"
DEBUG_LOG     = SCRIPT_DIR / "qwen_debug.log"
CONV_DIR      = SCRIPT_DIR / "conversations"

APP_TITLE     = "iSee"
APP_VERSION   = "1.2.0"

DEFAULT_SETTINGS: dict[str, Any] = {
    "active_theme":         "terminal",
    "custom_prompts":       [],     # list of {name, content}
    "active_prompt_name":   None,   # None or string matching a name
    "custom_accent":        "#aa66ff",   # hex color for the custom theme
    "preview_before_send":  False,  # show 5s preview modal on capture
    "window_geometry":      None,   # last "WxH+X+Y" string, restored on launch
}


# ───────────── Qwen tuning ──────────────────────────────────────────────
#
# These constants embed lessons from a previous build. Don't change
# them casually. Each is here for a reason.
#
QWEN_ENDPOINT = "http://localhost:11434/api/chat"
# Migrated from /api/generate to /api/chat. The /api/generate
# endpoint silently IGNORES `think: false` for qwen3.5:9b (Ollama
# issue #14793), so the model spends its output budget on hidden
# thinking tokens and returns sparse output. /api/chat correctly
# honors `think: false` when set at the TOP level of the payload
# (not nested inside options).

QWEN_MODEL                 = "qwen3.5:9b"
QWEN_TIMEOUT               = 600
# 9B vision calls run 60-90s steady state, longer cold. Beyond 10
# minutes is a sign of stuck Ollama, not slow inference.

QWEN_NUM_CTX               = 8192
QWEN_NUM_CTX_WITH_IMAGE    = 16384
# Bigger context with an image so the vision payload doesn't
# squeeze out conversation history.

QWEN_SAMPLING = {
    "temperature":      0.7,
    "top_p":            0.8,
    "top_k":            20,
    "presence_penalty": 1.5,    # reduces repetition in non-thinking mode
}

MAX_HISTORY_PAIRS = 50
JPEG_QUALITY      = 90    # screenshot encoding quality


# ───────────── System prompt ────────────────────────────────────────────
#
# Pure chat + optional screen vision. No tool routing, no agent
# scaffolding — those features are out of scope for this app.
#
SP_HEADER = (
    "You are a helpful local AI assistant running on the user's "
    "computer. Your conversations stay on the user's machine."
)

SP_SCREEN_VISION = (
    "You can see the user's screen when they attach a screenshot to "
    "their message. When a screenshot is present, examine it "
    "carefully and answer questions about what's visible. If a "
    "user's question references something on their screen ('this', "
    "'here', 'that button') and no screenshot was attached, ask "
    "them to enable the 📷 capture toggle and try again."
)

SP_STYLE = """\
STYLE — IMPORTANT:
- Be direct. Open with the answer, not preamble or restated questions.
- Don't preface answers with disclaimers ("Great question!", "It's
  important to note that...", "There are many factors..."). Just
  answer.
- Match the user's level of formality. Casual question → casual
  answer. Don't sound like a customer service script.
- Plain prose by default. Code blocks for code. Lists only when the
  content is genuinely list-shaped. Avoid heavy markdown — no
  unnecessary headers, no bolding every other phrase.
- Brevity by default. A short clear answer beats a long thorough one
  for most questions. If the user asks for depth, give depth."""

CUSTOM_PROMPT_SEPARATOR = "\n\n--- User instructions ---\n\n"


# ═══════════════════════════════════════════════════════════════════════
#  COLOR THEMES
# ═══════════════════════════════════════════════════════════════════════

THEMES = {
    "terminal": {
        "bg":           "#0a0e1a",
        "bg_mid":       "#060b14",
        "bg_bar":       "#111827",
        "fg":           "#c8d8e8",
        "fg_dim":       "#445566",
        "fg_status":    "#8899aa",
        "accent":       "#00ff88",
        "accent2":      "#00ccff",
        "bearish":      "#ff4466",
        "watch":        "#ffaa00",
        "border":       "#1a2535",
    },
    "gold": {
        "bg":           "#0c0a00",
        "bg_mid":       "#080600",
        "bg_bar":       "#1a1500",
        "fg":           "#f0e0b0",
        "fg_dim":       "#665500",
        "fg_status":    "#998844",
        "accent":       "#ffd700",
        "accent2":      "#ffaa00",
        "bearish":      "#ff4444",
        "watch":        "#ff8800",
        "border":       "#2a1e00",
    },
    "cyan": {
        "bg":           "#00111a",
        "bg_mid":       "#000d14",
        "bg_bar":       "#001824",
        "fg":           "#b0e8f0",
        "fg_dim":       "#224455",
        "fg_status":    "#4488aa",
        "accent":       "#00e5ff",
        "accent2":      "#00ffcc",
        "bearish":      "#ff3366",
        "watch":        "#ffcc00",
        "border":       "#001824",
    },
    "blood": {
        "bg":           "#110000",
        "bg_mid":       "#0c0000",
        "bg_bar":       "#1a0000",
        "fg":           "#f0c8c8",
        "fg_dim":       "#552222",
        "fg_status":    "#aa5555",
        "accent":       "#ff2244",
        "accent2":      "#ff6600",
        "bearish":      "#ff6600",
        "watch":        "#ffdd00",
        "border":       "#2a0000",
    },
    # The "custom" theme defaults to a desaturated purple but the
    # user's saved choice (settings["custom_accent"]) overrides
    # everything that's derived from accent on every load. Other
    # custom colors auto-derive from the chosen accent so the user
    # only has to pick one color to get a fully consistent theme.
    "custom": {
        "bg":           "#0a0a14",
        "bg_mid":       "#06060e",
        "bg_bar":       "#11111c",
        "fg":           "#d0d0e0",
        "fg_dim":       "#444455",
        "fg_status":    "#888899",
        "accent":       "#aa66ff",
        "accent2":      "#ddaaff",
        "bearish":      "#ff4466",
        "watch":        "#ffaa00",
        "border":       "#1a1a28",
    },
}


# ───────────── Custom-theme color helpers ──────────────────────────────
#
# When the user picks a custom accent color, we derive the rest of the
# palette from it so we get a coherent theme out of one click. The
# math is intentionally simple: convert RGB → HSV, scale value/sat
# differently for each role (bg = very dark + accent's hue, fg = light
# + accent's hue, etc.), convert back. This produces a palette that
# always feels like "the same color family" without the user having
# to pick five colors that happen to harmonize.

import colorsys as _colorsys


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = (hex_str or "").strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return (170, 102, 255)    # fall back to default purple
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except ValueError:
        return (170, 102, 255)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = (max(0, min(255, int(c))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _shift_value(hex_str: str, target_v: float,
                 target_s: float | None = None) -> str:
    """Take a hex color and return one with the given HSV value
    (and optionally saturation), preserving hue."""
    r, g, b = (c / 255.0 for c in _hex_to_rgb(hex_str))
    h, s, _v = _colorsys.rgb_to_hsv(r, g, b)
    if target_s is not None:
        s = target_s
    r2, g2, b2 = _colorsys.hsv_to_rgb(h, s, target_v)
    return _rgb_to_hex(
        (int(r2 * 255), int(g2 * 255), int(b2 * 255)))


def derive_custom_theme(accent_hex: str) -> dict[str, str]:
    """
    Build a complete theme dict from a single accent color. The
    bg/fg/border etc. share the accent's hue but are pushed to dark
    or light HSV values so the result is always readable.
    """
    a = (accent_hex or "#aa66ff").strip()
    if not a.startswith("#"):
        a = "#" + a
    return {
        # Backgrounds: very dark, very desaturated, hue from accent.
        "bg":          _shift_value(a, target_v=0.06, target_s=0.30),
        "bg_mid":      _shift_value(a, target_v=0.04, target_s=0.30),
        "bg_bar":      _shift_value(a, target_v=0.10, target_s=0.30),
        "border":      _shift_value(a, target_v=0.14, target_s=0.30),
        # Foreground text: light, slightly tinted with accent hue.
        "fg":          _shift_value(a, target_v=0.85, target_s=0.18),
        "fg_dim":      _shift_value(a, target_v=0.35, target_s=0.18),
        "fg_status":   _shift_value(a, target_v=0.55, target_s=0.18),
        # Accents: keep the user's exact pick; build accent2 as a
        # lighter tint of the same hue.
        "accent":      a,
        "accent2":     _shift_value(a, target_v=0.95, target_s=0.50),
        # Semantic colors: keep red-ish for bearish, amber for watch,
        # so errors and warnings always read correctly regardless of
        # accent choice.
        "bearish":     "#ff4466",
        "watch":       "#ffaa00",
    }


# ═══════════════════════════════════════════════════════════════════════
#  SETTINGS PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════

def load_settings() -> dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                for k, v in stored.items():
                    settings[k] = v
        except (json.JSONDecodeError, OSError) as e:
            print(f"[settings] failed to load {SETTINGS_FILE}: {e}",
                  file=sys.stderr)

    # Validate custom_prompts; drop malformed entries silently.
    valid_prompts = []
    for p in settings.get("custom_prompts", []):
        if (isinstance(p, dict)
                and isinstance(p.get("name"), str)
                and isinstance(p.get("content"), str)
                and p["name"].strip()):
            valid_prompts.append({
                "name":    p["name"].strip(),
                "content": p["content"],
            })
    settings["custom_prompts"] = valid_prompts

    return settings


def save_settings(settings: dict[str, Any]) -> None:
    tmp = SETTINGS_FILE.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        os.replace(tmp, SETTINGS_FILE)
    except OSError as e:
        print(f"[settings] failed to save {SETTINGS_FILE}: {e}",
              file=sys.stderr)


def is_first_launch() -> bool:
    return not SETTINGS_FILE.exists()


# ═══════════════════════════════════════════════════════════════════════
#  DEBUG LOG
# ═══════════════════════════════════════════════════════════════════════

def write_debug_log(label: str, payload: str) -> None:
    """Append a timestamped block to qwen_debug.log."""
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== {ts}  {label} ===\n")
            f.write(payload)
            f.write("\n")
    except OSError:
        pass    # logging must never crash the app


# ═══════════════════════════════════════════════════════════════════════
#  CONVERSATION PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════
#
# Each conversation is stored as a single JSON file in CONV_DIR. The
# filename is the conversation_id (a sortable timestamp). Images are
# never persisted — only the text turns plus the small "📷 attached:
# ..." marker so the user can see in saved logs that a screenshot was
# part of the original conversation.
#
# File schema:
# {
#   "id":         "20260503-191205-a1b2",
#   "title":      "First user message, truncated",
#   "created":    "2026-05-03T19:12:05",
#   "updated":    "2026-05-03T19:18:32",
#   "history":    [{"role":"user"|"assistant", "content":"..."}, ...],
#   "display":    [{"role":"user"|"assistant"|"system",
#                   "content":"...", "ts":"19:12:05",
#                   "capture_marker":"📷 attached: ..."|null}, ...]
# }
#
# Two parallel logs:
#   * `history` is what we replay to Qwen — minimal, just the turns
#   * `display` is what we render in the chat feed — includes
#     timestamps, capture markers, system lines

import re as _re

_TITLE_MAX_LEN  = 48
_CONV_INDEX_LIM = 200    # cap conversation list at this many


def _ensure_conv_dir() -> None:
    try:
        CONV_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[conv] failed to create {CONV_DIR}: {e}",
              file=sys.stderr)


def make_conversation_id() -> str:
    """A new conversation id: timestamp + short random tail."""
    import secrets
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tail = secrets.token_hex(2)
    return f"{ts}-{tail}"


def derive_title_from_history(
        display: list[dict[str, Any]] | None) -> str:
    """First user message, trimmed; fall back to '(untitled)'."""
    if not display:
        return "(untitled)"
    for item in display:
        if item.get("role") == "user":
            txt = (item.get("content") or "").strip()
            if not txt:
                continue
            txt = _re.sub(r"\s+", " ", txt)
            if len(txt) > _TITLE_MAX_LEN:
                txt = txt[: _TITLE_MAX_LEN - 1].rstrip() + "…"
            return txt
    return "(untitled)"


def save_conversation(conv_id: str,
                       title: str,
                       created_iso: str,
                       history: list[dict[str, str]],
                       display: list[dict[str, Any]]) -> bool:
    """
    Persist one conversation to disk. Returns True on success.
    """
    _ensure_conv_dir()
    path = CONV_DIR / f"{conv_id}.json"
    payload = {
        "id":      conv_id,
        "title":   title,
        "created": created_iso,
        "updated": datetime.now().isoformat(timespec="seconds"),
        "history": history,
        "display": display,
    }
    tmp = path.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        return True
    except OSError as e:
        print(f"[conv] failed to save {path}: {e}",
              file=sys.stderr)
        return False


def load_conversation(conv_id: str) -> dict[str, Any] | None:
    """Load a single conversation file. Returns None on failure."""
    path = CONV_DIR / f"{conv_id}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        # Sanity-check required fields and types.
        if not isinstance(data.get("history"), list):
            data["history"] = []
        if not isinstance(data.get("display"), list):
            data["display"] = []
        if not isinstance(data.get("title"), str):
            data["title"] = "(untitled)"
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"[conv] failed to load {path}: {e}",
              file=sys.stderr)
        return None


def list_conversations() -> list[dict[str, Any]]:
    """
    Return a list of conversation metadata dicts ordered most-recent
    first, capped at _CONV_INDEX_LIM. Each entry has id, title,
    created, updated.
    """
    _ensure_conv_dir()
    out: list[dict[str, Any]] = []
    try:
        for entry in CONV_DIR.iterdir():
            if not entry.is_file() or entry.suffix != ".json":
                continue
            if entry.name.endswith(".json.tmp"):
                continue
            try:
                with open(entry, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    continue
                out.append({
                    "id":      data.get("id") or entry.stem,
                    "title":   data.get("title") or "(untitled)",
                    "created": data.get("created") or "",
                    "updated": data.get("updated") or "",
                })
            except (json.JSONDecodeError, OSError):
                continue
    except OSError as e:
        print(f"[conv] failed to list {CONV_DIR}: {e}",
              file=sys.stderr)

    # Sort by `updated` desc (fall back to id which is timestamped).
    out.sort(
        key=lambda e: (e.get("updated") or "") + "|" + e.get("id", ""),
        reverse=True)
    return out[:_CONV_INDEX_LIM]


def delete_conversation(conv_id: str) -> bool:
    path = CONV_DIR / f"{conv_id}.json"
    try:
        if path.exists():
            path.unlink()
        return True
    except OSError as e:
        print(f"[conv] failed to delete {path}: {e}",
              file=sys.stderr)
        return False


def rename_conversation(conv_id: str, new_title: str) -> bool:
    data = load_conversation(conv_id)
    if data is None:
        return False
    data["title"] = (new_title or "(untitled)").strip() or "(untitled)"
    return save_conversation(
        conv_id,
        data["title"],
        data.get("created", datetime.now().isoformat(timespec="seconds")),
        data.get("history", []),
        data.get("display", []),
    )


# ═══════════════════════════════════════════════════════════════════════
#  SCREEN CAPTURE PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════

def capture_full_screen():
    """Grab a screenshot of the primary monitor. Returns a PIL.Image."""
    if not CAPTURE_AVAILABLE:
        raise RuntimeError(
            "Screen capture is unavailable. "
            "Install with: pip install mss Pillow")
    last_err = None
    for _ in range(2):
        try:
            with mss.mss() as sct:
                # Index 1 = primary monitor (index 0 = union of all).
                mon  = sct.monitors[1]
                shot = sct.grab(mon)
                return Image.frombytes(
                    "RGB", shot.size, shot.bgra, "raw", "BGRX")
        except Exception as e:    # noqa: BLE001
            last_err = e
            time.sleep(0.15)
    raise RuntimeError(
        f"Full-screen capture failed: "
        f"{type(last_err).__name__}: {last_err}")


def capture_region(region: tuple[int, int, int, int]):
    """Grab a screenshot of a specified bounding box.

    region: (left, top, right, bottom) in screen coordinates.
    """
    if not CAPTURE_AVAILABLE:
        raise RuntimeError(
            "Screen capture is unavailable. "
            "Install with: pip install mss Pillow")
    if region is None:
        raise RuntimeError(
            "No region selected. Pick a target via the ▾ button.")
    l, t, r, b = region
    w, h = r - l, b - t
    if w <= 0 or h <= 0:
        raise RuntimeError(
            f"Invalid region size {w}×{h}px. Re-select the target.")
    if w > 10000 or h > 10000:
        raise RuntimeError(
            f"Region absurdly large ({w}×{h}px). Re-select the target.")

    last_err = None
    for _ in range(2):
        try:
            with mss.mss() as sct:
                mon  = {"left": l, "top": t, "width": w, "height": h}
                shot = sct.grab(mon)
                return Image.frombytes(
                    "RGB", shot.size, shot.bgra, "raw", "BGRX")
        except Exception as e:    # noqa: BLE001
            last_err = e
            time.sleep(0.15)
    raise RuntimeError(
        f"Region capture failed: "
        f"{type(last_err).__name__}: {last_err}")


def encode_image_b64(img) -> str:
    """Encode a PIL.Image to a base64 JPEG string for Ollama."""
    if not HAS_PIL:
        raise RuntimeError("Pillow is required to encode screenshots.")
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ═══════════════════════════════════════════════════════════════════════
#  REGION SELECTOR
# ═══════════════════════════════════════════════════════════════════════

class RegionSelector:
    """
    Full-screen overlay where the user click-drags a bounding box.
    Result lands in `self.result` as (left, top, right, bottom) or
    None on cancel.
    """

    def __init__(self, parent: tk.Misc, accent_color: str = "#00ff88"):
        self.result  = None
        self.accent  = accent_color
        self.start_x = self.start_y = 0
        self.rect_id = None
        # Guard against the click that opened this overlay being
        # interpreted as the start/end of a drag. We only honor
        # mouse events after a short arming delay.
        self._armed     = False
        self._dragging  = False

        self.win = tk.Toplevel(parent)
        self.win.attributes("-fullscreen", True,
                            "-topmost",     True,
                            "-alpha",       0.35)
        self.win.configure(bg="black")
        self.win.focus_force()
        self.win.after(250, self._arm)

        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        self.canvas = tk.Canvas(
            self.win, bg="black", width=sw, height=sh,
            highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(
            sw // 2, 40,
            text="Click and drag to select an area   |   ESC to cancel",
            fill=accent_color, font=("Courier New", 14, "bold"))

        self.canvas.bind("<ButtonPress-1>",   self._press)
        self.canvas.bind("<B1-Motion>",       self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.win.bind("<Escape>", lambda e: self._cancel())

    def _arm(self):
        self._armed = True

    def _press(self, e):
        if not self._armed:
            return
        self.start_x, self.start_y = e.x, e.y
        self._dragging = True
        if self.rect_id:
            self.canvas.delete(self.rect_id)

    def _drag(self, e):
        if not self._armed or not self._dragging:
            return
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, e.x, e.y,
            outline=self.accent, width=2,
            fill=self.accent, stipple="gray12")
        self.canvas.delete("dims")
        self.canvas.create_text(
            e.x + 10, e.y + 10,
            text=f"{abs(e.x - self.start_x)} × {abs(e.y - self.start_y)}",
            fill=self.accent, font=("Courier New", 10),
            anchor="nw", tags="dims")

    def _release(self, e):
        if not self._armed or not self._dragging:
            return
        self._dragging = False
        x1, y1 = min(self.start_x, e.x), min(self.start_y, e.y)
        x2, y2 = max(self.start_x, e.x), max(self.start_y, e.y)
        if abs(x2 - x1) < 20 or abs(y2 - y1) < 20:
            return    # too small — wait for a real drag
        self.result = (x1, y1, x2, y2)
        self.win.destroy()

    def _cancel(self):
        self.result = None
        self.win.destroy()


# ═══════════════════════════════════════════════════════════════════════
#  WINDOW PICKER
# ═══════════════════════════════════════════════════════════════════════

class WindowPicker(tk.Toplevel):
    """
    Modal dialog showing a list of open windows. Result lands in
    `self.result` as (left, top, right, bottom) bounding box, with
    `self.selected_window_title` as the window title for display.
    """

    def __init__(self, parent: tk.Misc, theme: dict[str, str]):
        super().__init__(parent)
        self.title("Select Window")
        self.geometry("420x440")
        self.resizable(False, True)
        self.configure(bg=theme["bg"])
        self.result = None
        self.selected_window_title: str | None = None
        self.T = theme
        self.transient(parent)
        self.grab_set()
        self._windows: list[Any] = []
        self._build()

    def _build(self):
        T = self.T
        tk.Label(self, text="SELECT WINDOW",
                 font=("Courier New", 10, "bold"),
                 fg=T["accent"], bg=T["bg"], pady=10).pack(fill="x")

        f = tk.Frame(self, bg=T["bg"])
        f.pack(fill="both", expand=True, padx=12)
        sb = tk.Scrollbar(f); sb.pack(side="right", fill="y")
        self.lb = tk.Listbox(
            f, bg=T["bg_mid"], fg=T["fg"],
            font=("Courier New", 10), relief="flat", bd=0,
            selectbackground=T["accent"], selectforeground=T["bg"],
            activestyle="none", yscrollcommand=sb.set, cursor="hand2")
        self.lb.pack(side="left", fill="both", expand=True)
        sb.config(command=self.lb.yview)
        self.lb.bind("<Double-Button-1>", self._select)
        self._populate()

        bf = tk.Frame(self, bg=T["bg"], pady=8)
        bf.pack(fill="x", padx=12)
        tk.Button(bf, text="Select",
                  font=("Courier New", 10, "bold"),
                  fg=T["bg"], bg=T["accent"],
                  relief="flat", cursor="hand2",
                  padx=14, pady=5,
                  command=self._select).pack(side="left")
        tk.Button(bf, text="Refresh",
                  font=("Courier New", 9),
                  fg=T["fg_dim"], bg=T["bg"],
                  relief="flat", cursor="hand2",
                  padx=10, pady=5,
                  command=self._populate).pack(side="left", padx=6)
        tk.Button(bf, text="Cancel",
                  font=("Courier New", 9),
                  fg=T["bearish"], bg=T["bg"],
                  relief="flat", cursor="hand2",
                  padx=10, pady=5,
                  command=self.destroy).pack(side="right")

    def _populate(self):
        self.lb.delete(0, "end")
        self._windows = []
        if not HAS_PYGETWINDOW:
            self.lb.insert("end", "  pip install pygetwindow")
            return
        try:
            wins = sorted(
                [w for w in gw.getAllWindows()
                 if w.title.strip() and w.width > 50],
                key=lambda w: w.title.lower())
            for w in wins:
                self.lb.insert("end", f"  {w.title[:55]}")
                self._windows.append(w)
        except Exception as e:    # noqa: BLE001
            self.lb.insert("end", f"  Error: {e}")

    def _select(self, event=None):
        sel = self.lb.curselection()
        if not sel or not self._windows:
            return
        idx = sel[0]
        if idx >= len(self._windows):
            return
        w = self._windows[idx]
        try:
            sw = self.winfo_screenwidth()
            sh = self.winfo_screenheight()
            self.result = (
                max(0, w.left), max(0, w.top),
                min(sw, w.left + w.width), min(sh, w.top + w.height))
            self.selected_window_title = w.title
        except Exception:    # noqa: BLE001
            return
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════
#  CAPTURE-MODE PICKER
# ═══════════════════════════════════════════════════════════════════════

class CaptureModeDialog(tk.Toplevel):
    """
    Asks the user to choose a capture mode. Result is one of
    "full_screen" | "region" | "window" | None (cancelled).
    """

    def __init__(self, parent: tk.Misc, theme: dict[str, str]):
        super().__init__(parent)
        self.T = theme
        self.result: str | None = None

        self.title("Choose capture target")
        self.configure(bg=theme["bg"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._build()

        self.update_idletasks()
        try:
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            w  = self.winfo_width()
            h  = self.winfo_height()
            self.geometry(f"+{px + (pw - w) // 2}+{py + (ph - h) // 2}")
        except tk.TclError:
            pass

        self.bind("<Escape>", lambda e: self._cancel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _build(self):
        T = self.T

        tk.Label(self, text="CHOOSE CAPTURE TARGET",
                 font=("Courier New", 12, "bold"),
                 fg=T["accent"], bg=T["bg"]).pack(
            anchor="w", padx=20, pady=(16, 4))
        tk.Label(self,
                 text="Where should the assistant look when 📷 is on?",
                 font=("Courier New", 9), wraplength=400, justify="left",
                 fg=T["fg_status"], bg=T["bg"]).pack(
            anchor="w", padx=20, pady=(0, 14))

        opts = tk.Frame(self, bg=T["bg"])
        opts.pack(fill="x", padx=20, pady=(0, 8))

        self._opt_button(opts, "🖥  Full Screen",
                         "Capture the entire primary monitor.",
                         lambda: self._pick("full_screen"))
        self._opt_button(opts, "✥  Region",
                         "Click and drag to pick a rectangular area.",
                         lambda: self._pick("region"))
        win_help = "Pick from the list of open windows."
        if not HAS_PYGETWINDOW:
            win_help += "  (requires: pip install pygetwindow)"
        self._opt_button(opts, "⊞  Window",
                         win_help,
                         lambda: self._pick("window"),
                         enabled=HAS_PYGETWINDOW)

        btn_row = tk.Frame(self, bg=T["bg"])
        btn_row.pack(fill="x", padx=20, pady=(8, 18))
        tk.Button(btn_row, text="Cancel",
                  font=("Courier New", 9),
                  fg=T["fg_dim"], bg=T["bg"],
                  relief="flat", cursor="hand2", padx=12, pady=6,
                  command=self._cancel).pack(side="right")

    def _opt_button(self, parent: tk.Misc, label: str,
                    help_text: str, cmd: Any,
                    enabled: bool = True) -> None:
        T = self.T
        row = tk.Frame(parent, bg=T["bg"], cursor="hand2" if enabled else "")
        row.pack(fill="x", pady=4)
        fg = T["accent"] if enabled else T["fg_dim"]
        btn = tk.Label(row, text=label,
                       font=("Courier New", 11, "bold"),
                       fg=fg, bg=T["bg"], anchor="w")
        btn.pack(fill="x", pady=(2, 0))
        sub = tk.Label(row, text=help_text,
                       font=("Courier New", 8),
                       fg=T["fg_dim"], bg=T["bg"], anchor="w",
                       wraplength=400, justify="left")
        sub.pack(fill="x")
        if enabled:
            for w in (row, btn, sub):
                w.bind("<Button-1>", lambda e, c=cmd: c())

    def _pick(self, mode: str):
        self.result = mode
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════
#  QWEN CLIENT
# ═══════════════════════════════════════════════════════════════════════

def assemble_system_prompt(custom_prompt: str | None,
                           screen_vision_enabled: bool) -> str:
    """Build the final system prompt sent to Qwen."""
    parts = [SP_HEADER]
    if screen_vision_enabled:
        parts.append(SP_SCREEN_VISION)
    parts.append(SP_STYLE)
    base = "\n\n".join(p.strip() for p in parts)
    if custom_prompt and custom_prompt.strip():
        return base + CUSTOM_PROMPT_SEPARATOR + custom_prompt.strip()
    return base


def clean_qwen_response(text: str) -> str:
    """
    Strip <think>...</think> blocks and bare ``` fences. Handles
    malformed (unclosed) <think> by dropping everything from the
    opening tag.
    """
    if not isinstance(text, str):
        return ""
    clean = text.strip()

    while "<think>" in clean:
        think_start = clean.find("<think>")
        think_end   = clean.find("</think>", think_start)
        if think_end >= 0:
            clean = (clean[:think_start]
                     + clean[think_end + len("</think>"):]).strip()
        else:
            clean = clean[:think_start].strip()
            break

    if "```" in clean:
        out = []
        for line in clean.split("\n"):
            if line.strip().startswith("```"):
                continue
            out.append(line)
        clean = "\n".join(out).strip()

    return clean


def call_qwen(messages: list[dict[str, Any]],
              custom_prompt: str | None = None,
              image_b64: str | None = None,
              screen_vision_enabled: bool = False,
              ) -> tuple[str, str]:
    """
    Make a Qwen call. Returns (cleaned_text, raw_text).

    BLOCKING — call from a worker thread; can take 60-90 seconds.

    `messages` is the conversation so far. If `image_b64` is given
    it is attached to the LAST user message in the list (Ollama
    /api/chat shape: include an "images" array on the message).
    Historical messages never carry images — only the current turn's
    image is sent.
    """
    msgs: list[dict[str, Any]] = list(messages)
    if image_b64 and msgs and msgs[-1].get("role") == "user":
        last = dict(msgs[-1])
        last["images"] = [image_b64]
        msgs[-1] = last

    system_content = assemble_system_prompt(
        custom_prompt, screen_vision_enabled)
    full_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_content},
    ]
    full_messages.extend(msgs)

    num_ctx = QWEN_NUM_CTX_WITH_IMAGE if image_b64 else QWEN_NUM_CTX
    payload = {
        "model":    QWEN_MODEL,
        "messages": full_messages,
        "stream":   False,
        "think":    False,    # TOP LEVEL — do NOT nest in options
        "options": {
            **QWEN_SAMPLING,
            "num_ctx": num_ctx,
        },
    }

    response = requests.post(
        QWEN_ENDPOINT,
        json=payload,
        timeout=QWEN_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    raw  = data.get("message", {}).get("content", "") or ""
    return clean_qwen_response(raw), raw


# ═══════════════════════════════════════════════════════════════════════
#  CAPTURE PREVIEW DIALOG
# ═══════════════════════════════════════════════════════════════════════

class PreviewDialog(tk.Toplevel):
    """
    Brief modal that surfaces a captured screenshot before it's sent
    to Qwen. A 2-second countdown auto-sends; the user can hit
    Cancel to abort or Send Now to skip the wait.

    Result is read by the caller after self.wait_window():
        self.result : "send" | "cancel"

    The PIL.Image is held by the caller; we receive a pre-rendered
    thumbnail PhotoImage to keep the dialog dumb. We DO NOT keep a
    reference to the source PIL image — that's the caller's job.
    """

    AUTO_SEND_MS = 5000     # 5 seconds
    TICK_MS      = 100      # countdown refresh rate

    def __init__(self, parent: tk.Misc, theme: dict[str, str],
                 thumb_photo: tk.PhotoImage,
                 prompt_text: str,
                 capture_label: str):
        super().__init__(parent)
        self.T            = theme
        self._thumb       = thumb_photo    # hold ref so it doesn't GC
        self.result       : str = "send"   # default if nothing clicked
        self._cancelled_timer = False
        self._remaining_ms : int = self.AUTO_SEND_MS
        self._tick_job    = None

        self.title("Sending in 5s...")
        self.configure(bg=theme["bg"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._build(prompt_text, capture_label)
        self._center_on_parent(parent)

        self.bind("<Escape>", lambda e: self._cancel())
        self.bind("<Return>", lambda e: self._send_now())
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        # Kick off the countdown.
        self._schedule_tick()

    def _build(self, prompt_text: str, capture_label: str) -> None:
        T = self.T

        # Header
        tk.Label(self, text="PREVIEW BEFORE SEND",
                 font=("Courier New", 11, "bold"),
                 fg=T["accent"], bg=T["bg"]).pack(
            anchor="w", padx=18, pady=(14, 2))
        tk.Label(self, text=capture_label,
                 font=("Courier New", 8, "italic"),
                 fg=T["fg_dim"], bg=T["bg"]).pack(
            anchor="w", padx=18, pady=(0, 8))

        # Thumbnail with a 1px accent border to mirror the
        # in-feed treatment of attached captures.
        thumb_wrap = tk.Frame(self, bg=T["accent"])
        thumb_wrap.pack(padx=18, pady=(0, 10))
        thumb_inner = tk.Frame(thumb_wrap, bg=T["bg_mid"])
        thumb_inner.pack(padx=1, pady=1)
        tk.Label(thumb_inner, image=self._thumb,
                 bg=T["bg_mid"]).pack()

        # Prompt readback (clipped — the full prompt is already in
        # the input box; this just confirms what the user typed).
        prompt_short = (prompt_text if len(prompt_text) <= 200
                        else prompt_text[:200].rstrip() + "…")
        tk.Label(self, text=f"Prompt: {prompt_short}",
                 font=("Courier New", 9),
                 wraplength=460, justify="left",
                 fg=T["fg"], bg=T["bg"]).pack(
            anchor="w", padx=18, pady=(0, 8))

        # Countdown line + buttons.
        bar = tk.Frame(self, bg=T["bg"])
        bar.pack(fill="x", padx=18, pady=(0, 16))

        self._countdown_var = tk.StringVar(
            value=self._countdown_text())
        tk.Label(bar, textvariable=self._countdown_var,
                 font=("Courier New", 9, "bold"),
                 fg=T["watch"], bg=T["bg"]).pack(side="left")

        tk.Button(bar, text="✕ Cancel",
                  font=("Courier New", 9),
                  fg=T["bearish"], bg=T["bg"],
                  relief="flat", cursor="hand2",
                  padx=12, pady=4,
                  command=self._cancel).pack(side="right",
                                              padx=(8, 0))
        tk.Button(bar, text="▶ Send now",
                  font=("Courier New", 9, "bold"),
                  fg=T["bg"], bg=T["accent"],
                  relief="flat", cursor="hand2",
                  padx=14, pady=4,
                  command=self._send_now).pack(side="right")

    def _center_on_parent(self, parent: tk.Misc) -> None:
        self.update_idletasks()
        try:
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            w  = self.winfo_width()
            h  = self.winfo_height()
            self.geometry(
                f"+{px + (pw - w) // 2}+{py + (ph - h) // 2}")
        except tk.TclError:
            pass

    def _countdown_text(self) -> str:
        secs = max(0, self._remaining_ms) / 1000.0
        return f"⏱  Auto-send in {secs:.1f}s"

    def _schedule_tick(self) -> None:
        if self._cancelled_timer:
            return
        self._countdown_var.set(self._countdown_text())
        if self._remaining_ms <= 0:
            # Time's up — auto-send.
            self.result = "send"
            self.destroy()
            return
        self._remaining_ms -= self.TICK_MS
        self._tick_job = self.after(self.TICK_MS, self._schedule_tick)

    def _cancel_timer(self) -> None:
        self._cancelled_timer = True
        if self._tick_job is not None:
            try:
                self.after_cancel(self._tick_job)
            except tk.TclError:
                pass
            self._tick_job = None

    def _send_now(self) -> None:
        self._cancel_timer()
        self.result = "send"
        self.destroy()

    def _cancel(self) -> None:
        self._cancel_timer()
        self.result = "cancel"
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════
#  MANAGE PROMPTS DIALOG
# ═══════════════════════════════════════════════════════════════════════

class ManagePromptsDialog(tk.Toplevel):
    """Modal dialog for creating, editing, and deleting custom prompts."""

    def __init__(self, parent: tk.Misc, settings: dict[str, Any],
                 on_change: Any,
                 select_name: str | None = None,
                 start_in_new_mode: bool = False):
        super().__init__(parent)
        self.settings  = settings
        self.on_change = on_change

        T = THEMES.get(settings.get("active_theme", "terminal"),
                       THEMES["terminal"])
        self.T = T

        self.title("Manage Custom Prompts")
        self.configure(bg=T["bg"])
        self.resizable(True, True)
        self.minsize(620, 420)
        self.geometry("720x460")
        self.transient(parent)
        self.grab_set()

        self._loaded_name: str | None = None

        self._build_ui()
        self._refresh_listbox(select_name=select_name)

        if start_in_new_mode and not select_name:
            self._cmd_new()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Escape>", lambda e: self._on_close())

    def _build_ui(self) -> None:
        T = self.T

        tk.Label(self, text="MANAGE CUSTOM PROMPTS",
                 font=("Courier New", 12, "bold"),
                 fg=T["accent"], bg=T["bg"]).pack(
            anchor="w", padx=18, pady=(14, 6))
        tk.Label(self,
                 text=("Define reusable instructions to add on top of "
                       "the base assistant prompt. The active prompt "
                       "is appended to every Qwen call until you "
                       "switch back to Default."),
                 font=("Courier New", 9), wraplength=680, justify="left",
                 fg=T["fg_status"], bg=T["bg"]).pack(
            anchor="w", padx=18, pady=(0, 10))

        body = tk.Frame(self, bg=T["bg"])
        body.pack(fill="both", expand=True, padx=18, pady=(0, 6))

        left = tk.Frame(body, bg=T["bg"])
        left.pack(side="left", fill="y")
        tk.Label(left, text="Saved prompts",
                 font=("Courier New", 9, "bold"),
                 fg=T["fg_dim"], bg=T["bg"]).pack(anchor="w")
        self.listbox = tk.Listbox(
            left, font=("Courier New", 10),
            fg=T["fg"], bg=T["bg_mid"],
            selectbackground=T["accent"],
            selectforeground=T["bg"],
            activestyle="none",
            relief="flat", bd=0, width=24, height=14,
            highlightthickness=0, exportselection=False)
        self.listbox.pack(fill="y", expand=False, pady=(4, 0))
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        right = tk.Frame(body, bg=T["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(14, 0))

        tk.Label(right, text="Name",
                 font=("Courier New", 9, "bold"),
                 fg=T["fg_dim"], bg=T["bg"]).pack(anchor="w")
        self.name_var = tk.StringVar()
        self.name_entry = tk.Entry(
            right, textvariable=self.name_var,
            font=("Courier New", 10),
            fg=T["fg"], bg=T["bg_mid"],
            insertbackground=T["accent"],
            relief="flat", bd=0)
        self.name_entry.pack(fill="x", ipady=4, pady=(2, 10))

        tk.Label(right, text="Content",
                 font=("Courier New", 9, "bold"),
                 fg=T["fg_dim"], bg=T["bg"]).pack(anchor="w")
        self.content_text = scrolledtext.ScrolledText(
            right, font=("Courier New", 10),
            fg=T["fg"], bg=T["bg_mid"],
            insertbackground=T["accent"],
            relief="flat", bd=0, wrap=tk.WORD,
            height=10)
        self.content_text.pack(fill="both", expand=True, pady=(2, 0))

        btn_row = tk.Frame(self, bg=T["bg"])
        btn_row.pack(fill="x", padx=18, pady=(8, 14))

        tk.Button(btn_row, text="+ New",
                  font=("Courier New", 9, "bold"),
                  fg=T["fg"], bg=T["bg_mid"],
                  relief="flat", cursor="hand2", padx=12, pady=6,
                  command=self._cmd_new).pack(side="left")
        tk.Button(btn_row, text="✓ Save",
                  font=("Courier New", 9, "bold"),
                  fg=T["bg"], bg=T["accent"],
                  relief="flat", cursor="hand2", padx=14, pady=6,
                  command=self._cmd_save).pack(side="left", padx=(8, 0))
        tk.Button(btn_row, text="✕ Delete",
                  font=("Courier New", 9),
                  fg=T["bearish"], bg=T["bg"],
                  relief="flat", cursor="hand2", padx=12, pady=6,
                  command=self._cmd_delete).pack(side="left", padx=(8, 0))
        tk.Button(btn_row, text="Close",
                  font=("Courier New", 9),
                  fg=T["fg_dim"], bg=T["bg"],
                  relief="flat", cursor="hand2", padx=12, pady=6,
                  command=self._on_close).pack(side="right")

    def _refresh_listbox(self, select_name: str | None = None) -> None:
        active = self.settings.get("active_prompt_name")
        prompts = self.settings.get("custom_prompts", [])

        self.listbox.delete(0, "end")
        for p in prompts:
            label = "● " + p["name"] if active == p["name"] \
                    else "  " + p["name"]
            self.listbox.insert("end", label)

        target = select_name
        if target is None and self._loaded_name:
            target = self._loaded_name
        if target:
            for idx, p in enumerate(prompts):
                if p["name"] == target:
                    self.listbox.selection_clear(0, "end")
                    self.listbox.selection_set(idx)
                    self.listbox.see(idx)
                    self._load_index(idx)
                    return

        if not prompts:
            self._clear_fields(loaded_name=None)

    def _on_select(self, event: tk.Event | None = None) -> None:
        sel = self.listbox.curselection()
        if not sel:
            return
        self._load_index(sel[0])

    def _load_index(self, idx: int) -> None:
        prompts = self.settings.get("custom_prompts", [])
        if not (0 <= idx < len(prompts)):
            return
        p = prompts[idx]
        self.name_var.set(p["name"])
        self.content_text.delete("1.0", "end")
        self.content_text.insert("1.0", p["content"])
        self._loaded_name = p["name"]

    def _clear_fields(self, loaded_name: str | None) -> None:
        self.name_var.set("")
        self.content_text.delete("1.0", "end")
        self._loaded_name = loaded_name

    def _cmd_new(self) -> None:
        self.listbox.selection_clear(0, "end")
        self._clear_fields(loaded_name=None)
        self.name_entry.focus_set()

    def _cmd_save(self) -> None:
        name    = self.name_var.get().strip()
        content = self.content_text.get("1.0", "end").rstrip()

        if not name:
            messagebox.showwarning("Name required",
                                   "Please enter a name for this prompt.",
                                   parent=self)
            return
        if not content:
            messagebox.showwarning("Content required",
                                   "The prompt content is empty.",
                                   parent=self)
            return

        prompts = list(self.settings.get("custom_prompts", []))

        # Detect collision with another entry of the same name.
        collision_idx = None
        for idx, p in enumerate(prompts):
            if p["name"] == name and p["name"] != self._loaded_name:
                collision_idx = idx
                break
        if collision_idx is not None:
            if not messagebox.askyesno(
                    "Overwrite?",
                    f"A prompt named '{name}' already exists. "
                    "Overwrite it?",
                    parent=self):
                return
            prompts.pop(collision_idx)

        # Update or insert.
        replaced = False
        for idx, p in enumerate(prompts):
            if p["name"] == self._loaded_name:
                prompts[idx] = {"name": name, "content": content}
                replaced = True
                break
        if not replaced:
            prompts.append({"name": name, "content": content})

        # If we renamed the active prompt, keep active pointed at it.
        active_name = self.settings.get("active_prompt_name")
        if active_name and active_name == self._loaded_name and \
                self._loaded_name != name:
            self.settings["active_prompt_name"] = name

        self.settings["custom_prompts"] = prompts
        save_settings(self.settings)
        self._loaded_name = name

        if callable(self.on_change):
            self.on_change()
        self._refresh_listbox(select_name=name)

    def _cmd_delete(self) -> None:
        if not self._loaded_name:
            return
        if not messagebox.askyesno(
                "Delete prompt?",
                f"Delete custom prompt '{self._loaded_name}'? "
                "This can't be undone.",
                parent=self):
            return

        target = self._loaded_name
        prompts = [p for p in self.settings.get("custom_prompts", [])
                   if p["name"] != target]
        self.settings["custom_prompts"] = prompts

        if self.settings.get("active_prompt_name") == target:
            self.settings["active_prompt_name"] = None

        save_settings(self.settings)
        self._clear_fields(loaded_name=None)
        if callable(self.on_change):
            self.on_change()
        self._refresh_listbox(select_name=None)

    def _on_close(self) -> None:
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════

class ISeeApp:
    """
    Top-level controller. Owns root window, settings, theme, chat
    history, and capture state. Dispatches Qwen calls on a worker
    thread.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.first_launch_flag = is_first_launch()
        self.settings = load_settings()
        if self.first_launch_flag:
            save_settings(self.settings)

        active_theme = self.settings.get("active_theme", "terminal")
        if active_theme not in THEMES:
            active_theme = "terminal"
        self.theme_name = active_theme
        self.theme      = self._resolve_theme(active_theme)

        # Conversation history sent to Qwen on each call.
        # Format: list of {"role": "user"|"assistant", "content": str}.
        # Images are attached on send and not retained in history.
        self.history: list[dict[str, str]] = []

        # Parallel display log used to render the chat feed and to
        # persist the conversation to disk. Each entry has role,
        # content, ts, and optional capture_marker. We keep this
        # separate from `history` because the model only needs role
        # and content; the display log carries presentation
        # metadata that would just be noise for the model.
        self.display: list[dict[str, Any]] = []

        # Identity of the currently-loaded conversation. None means
        # the current chat hasn't been touched yet (no first user
        # turn) — once the user sends their first message we mint
        # an id and start tracking it.
        self.conv_id      : str | None = None
        self.conv_title   : str        = ""
        self.conv_created : str        = ""

        # Threading state
        self._call_in_flight = False

        # ── Capture state ──
        # capture_enabled : True/False — toggle state (defaults OFF
        #                   on every launch)
        # capture_mode    : "full_screen" | "region" | "window" | None
        # capture_target  : (l,t,r,b) bounding box for region/window
        # capture_label   : short description for the chat marker
        self.capture_enabled : bool                              = False
        self.capture_mode    : str | None                        = None
        self.capture_target  : tuple[int, int, int, int] | None  = None
        self.capture_label   : str                               = ""

        self._build_ui()
        self._apply_theme()
        self._refresh_prompt_menu()
        self._update_prompt_label()
        self._update_capture_button()
        self._update_preview_button()

    # -- UI scaffolding ----------------------------------------------

    def _build_ui(self) -> None:
        T = self.theme
        r = self.root
        r.title(APP_TITLE)
        r.configure(bg=T["bg"])
        # Restore saved geometry if present and looks valid; otherwise
        # fall back to the default size. We do a bounds-check to
        # prevent a window saved on a now-disconnected monitor from
        # opening offscreen.
        saved_geom = self.settings.get("window_geometry")
        applied = False
        if isinstance(saved_geom, str) and saved_geom.strip():
            try:
                # Validate format "WxH+X+Y" or "WxH-X-Y"
                import re as _re_g
                if _re_g.match(r"^\d+x\d+[+-]-?\d+[+-]-?\d+$",
                               saved_geom):
                    r.geometry(saved_geom)
                    applied = True
            except (tk.TclError, ValueError):
                applied = False
        if not applied:
            r.geometry("1140x720")
        r.minsize(700, 520)
        self._set_window_icon()

        # ── Outer split: sidebar (left) + main column (right) ──
        # The sidebar can be collapsed via the chevron toggle.
        self.sidebar_visible = True
        self.sidebar = tk.Frame(r, bg=T["bg_mid"], width=220)
        # Don't let inner widgets force the sidebar wider:
        self.sidebar.pack_propagate(False)
        self.sidebar.pack(side="left", fill="y")

        # Thin border line between sidebar and main
        self.sidebar_sep = tk.Frame(r, bg=T["border"], width=1)
        self.sidebar_sep.pack(side="left", fill="y")

        main = tk.Frame(r, bg=T["bg"])
        main.pack(side="left", fill="both", expand=True)
        self._main_col = main

        self._build_sidebar()
        self._build_main(main)

    def _set_window_icon(self) -> None:
        """
        Set the taskbar / titlebar icon to a rendered eye glyph.
        Pillow is required (already a hard dep). On any failure
        (PIL missing, font issue, tkinter platform quirk) we just
        leave the default icon — never crash the app over chrome.
        """
        if not HAS_PIL:
            return
        try:
            # Render the 👁 emoji into a 64×64 transparent PNG.
            # tkinter's PhotoImage handles PNG natively in Py3.
            size  = 64
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)

            # Try a few common system fonts that have decent emoji
            # coverage. If none load, draw a plain circle instead.
            font = None
            for name in ("seguiemj.ttf",   # Windows emoji
                         "AppleColorEmoji.ttf",
                         "NotoColorEmoji.ttf",
                         "DejaVuSans.ttf"):
                try:
                    font = ImageFont.truetype(name, size - 12)
                    break
                except (OSError, IOError):
                    continue

            if font is not None:
                # Center the glyph
                bbox = draw.textbbox((0, 0), "👁", font=font,
                                      embedded_color=True)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                x  = (size - tw) // 2 - bbox[0]
                y  = (size - th) // 2 - bbox[1]
                # embedded_color=True so colored emoji glyphs render
                # in their native palette (the white/grey eye look).
                try:
                    draw.text((x, y), "👁", font=font,
                              embedded_color=True)
                except TypeError:
                    # Older Pillow without embedded_color kwarg
                    draw.text((x, y), "👁", font=font, fill=(220, 220, 220))
            else:
                # Generic eye fallback: a circle with a smaller dark
                # circle inside it. Keeps the spirit even if no
                # emoji-capable font is available.
                draw.ellipse((6, 14, size - 6, size - 14),
                             fill=(220, 220, 220, 255))
                draw.ellipse((size // 2 - 10, size // 2 - 10,
                              size // 2 + 10, size // 2 + 10),
                             fill=(20, 20, 20, 255))

            # Save to a temp file because PhotoImage(file=...) is the
            # most portable path — `data=` with raw bytes is finicky
            # across tk versions.
            import tempfile
            tmp_path = Path(tempfile.gettempdir()) / "isee_icon.png"
            img.save(tmp_path, format="PNG")

            icon = tk.PhotoImage(file=str(tmp_path))
            # Hold a reference so it isn't GC'd. Without this the
            # taskbar icon often blanks out a moment later.
            self._app_icon = icon
            self.root.iconphoto(True, icon)
        except Exception:    # noqa: BLE001
            # Icon is pure chrome — never let it break startup.
            pass

    def _build_main(self, main: tk.Frame) -> None:
        """Build the main column (everything that isn't the sidebar)."""
        T = self.theme

        # ── Header ──
        header = tk.Frame(main, bg=T["bg"])
        header.pack(fill="x", padx=16, pady=(14, 4))

        # Sidebar collapse/expand chevron (left of title)
        self.sidebar_toggle_btn = tk.Button(
            header, text="◀",
            font=("Courier New", 10, "bold"),
            fg=T["fg_dim"], bg=T["bg"],
            relief="flat", cursor="hand2",
            padx=6, pady=2,
            command=self._toggle_sidebar)
        self.sidebar_toggle_btn.pack(side="left", padx=(0, 8))

        # Eye glyph in the lighter accent2 shade so it complements
        # the brand text without looking like a duplicate. The two
        # labels read as a single composed wordmark: 👁 iSee.
        self.brand_eye = tk.Label(
            header, text="👁",
            font=("Courier New", 16, "bold"),
            fg=T["accent2"], bg=T["bg"])
        self.brand_eye.pack(side="left", padx=(0, 6))
        self.brand_label = tk.Label(
            header, text="iSee",
            font=("Courier New", 16, "bold"),
            fg=T["accent"], bg=T["bg"])
        self.brand_label.pack(side="left")
        self.signal_label = tk.Label(
            header, text="● READY",
            font=("Courier New", 11, "bold"),
            fg=T["accent"], bg=T["bg"])
        self.signal_label.pack(side="right")

        # ── Status bar ──
        self.status_var = tk.StringVar(value="ready")
        self.status_label = tk.Label(
            main, textvariable=self.status_var,
            font=("Courier New", 9),
            fg=T["fg_status"], bg=T["bg_bar"],
            anchor="w", padx=10, pady=4)
        self.status_label.pack(fill="x")

        # ── Top-controls row ──
        ctrl_row = tk.Frame(main, bg=T["bg"])
        ctrl_row.pack(fill="x", padx=14, pady=(8, 2))

        # Tagline — split so the actionable "share your screen" bit
        # picks up the theme accent and draws the eye. The 📷 glyph
        # also pops in accent so users connect it to the toggle
        # button down below.
        tag_left = tk.Label(
            ctrl_row, text="Local AI assistant.   ",
            font=("Courier New", 8),
            fg=T["fg_dim"], bg=T["bg"])
        tag_left.pack(side="left")

        tag_emoji = tk.Label(
            ctrl_row, text="📷",
            font=("Courier New", 9, "bold"),
            fg=T["accent"], bg=T["bg"])
        tag_emoji.pack(side="left")

        tag_right = tk.Label(
            ctrl_row, text=" Toggle to share your screen with the AI",
            font=("Courier New", 8, "bold"),
            fg=T["accent"], bg=T["bg"])
        tag_right.pack(side="left")

        # Stash refs so theme switching can recolor cleanly.
        self._tagline_widgets = (tag_left, tag_emoji, tag_right)

        tk.Button(ctrl_row, text="⌫ Clear",
                  font=("Courier New", 8),
                  fg=T["fg_dim"], bg=T["bg"],
                  relief="flat", cursor="hand2",
                  padx=8, pady=2,
                  command=self._clear_chat).pack(side="right")

        # ── Chat feed ──
        tk.Frame(main, bg=T["border"], height=1).pack(fill="x")
        self.feed = scrolledtext.ScrolledText(
            main, bg=T["bg_mid"], fg=T["fg"],
            font=("Courier New", 10),
            relief="flat", bd=0,
            insertbackground=T["accent"],
            wrap=tk.WORD, padx=14, pady=10,
            state="disabled")
        self.feed.pack(fill="both", expand=True, padx=0, pady=0)
        self._configure_feed_tags()
        self._register_drop_target(self.feed)

        # ── Input row ──
        input_frame = tk.Frame(main, bg=T["bg"])
        input_frame.pack(fill="x", padx=12, pady=(4, 4))

        self.capture_btn = tk.Button(
            input_frame, text="📷 OFF",
            font=("Courier New", 9, "bold"),
            fg=T["fg_dim"], bg=T["bg_mid"],
            relief="flat", cursor="hand2",
            padx=8, pady=8,
            command=self._toggle_capture)
        self.capture_btn.pack(side="left")

        self.capture_picker_btn = tk.Button(
            input_frame, text="▾",
            font=("Courier New", 9, "bold"),
            fg=T["fg_dim"], bg=T["bg_mid"],
            relief="flat", cursor="hand2",
            padx=4, pady=8,
            command=self._open_capture_picker)
        self.capture_picker_btn.pack(side="left", padx=(2, 6))

        self.input_var = tk.StringVar()
        self.input_box = tk.Entry(
            input_frame, textvariable=self.input_var,
            font=("Courier New", 10),
            fg=T["fg"], bg=T["bg_mid"],
            insertbackground=T["accent"],
            relief="flat", bd=0)
        self.input_box.pack(side="left", fill="x", expand=True,
                            ipady=8, padx=(0, 6))
        self.input_box.bind("<Return>",   self._on_send)
        self.input_box.bind("<KP_Enter>", self._on_send)

        self.send_btn = tk.Button(
            input_frame, text="▶  Send",
            font=("Courier New", 10, "bold"),
            fg=T["bg"], bg=T["accent"],
            relief="flat", cursor="hand2",
            padx=16, pady=8,
            command=self._on_send)
        self.send_btn.pack(side="right")

        # ── Footer (theme + prompt dropdown) ──
        footer = tk.Frame(main, bg=T["bg"])
        footer.pack(fill="x", padx=14, pady=(2, 8))

        self.theme_row_label = tk.Label(
            footer, text="Theme:",
            font=("Courier New", 8),
            fg=T["fg_dim"], bg=T["bg"])
        self.theme_row_label.pack(side="left")
        self._theme_btns: dict[str, tk.Button] = {}
        for name in THEMES:
            # The custom slot's color comes from the user's saved
            # accent rather than the THEMES dict default. Label is
            # shown with a tiny "+" marker so users see it's the
            # editable one.
            if name == "custom":
                c = self.settings.get(
                    "custom_accent", THEMES["custom"]["accent"])
                label = "custom +"
            else:
                c = THEMES[name]["accent"]
                label = name
            b = tk.Button(
                footer, text=label,
                font=("Courier New", 8,
                      "bold" if name == self.theme_name else "normal"),
                fg=c, bg=T["bg"],
                relief="flat", cursor="hand2",
                padx=6, pady=2,
                command=lambda n=name: self._switch_theme(n))
            b.pack(side="left", padx=(4, 0))
            self._theme_btns[name] = b

        # Vertical separator + Preview toggle. The toggle controls
        # whether captured screenshots get a 2s preview-before-send
        # modal. Off by default for seamless capture; users who want
        # the safety net flip it on here.
        tk.Frame(footer, bg=T["fg_dim"], width=1, height=14).pack(
            side="left", padx=(10, 8), pady=2)
        self.preview_toggle_btn = tk.Button(
            footer, text=self._preview_btn_label(),
            font=("Courier New", 8),
            fg=self._preview_btn_fg(),
            bg=T["bg"],
            relief="flat", cursor="hand2",
            padx=6, pady=2,
            command=self._toggle_preview_setting)
        self.preview_toggle_btn.pack(side="left")

        self.version_label = tk.Label(
            footer, text=f"v{APP_VERSION}",
            font=("Courier New", 8),
            fg=T["fg_dim"], bg=T["bg"])
        self.version_label.pack(side="right")

        self.prompt_menu_btn = tk.Menubutton(
            footer, text="Prompt: Default",
            font=("Courier New", 8),
            fg=T["accent2"], bg=T["bg"],
            activebackground=T["bg_mid"], activeforeground=T["accent2"],
            relief="flat", cursor="hand2",
            padx=8, pady=2,
            indicatoron=False, bd=0)
        self.prompt_menu_btn.pack(side="right", padx=(0, 12))

        self.prompt_menu = tk.Menu(
            self.prompt_menu_btn, tearoff=False,
            bg=T["bg_mid"], fg=T["fg"],
            activebackground=T["accent"], activeforeground=T["bg"],
            relief="flat", bd=0)
        self.prompt_menu_btn["menu"] = self.prompt_menu

        # Welcome lines
        self._post_system_line(
            f"iSee v{APP_VERSION} — local screen-aware chat.")
        if not CAPTURE_AVAILABLE:
            self._post_system_line(
                "screen capture disabled — install with "
                "'pip install mss Pillow pygetwindow'")
        self._post_system_line(
            "Make sure Ollama is running and qwen3.5:9b is pulled.")

        # Extra hint on the very first launch — points new users at
        # the screen-vision feature so they don't miss the headline
        # capability. Only fires when there's no existing settings
        # file.
        if self.first_launch_flag and CAPTURE_AVAILABLE:
            self._post_system_line(
                "👋 First time? Try this: click 📷 on the input row, "
                "pick a window or region, then ask the assistant "
                "about what's on screen.")
            self._post_system_line(
                "Tip: custom prompts (footer dropdown) make a huge "
                "difference. A DaVinci or Photoshop expert prompt "
                "turns Qwen into a domain assistant — see the "
                "prompts/ folder for examples.")

        self.input_box.focus_set()

    # -- Sidebar (conversation history) -----------------------------

    def _build_sidebar(self) -> None:
        """Construct the left sidebar with the conversation list."""
        T = self.theme
        s = self.sidebar
        # Clear if rebuilding (theme switch reuses widgets, but
        # keeping this idempotent is cheap insurance).
        for child in s.winfo_children():
            child.destroy()

        # Header row inside the sidebar — title + new-chat button
        sb_head = tk.Frame(s, bg=T["bg_mid"])
        sb_head.pack(fill="x", padx=10, pady=(12, 6))

        tk.Label(sb_head, text="HISTORY",
                 font=("Courier New", 9, "bold"),
                 fg=T["fg_dim"], bg=T["bg_mid"]).pack(
            side="left", anchor="w")
        self.new_chat_btn = tk.Button(
            sb_head, text="+  New chat",
            font=("Courier New", 9, "bold"),
            fg=T["bg"], bg=T["accent"],
            relief="flat", cursor="hand2",
            padx=8, pady=2,
            command=self._on_new_chat)
        self.new_chat_btn.pack(side="right")

        # Scrollable container — Listbox is simplest and fits the
        # terminal aesthetic. We render entries as "Title — relative
        # time" lines.
        list_wrap = tk.Frame(s, bg=T["bg_mid"])
        list_wrap.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        sb = tk.Scrollbar(list_wrap)
        sb.pack(side="right", fill="y")
        self.conv_listbox = tk.Listbox(
            list_wrap,
            font=("Courier New", 9),
            fg=T["fg"], bg=T["bg_mid"],
            selectbackground=T["accent"],
            selectforeground=T["bg"],
            activestyle="none",
            relief="flat", bd=0,
            highlightthickness=0,
            yscrollcommand=sb.set,
            cursor="hand2",
            exportselection=False)
        self.conv_listbox.pack(side="left", fill="both", expand=True)
        sb.config(command=self.conv_listbox.yview)
        self.conv_listbox.bind("<Double-Button-1>",
                                self._on_conv_open)
        self.conv_listbox.bind("<Return>", self._on_conv_open)
        self.conv_listbox.bind("<Delete>", self._on_conv_delete_key)
        # Right-click → context menu (rename / delete).
        self.conv_listbox.bind("<Button-3>",
                                self._on_conv_right_click)

        # Cached id list parallel to listbox indices, populated by
        # _refresh_conv_list().
        self._conv_index_ids: list[str] = []

        # Footer with hints.
        hint = tk.Label(
            s,
            text="Double-click to open\nRight-click to rename/delete",
            font=("Courier New", 8),
            fg=T["fg_dim"], bg=T["bg_mid"], justify="left")
        hint.pack(side="bottom", fill="x", padx=10, pady=(0, 10))

        self._refresh_conv_list()

    def _toggle_sidebar(self) -> None:
        if self.sidebar_visible:
            self.sidebar.pack_forget()
            self.sidebar_sep.pack_forget()
            self.sidebar_toggle_btn.configure(text="☰")
            self.sidebar_visible = False
        else:
            # Re-pack BEFORE the main column. tkinter's pack-side=left
            # places newly-packed widgets to the right of existing
            # left-packed widgets, so we need to forget all three
            # and re-pack in order.
            self.sidebar.pack(side="left", fill="y",
                               before=self._main_col)
            self.sidebar_sep.pack(side="left", fill="y",
                                   before=self._main_col)
            self.sidebar_toggle_btn.configure(text="◀")
            self.sidebar_visible = True

    def _refresh_conv_list(self) -> None:
        """Repopulate the conversation listbox from disk."""
        if not hasattr(self, "conv_listbox"):
            return
        entries = list_conversations()
        self.conv_listbox.delete(0, "end")
        self._conv_index_ids = []
        for e in entries:
            title = e.get("title") or "(untitled)"
            # Mark the currently-loaded conversation with a bullet.
            mark = "● " if (self.conv_id and
                             e.get("id") == self.conv_id) else "  "
            # Trim title to fit the sidebar comfortably.
            shown = title if len(title) <= 26 else title[:25] + "…"
            self.conv_listbox.insert("end", mark + shown)
            self._conv_index_ids.append(e.get("id", ""))

    def _on_new_chat(self) -> None:
        """Save current conversation, then start a fresh one."""
        # Persist whatever's on screen now (if there's anything).
        self._persist_current_conversation()
        # Reset state.
        self.conv_id      = None
        self.conv_title   = ""
        self.conv_created = ""
        self.history      = []
        self.display      = []
        # Clear feed.
        self.feed.configure(state="normal")
        self.feed.delete("1.0", "end")
        self.feed.configure(state="disabled")
        self._post_system_line("new chat — start typing")
        self._refresh_conv_list()

    def _on_conv_open(self, event: tk.Event | None = None) -> str:
        sel = self.conv_listbox.curselection()
        if not sel:
            return "break"
        idx = sel[0]
        if idx >= len(self._conv_index_ids):
            return "break"
        target_id = self._conv_index_ids[idx]
        if not target_id:
            return "break"
        # If user has unsaved work in the current chat, save it first.
        if target_id != self.conv_id:
            self._persist_current_conversation()
            self._load_conversation_into_feed(target_id)
            self._refresh_conv_list()
        return "break"

    def _on_conv_delete_key(self, event: tk.Event) -> str:
        self._on_conv_delete_selected()
        return "break"

    def _on_conv_right_click(self, event: tk.Event) -> None:
        # Select the row under the cursor first
        try:
            idx = self.conv_listbox.nearest(event.y)
            self.conv_listbox.selection_clear(0, "end")
            self.conv_listbox.selection_set(idx)
        except tk.TclError:
            return
        T = self.theme
        menu = tk.Menu(
            self.conv_listbox, tearoff=False,
            bg=T["bg_mid"], fg=T["fg"],
            activebackground=T["accent"], activeforeground=T["bg"],
            relief="flat", bd=0)
        menu.add_command(label="Open",
                          command=self._on_conv_open)
        menu.add_command(label="Rename...",
                          command=self._on_conv_rename_selected)
        menu.add_separator()
        menu.add_command(label="Delete",
                          command=self._on_conv_delete_selected)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _on_conv_rename_selected(self) -> None:
        sel = self.conv_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self._conv_index_ids):
            return
        cid = self._conv_index_ids[idx]
        # Look up current title.
        current_title = ""
        for e in list_conversations():
            if e.get("id") == cid:
                current_title = e.get("title", "")
                break
        new_title = self._prompt_for_text(
            "Rename conversation",
            "New title:",
            current_title)
        if new_title is None:
            return
        rename_conversation(cid, new_title)
        # If renaming the currently-loaded conversation, update
        # in-memory title too.
        if cid == self.conv_id:
            self.conv_title = new_title or "(untitled)"
        self._refresh_conv_list()

    def _on_conv_delete_selected(self) -> None:
        sel = self.conv_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self._conv_index_ids):
            return
        cid = self._conv_index_ids[idx]
        # Find title for the prompt
        title = ""
        for e in list_conversations():
            if e.get("id") == cid:
                title = e.get("title") or "(untitled)"
                break
        if not messagebox.askyesno(
                "Delete conversation?",
                f"Delete '{title}'? This can't be undone.",
                parent=self.root):
            return
        delete_conversation(cid)
        # If the deleted conversation was the active one, clear the
        # current chat too (the on-disk file is gone).
        if cid == self.conv_id:
            self._on_new_chat()
        else:
            self._refresh_conv_list()

    def _prompt_for_text(self, title: str, prompt: str,
                         initial: str = "") -> str | None:
        """A minimal modal text input dialog. Returns None on cancel."""
        T = self.theme
        dlg = tk.Toplevel(self.root)
        dlg.title(title)
        dlg.configure(bg=T["bg"])
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)

        tk.Label(dlg, text=prompt,
                 font=("Courier New", 10),
                 fg=T["fg"], bg=T["bg"]).pack(
            anchor="w", padx=18, pady=(14, 6))

        var = tk.StringVar(value=initial)
        entry = tk.Entry(
            dlg, textvariable=var,
            font=("Courier New", 10),
            fg=T["fg"], bg=T["bg_mid"],
            insertbackground=T["accent"],
            relief="flat", bd=0, width=40)
        entry.pack(fill="x", padx=18, ipady=4)
        entry.focus_set()
        entry.select_range(0, "end")

        result: dict[str, str | None] = {"value": None}

        def _ok():
            result["value"] = var.get()
            dlg.destroy()

        def _cancel():
            result["value"] = None
            dlg.destroy()

        btns = tk.Frame(dlg, bg=T["bg"])
        btns.pack(fill="x", padx=18, pady=(10, 14))
        tk.Button(btns, text="Cancel",
                  font=("Courier New", 9),
                  fg=T["fg_dim"], bg=T["bg"],
                  relief="flat", cursor="hand2",
                  padx=12, pady=4,
                  command=_cancel).pack(side="right", padx=(8, 0))
        tk.Button(btns, text="OK",
                  font=("Courier New", 9, "bold"),
                  fg=T["bg"], bg=T["accent"],
                  relief="flat", cursor="hand2",
                  padx=14, pady=4,
                  command=_ok).pack(side="right")

        dlg.bind("<Return>",   lambda e: _ok())
        dlg.bind("<Escape>",   lambda e: _cancel())
        dlg.protocol("WM_DELETE_WINDOW", _cancel)

        # Center on parent
        dlg.update_idletasks()
        try:
            px = self.root.winfo_rootx()
            py = self.root.winfo_rooty()
            pw = self.root.winfo_width()
            ph = self.root.winfo_height()
            w  = dlg.winfo_width()
            h  = dlg.winfo_height()
            dlg.geometry(f"+{px + (pw - w) // 2}+{py + (ph - h) // 2}")
        except tk.TclError:
            pass

        self.root.wait_window(dlg)
        return result["value"]

    # -- Conversation persistence ----------------------------------

    def _persist_current_conversation(self) -> None:
        """
        Save the current conversation to disk, if there's anything
        worth saving. Called on new-chat, on opening another
        conversation, and on app exit.
        """
        # Nothing user-meaningful happened yet → don't write a file.
        if not self.history:
            return

        if self.conv_id is None:
            self.conv_id      = make_conversation_id()
            self.conv_created = datetime.now().isoformat(timespec="seconds")
        # Re-derive title each time so editing the first message
        # keeps it accurate. If the user explicitly renamed (via
        # the rename command), conv_title takes precedence.
        if not self.conv_title:
            self.conv_title = derive_title_from_history(self.display)

        save_conversation(
            self.conv_id, self.conv_title, self.conv_created,
            self.history, self.display)

    def _load_conversation_into_feed(self, conv_id: str) -> None:
        """Load a saved conversation into the chat feed and state."""
        data = load_conversation(conv_id)
        if data is None:
            self._post_error_line(
                f"Couldn't load conversation {conv_id}.")
            return
        # Reset everything, then replay.
        self.feed.configure(state="normal")
        self.feed.delete("1.0", "end")
        self.feed.configure(state="disabled")

        self.conv_id      = data.get("id") or conv_id
        self.conv_title   = data.get("title") or "(untitled)"
        self.conv_created = data.get("created") or ""
        self.history      = list(data.get("history") or [])
        self.display      = list(data.get("display") or [])

        # Render display entries to the feed.
        for item in self.display:
            role = item.get("role")
            content = item.get("content") or ""
            ts = item.get("ts") or ""
            marker = item.get("capture_marker")
            if role == "user":
                self._render_replay("YOU", "role_user",
                                    "body_user", content, ts, marker)
            elif role == "assistant":
                self._render_replay("ASSISTANT", "role_assistant",
                                    "body_assistant", content, ts,
                                    None)
            elif role == "system":
                self._render_replay("·", "role_system",
                                    "body_system", content, ts, None)
            elif role == "error":
                self._render_replay("ERROR", "error",
                                    "error", content, ts, None)
        # A short marker so users can see we just opened an old chat.
        self._post_system_line(
            f"loaded conversation: {self.conv_title}")

    def _render_replay(self, role_label: str, role_tag: str,
                       body_tag: str, text: str, ts: str,
                       capture_marker: str | None) -> None:
        """Like _append but uses the saved timestamp instead of now()."""
        self.feed.configure(state="normal")
        if role_tag == "error":
            self.feed.insert("end", f"\n{role_label} ", "error")
        else:
            self.feed.insert("end", f"\n{role_label} ", role_tag)
        self.feed.insert("end", f"{ts}\n", "timestamp")
        if capture_marker:
            self.feed.insert("end",
                             f"  {capture_marker}\n",
                             "capture_marker")
        self.feed.insert("end", text + "\n", body_tag)
        self.feed.configure(state="disabled")
        self.feed.see("end")

    # -- Theme handling ----------------------------------------------

    def _configure_feed_tags(self) -> None:
        T = self.theme
        f = self.feed
        f.tag_configure("role_user",      foreground=T["accent2"],
                        font=("Courier New", 10, "bold"),
                        spacing1=8)
        f.tag_configure("role_assistant", foreground=T["accent"],
                        font=("Courier New", 10, "bold"),
                        spacing1=8)
        f.tag_configure("role_system",    foreground=T["fg_dim"],
                        font=("Courier New", 9, "italic"))
        f.tag_configure("body_user",      foreground=T["fg"])
        f.tag_configure("body_assistant", foreground=T["fg"])
        f.tag_configure("body_system",    foreground=T["fg_dim"],
                        font=("Courier New", 9, "italic"))
        f.tag_configure("error",          foreground=T["bearish"])
        f.tag_configure("dim",             foreground=T["fg_dim"])
        f.tag_configure("accent",          foreground=T["accent"])
        f.tag_configure("capture_marker",  foreground=T["accent2"],
                        font=("Courier New", 9, "italic"))
        f.tag_configure("timestamp",       foreground=T["fg_dim"],
                        font=("Courier New", 8))

    def _apply_theme(self) -> None:
        T = self.theme
        r = self.root
        r.configure(bg=T["bg"])

        # Build sets of "known" bg colors from every theme we know
        # about, including the currently-active custom palette and
        # the previously-active palette (so a switch from custom to
        # terminal still finds the old custom-accent-tinted frames).
        # Without this, dynamically-derived custom themes would leave
        # frames un-repainted after a switch.
        all_bg      = set()
        all_bg_mid  = set()
        all_bg_bar  = set()
        all_borders = set()
        for tn in THEMES:
            t = (self._resolve_theme(tn) if tn == "custom"
                 else THEMES[tn])
            all_bg.add(t["bg"])
            all_bg_mid.add(t["bg_mid"])
            all_bg_bar.add(t["bg_bar"])
            all_borders.add(t["border"])
        # Also include whatever was painted last time (in case the
        # custom accent has been changed since startup).
        if hasattr(self, "_last_palette"):
            lp = self._last_palette
            all_bg.add(lp.get("bg", ""))
            all_bg_mid.add(lp.get("bg_mid", ""))
            all_bg_bar.add(lp.get("bg_bar", ""))
            all_borders.add(lp.get("border", ""))

        def walk(widget: tk.Misc) -> None:
            try:
                cls = widget.winfo_class()
                if cls == "Frame":
                    cur_bg = widget.cget("bg")
                    if cur_bg in all_bg:
                        widget.configure(bg=T["bg"])
                    elif cur_bg in all_bg_mid:
                        widget.configure(bg=T["bg_mid"])
                    elif cur_bg in all_bg_bar:
                        widget.configure(bg=T["bg_bar"])
                    elif cur_bg in all_borders:
                        widget.configure(bg=T["border"])
            except tk.TclError:
                pass
            for child in widget.winfo_children():
                walk(child)
        walk(r)
        # Remember the palette we just painted so the NEXT switch
        # can find these frames.
        self._last_palette = dict(T)

        try:
            # Header / chrome labels — these need explicit recoloring
            # because they live in the widget tree but aren't reached
            # by the Frame-walk above. Using stable references keeps
            # the recolor cheap and reliable across theme switches
            # (including dynamic custom palettes).
            self.brand_label.configure(fg=T["accent"], bg=T["bg"])
            self.brand_eye.configure(fg=T["accent2"], bg=T["bg"])
            self.signal_label.configure(fg=T["accent"], bg=T["bg"])
            self.status_label.configure(
                fg=T["fg_status"], bg=T["bg_bar"])
            self.theme_row_label.configure(
                fg=T["fg_dim"], bg=T["bg"])
            self.version_label.configure(
                fg=T["fg_dim"], bg=T["bg"])
            self.feed.configure(bg=T["bg_mid"], fg=T["fg"],
                                insertbackground=T["accent"])
            self.input_box.configure(bg=T["bg_mid"], fg=T["fg"],
                                     insertbackground=T["accent"])
            self.send_btn.configure(bg=T["accent"], fg=T["bg"])
            self.prompt_menu_btn.configure(
                fg=T["accent2"], bg=T["bg"],
                activebackground=T["bg_mid"],
                activeforeground=T["accent2"])
            self.prompt_menu.configure(
                bg=T["bg_mid"], fg=T["fg"],
                activebackground=T["accent"],
                activeforeground=T["bg"])
            self._configure_feed_tags()
            for name, btn in self._theme_btns.items():
                if name == "custom":
                    c = self.settings.get(
                        "custom_accent",
                        THEMES["custom"]["accent"])
                else:
                    c = THEMES[name]["accent"]
                btn.configure(
                    fg=c, bg=T["bg"],
                    font=("Courier New", 8,
                          "bold" if name == self.theme_name else "normal"))
            self._update_capture_button()
            self._update_preview_button()
            # Repaint the highlighted tagline so the accent color
            # follows the active theme.
            try:
                if hasattr(self, "_tagline_widgets"):
                    tl, te, tr = self._tagline_widgets
                    tl.configure(fg=T["fg_dim"], bg=T["bg"])
                    te.configure(fg=T["accent"], bg=T["bg"])
                    tr.configure(fg=T["accent"], bg=T["bg"])
            except (AttributeError, tk.TclError):
                pass
            # Repaint sidebar containers + chrome
            try:
                self.sidebar.configure(bg=T["bg_mid"])
                self.sidebar_sep.configure(bg=T["border"])
                self.sidebar_toggle_btn.configure(
                    fg=T["fg_dim"], bg=T["bg"])
                self._build_sidebar()    # rebuild children with new theme
            except (AttributeError, tk.TclError):
                pass
        except (AttributeError, tk.TclError):
            pass

    def _switch_theme(self, name: str) -> None:
        if name not in THEMES:
            return
        # The custom button always opens the color picker. If the
        # user just wants to swap *back* to their saved custom
        # theme without changing the color, they can cancel the
        # picker — settings preserved either way.
        if name == "custom":
            self._pick_custom_color()
            return
        if name == self.theme_name:
            return
        self.theme_name = name
        self.theme      = self._resolve_theme(name)
        self.settings["active_theme"] = name
        save_settings(self.settings)
        self._apply_theme()
        self._post_system_line(f"theme switched to '{name}'")

    def _resolve_theme(self, name: str) -> dict[str, str]:
        """
        Return the theme dict for `name`, applying any user-chosen
        custom_accent override when the theme is "custom".
        """
        if name == "custom":
            accent = self.settings.get(
                "custom_accent",
                THEMES["custom"]["accent"])
            return derive_custom_theme(accent)
        return dict(THEMES.get(name, THEMES["terminal"]))

    def _pick_custom_color(self) -> None:
        """
        Open the OS color picker for the user's custom-theme accent.
        On confirm, derive a fresh theme from the chosen color and
        apply it. Theme switches to "custom" automatically.
        """
        from tkinter import colorchooser
        current = self.settings.get(
            "custom_accent", THEMES["custom"]["accent"])
        try:
            chosen = colorchooser.askcolor(
                color=current,
                title="Pick a custom accent color",
                parent=self.root)
        except tk.TclError:
            return
        if not chosen or not chosen[1]:
            return    # user cancelled
        accent_hex = chosen[1]
        self.settings["custom_accent"] = accent_hex
        # Switch active theme to custom + apply.
        self.theme_name = "custom"
        self.settings["active_theme"] = "custom"
        self.theme = self._resolve_theme("custom")
        save_settings(self.settings)
        self._apply_theme()
        self._post_system_line(
            f"custom accent set to {accent_hex}")

    # -- Custom prompts ----------------------------------------------

    def _get_active_custom_prompt(self) -> str | None:
        active = self.settings.get("active_prompt_name")
        if not active:
            return None
        for p in self.settings.get("custom_prompts", []):
            if p["name"] == active:
                return p.get("content", "")
        # active_prompt_name pointed to a deleted prompt — repair.
        self.settings["active_prompt_name"] = None
        save_settings(self.settings)
        return None

    def _switch_prompt(self, name: str | None) -> None:
        prev = self.settings.get("active_prompt_name")
        if name == prev:
            return
        if name is not None:
            valid = any(p["name"] == name
                        for p in self.settings.get("custom_prompts", []))
            if not valid:
                name = None
        self.settings["active_prompt_name"] = name
        save_settings(self.settings)
        self._refresh_prompt_menu()
        self._update_prompt_label()
        label = name if name else "Default"
        self._post_system_line(f"switched prompt to '{label}'")

    def _refresh_prompt_menu(self) -> None:
        self.prompt_menu.delete(0, "end")
        active = self.settings.get("active_prompt_name")

        def _add(label, cmd):
            self.prompt_menu.add_command(label=label, command=cmd)

        prefix_default = "● " if not active else "   "
        _add(f"{prefix_default}Default",
             lambda: self._switch_prompt(None))

        prompts = self.settings.get("custom_prompts", [])
        if prompts:
            self.prompt_menu.add_separator()
            for p in prompts:
                pref = "● " if active == p["name"] else "   "
                name = p["name"]
                _add(f"{pref}{name}",
                     lambda n=name: self._switch_prompt(n))

        self.prompt_menu.add_separator()
        _add("+ New prompt...",  self._open_new_prompt_dialog)
        _add("Manage prompts...", self._open_manage_prompts_dialog)

    def _update_prompt_label(self) -> None:
        active = self.settings.get("active_prompt_name")
        label  = active if active else "Default"
        if len(label) > 22:
            label = label[:19] + "…"
        self.prompt_menu_btn.configure(text=f"Prompt: {label}")

    def _open_manage_prompts_dialog(self,
                                     start_in_new_mode: bool = False
                                     ) -> None:
        active = self.settings.get("active_prompt_name")
        try:
            dlg = ManagePromptsDialog(
                self.root, self.settings,
                on_change=self._on_prompts_changed,
                select_name=active,
                start_in_new_mode=start_in_new_mode)
            self.root.wait_window(dlg)
        except tk.TclError as e:
            messagebox.showerror("Dialog error", str(e))

    def _open_new_prompt_dialog(self) -> None:
        self._open_manage_prompts_dialog(start_in_new_mode=True)

    def _on_prompts_changed(self) -> None:
        self._refresh_prompt_menu()
        self._update_prompt_label()

    # -- Preview-before-send toggle ---------------------------------

    def _preview_btn_label(self) -> str:
        on = bool(self.settings.get("preview_before_send", False))
        return "Snapshot Preview: ON" if on else "Snapshot Preview: off"

    def _preview_btn_fg(self) -> str:
        T = self.theme
        on = bool(self.settings.get("preview_before_send", False))
        return T["accent"] if on else T["fg_dim"]

    def _toggle_preview_setting(self) -> None:
        new_val = not bool(
            self.settings.get("preview_before_send", False))
        self.settings["preview_before_send"] = new_val
        save_settings(self.settings)
        self._update_preview_button()
        self._post_system_line(
            f"snapshot preview {'ON (5s countdown)' if new_val else 'OFF'}")

    def _update_preview_button(self) -> None:
        try:
            self.preview_toggle_btn.configure(
                text=self._preview_btn_label(),
                fg=self._preview_btn_fg())
        except (AttributeError, tk.TclError):
            pass

    # -- Capture toggle / target picker ------------------------------

    def _update_capture_button(self) -> None:
        T = self.theme
        try:
            if not CAPTURE_AVAILABLE:
                self.capture_btn.configure(
                    text="📷 N/A",
                    fg=T["fg_dim"], bg=T["bg_mid"],
                    state="disabled", cursor="")
                self.capture_picker_btn.configure(
                    fg=T["fg_dim"], bg=T["bg_mid"],
                    state="disabled", cursor="")
                return

            self.capture_picker_btn.configure(
                fg=T["fg_dim"], bg=T["bg_mid"],
                state="normal", cursor="hand2")
            if self.capture_enabled:
                self.capture_btn.configure(
                    text="📷 ON",
                    fg=T["bg"], bg=T["accent"],
                    state="normal", cursor="hand2")
            else:
                self.capture_btn.configure(
                    text="📷 OFF",
                    fg=T["fg_dim"], bg=T["bg_mid"],
                    state="normal", cursor="hand2")
        except (AttributeError, tk.TclError):
            pass

    def _toggle_capture(self) -> None:
        if not CAPTURE_AVAILABLE:
            self._post_error_line(
                "Screen capture requires: pip install mss Pillow")
            return

        if self.capture_enabled:
            # ON → OFF
            self.capture_enabled = False
            self._update_capture_button()
            self._post_system_line("📷 capture OFF")
            return

        # OFF → ON. Need a target first.
        if self.capture_mode is None or (
                self.capture_mode in ("region", "window")
                and self.capture_target is None):
            self._open_capture_picker(after_pick_enable=True)
            return

        # Have a target already → just turn ON.
        self.capture_enabled = True
        self._update_capture_button()
        self._post_system_line(f"📷 capture ON — {self.capture_label}")

    def _open_capture_picker(self,
                             after_pick_enable: bool = False) -> None:
        if not CAPTURE_AVAILABLE:
            self._post_error_line(
                "Screen capture requires: pip install mss Pillow")
            return

        try:
            dlg = CaptureModeDialog(self.root, self.theme)
            self.root.wait_window(dlg)
        except tk.TclError as e:
            messagebox.showerror("Dialog error", str(e))
            return

        if dlg.result is None:
            return    # cancelled

        if dlg.result == "full_screen":
            self._set_capture_target(
                "full_screen", None, "full screen",
                auto_enable=after_pick_enable)
        elif dlg.result == "region":
            self._launch_region_selector(after_pick_enable=after_pick_enable)
        elif dlg.result == "window":
            self._launch_window_picker(after_pick_enable=after_pick_enable)

    def _launch_region_selector(self,
                                after_pick_enable: bool = False) -> None:
        try:
            sel = RegionSelector(self.root,
                                 accent_color=self.theme["accent"])
            self.root.wait_window(sel.win)
        except tk.TclError as e:
            messagebox.showerror("Region selector error", str(e))
            return
        if not sel.result:
            return
        l, t, r, b = sel.result
        label = f"region — {r - l}×{b - t}px @ ({l},{t})"
        self._set_capture_target(
            "region", sel.result, label, auto_enable=after_pick_enable)

    def _launch_window_picker(self,
                              after_pick_enable: bool = False) -> None:
        if not HAS_PYGETWINDOW:
            self._post_error_line(
                "Window picker requires: pip install pygetwindow")
            return
        try:
            picker = WindowPicker(self.root, self.theme)
            self.root.wait_window(picker)
        except tk.TclError as e:
            messagebox.showerror("Window picker error", str(e))
            return
        if not picker.result:
            return
        l, t, r, b = picker.result
        title = picker.selected_window_title or "window"
        short = title if len(title) <= 40 else title[:37] + "…"
        label = f"window — {short} ({r - l}×{b - t}px)"
        self._set_capture_target(
            "window", picker.result, label,
            auto_enable=after_pick_enable)

    def _set_capture_target(self, mode: str,
                            target: tuple[int, int, int, int] | None,
                            label: str,
                            auto_enable: bool = False) -> None:
        self.capture_mode   = mode
        self.capture_target = target
        self.capture_label  = label
        self._post_system_line(f"capture target: {label}")

        if auto_enable:
            self.capture_enabled = True
            self._post_system_line("📷 capture ON")
        self._update_capture_button()

    # -- Chat log helpers --------------------------------------------

    def _post_user_line(self, text: str,
                        capture_marker: str | None = None) -> None:
        self._append("YOU", "role_user", "body_user", text,
                     capture_marker=capture_marker,
                     display_role="user")

    def _post_assistant_line(self, text: str) -> None:
        self._append("ASSISTANT", "role_assistant",
                     "body_assistant", text,
                     display_role="assistant")

    def _post_system_line(self, text: str) -> None:
        self._append("·", "role_system", "body_system", text,
                     display_role="system")

    def _post_error_line(self, text: str) -> None:
        self._append("ERROR", "role_assistant", "error", text,
                     display_role="error")

    def _append(self, role_label: str, role_tag: str,
                body_tag: str, text: str,
                capture_marker: str | None = None,
                display_role: str | None = None) -> None:
        self.feed.configure(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        if role_tag == "role_assistant" and body_tag == "error":
            self.feed.insert("end", f"\n{role_label} ", "error")
        else:
            self.feed.insert("end", f"\n{role_label} ", role_tag)
        self.feed.insert("end", f"{ts}\n", "timestamp")
        if capture_marker:
            self.feed.insert("end",
                             f"  {capture_marker}\n",
                             "capture_marker")
        self.feed.insert("end", text + "\n", body_tag)
        self.feed.configure(state="disabled")
        self.feed.see("end")

        # Record the entry in the parallel display log so the
        # conversation can be persisted and replayed later.
        if display_role:
            entry: dict[str, Any] = {
                "role":    display_role,
                "content": text,
                "ts":      ts,
            }
            if capture_marker:
                entry["capture_marker"] = capture_marker
            self.display.append(entry)

    def _clear_chat(self) -> None:
        if self._call_in_flight:
            if not messagebox.askyesno(
                    "Clear during call?",
                    "A reply is still being generated. Clear anyway? "
                    "The reply will still be added when it returns.",
                    parent=self.root):
                return
        self.feed.configure(state="normal")
        self.feed.delete("1.0", "end")
        self.feed.configure(state="disabled")
        self.history = []
        self.display = []
        self._post_system_line("chat cleared")

    # -- Conversation history ---------------------------------------

    def _trim_history(self) -> None:
        max_msgs = MAX_HISTORY_PAIRS * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    # -- Send / Qwen call -------------------------------------------

    def _on_send(self, event: tk.Event | None = None) -> str:
        if self._call_in_flight:
            return "break"
        text = self.input_var.get().strip()
        if not text:
            return "break"

        # If capture is ON, grab the screen BEFORE clearing the input
        # box so the user can re-send if capture fails.
        image_b64       : str | None = None
        capture_marker  : str | None = None
        captured_img = None    # PIL.Image for optional preview thumb
        if self.capture_enabled:
            try:
                captured_img  = self._do_capture()
                image_b64     = encode_image_b64(captured_img)
                capture_marker = f"📷 attached: {self.capture_label}"
            except Exception as e:    # noqa: BLE001
                self._post_error_line(
                    f"Screen capture failed: {e}\n"
                    "Disable 📷 or pick a new target with ▾.")
                return "break"

        # Optional preview-before-send modal. Only fires when both
        # capture is on AND preview is enabled in settings AND we
        # actually have an image. The user has 2 seconds to hit
        # Cancel; otherwise we auto-send.
        if (self.capture_enabled
                and self.settings.get("preview_before_send", False)
                and captured_img is not None):
            decision = self._show_capture_preview(
                captured_img, text, capture_marker or "")
            if decision == "cancel":
                # Restore the prompt so the user doesn't have to
                # retype it.
                self.input_var.set(text)
                self._post_system_line(
                    "capture cancelled — prompt restored")
                return "break"

        self.input_var.set("")
        self._post_user_line(text, capture_marker=capture_marker)
        self.history.append({"role": "user", "content": text})
        self._trim_history()

        self._begin_call()

        messages_snapshot = list(self.history)
        custom = self._get_active_custom_prompt()
        screen_vision_enabled = self.capture_enabled

        threading.Thread(
            target=self._qwen_worker,
            args=(messages_snapshot, custom, image_b64,
                  screen_vision_enabled),
            daemon=True,
        ).start()
        return "break"

    def _show_capture_preview(self, pil_image, prompt_text: str,
                               capture_label: str) -> str:
        """
        Show the PreviewDialog with a downscaled thumbnail of
        `pil_image`. Blocks until the modal closes. Returns
        "send" or "cancel".
        """
        if not HAS_PIL:
            return "send"    # can't preview without Pillow
        try:
            # Build a thumbnail that fits the modal — max 480x320
            # preserves aspect, plenty big enough to verify content.
            thumb = pil_image.copy()
            thumb.thumbnail((480, 320))
            # Convert PIL → tk via temp file (most compatible path
            # across tk versions).
            import tempfile
            tmp = Path(tempfile.gettempdir()) / "isee_preview.png"
            thumb.save(tmp, format="PNG")
            photo = tk.PhotoImage(file=str(tmp))
        except Exception as e:    # noqa: BLE001
            # If the preview itself fails, fall through to send.
            write_debug_log(
                "preview_thumb_error",
                f"{type(e).__name__}: {e}")
            return "send"

        try:
            dlg = PreviewDialog(
                self.root, self.theme,
                thumb_photo=photo,
                prompt_text=prompt_text,
                capture_label=capture_label)
            self.root.wait_window(dlg)
            return dlg.result or "send"
        except tk.TclError:
            return "send"

    def _do_capture(self):
        mode = self.capture_mode
        if mode == "full_screen":
            return capture_full_screen()
        if mode in ("region", "window"):
            if self.capture_target is None:
                raise RuntimeError(
                    "Capture target lost. Pick a new one with ▾.")
            return capture_region(self.capture_target)
        raise RuntimeError(
            "No capture mode set. Pick a target with ▾ first.")

    # -- Drag-and-drop image support --------------------------------

    _DND_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif",
                        ".bmp", ".webp")

    def _register_drop_target(self, widget: tk.Misc) -> None:
        """
        Register a widget as a target for OS file drops. No-op when
        tkinterdnd2 isn't installed or when the root is plain tk.Tk
        (fallback path inside main()).
        """
        if not HAS_DND:
            return
        try:
            widget.drop_target_register(DND_FILES)    # type: ignore
            widget.dnd_bind("<<Drop>>", self._on_file_drop)    # type: ignore
            widget.dnd_bind("<<DropEnter>>",    # type: ignore
                             self._on_drop_enter)
            widget.dnd_bind("<<DropLeave>>",    # type: ignore
                             self._on_drop_leave)
        except (tk.TclError, AttributeError):
            # Root wasn't TkinterDnD-aware (fell back to tk.Tk()).
            pass

    @staticmethod
    def _parse_dnd_paths(raw: str) -> list[str]:
        """
        tkdnd's <<Drop>> event.data is a string of paths separated
        by spaces. Paths containing spaces are wrapped in {curly
        braces}. This parses that format into a clean list.
        """
        if not raw:
            return []
        out: list[str] = []
        i = 0
        n = len(raw)
        while i < n:
            # Skip whitespace
            while i < n and raw[i].isspace():
                i += 1
            if i >= n:
                break
            if raw[i] == "{":
                # Brace-quoted path
                j = raw.find("}", i + 1)
                if j == -1:
                    out.append(raw[i + 1:])
                    break
                out.append(raw[i + 1:j])
                i = j + 1
            else:
                # Unquoted path — runs until next whitespace
                j = i
                while j < n and not raw[j].isspace():
                    j += 1
                out.append(raw[i:j])
                i = j
        return out

    def _on_drop_enter(self, event: tk.Event) -> str:
        """Visual feedback when a file is being dragged over."""
        try:
            T = self.theme
            self.feed.configure(highlightthickness=2,
                                 highlightbackground=T["accent"],
                                 highlightcolor=T["accent"])
            self.status_var.set("⤓  drop image to attach")
        except tk.TclError:
            pass
        return event.action if hasattr(event, "action") else "copy"

    def _on_drop_leave(self, event: tk.Event) -> str:
        try:
            self.feed.configure(highlightthickness=0)
            self.status_var.set("ready")
        except tk.TclError:
            pass
        return "default"

    def _on_file_drop(self, event: tk.Event) -> str:
        """Handler for OS file drops onto the chat feed."""
        # Restore the feed's normal border first.
        try:
            self.feed.configure(highlightthickness=0)
        except tk.TclError:
            pass

        if self._call_in_flight:
            self._post_error_line(
                "Wait for the current reply before dropping a "
                "new image.")
            return "break"

        raw = getattr(event, "data", "") or ""
        paths = self._parse_dnd_paths(raw)
        if not paths:
            return "break"
        # If user dropped multiple files, take the FIRST image and
        # tell them about the rest. Multi-image per turn would need
        # work upstream in the Qwen call payload.
        chosen: str | None = None
        skipped: list[str] = []
        for p in paths:
            if p.lower().endswith(self._DND_IMAGE_EXTS):
                if chosen is None:
                    chosen = p
                else:
                    skipped.append(p)
            else:
                skipped.append(p)
        if chosen is None:
            self._post_error_line(
                f"No supported image in drop ({', '.join(paths[:3])}). "
                f"Supported: PNG, JPG, GIF, BMP, WebP.")
            return "break"
        if skipped:
            self._post_system_line(
                f"only the first image was attached; ignored: "
                f"{len(skipped)} other file(s)")

        # Open and encode. Reuse the same encode path as screenshots.
        if not HAS_PIL:
            self._post_error_line(
                "Image drop needs Pillow. Run: pip install Pillow")
            return "break"
        try:
            img = Image.open(chosen).convert("RGB")
            image_b64 = encode_image_b64(img)
        except Exception as e:    # noqa: BLE001
            self._post_error_line(
                f"Couldn't read image '{chosen}': {e}")
            return "break"

        # Check for prompt text in the input box. Drag-drop with no
        # text means "look at this image" (use a default prompt);
        # with text means "look at this image and answer my
        # question."
        text = self.input_var.get().strip()
        if not text:
            text = "What's in this image?"
        self.input_var.set("")

        filename = os.path.basename(chosen) or chosen
        marker   = f"📎 dropped: {filename}"
        self._post_user_line(text, capture_marker=marker)
        self.history.append({"role": "user", "content": text})
        self._trim_history()

        self._begin_call()

        messages_snapshot = list(self.history)
        custom = self._get_active_custom_prompt()

        threading.Thread(
            target=self._qwen_worker,
            args=(messages_snapshot, custom, image_b64,
                  True),    # screen_vision_enabled=True for the prompt
            daemon=True,
        ).start()
        return "break"

    def _begin_call(self) -> None:
        T = self.theme
        self._call_in_flight = True
        self.send_btn.configure(state="disabled", text="...")
        if self.capture_enabled:
            self.status_var.set(
                "⚡ thinking — Qwen 3.5:9b reading screen...")
        else:
            self.status_var.set(
                "⚡ thinking — Qwen 3.5:9b "
                "(first call may take 60-90s)")
        self.signal_label.configure(text="● THINKING", fg=T["watch"])

    def _end_call(self) -> None:
        T = self.theme
        self._call_in_flight = False
        self.send_btn.configure(state="normal", text="▶  Send")
        self.status_var.set("ready")
        self.signal_label.configure(text="● READY", fg=T["accent"])

    def _qwen_worker(self, messages: list[dict[str, Any]],
                     custom_prompt: str | None,
                     image_b64: str | None,
                     screen_vision_enabled: bool) -> None:
        try:
            cleaned, raw = call_qwen(
                messages, custom_prompt,
                image_b64=image_b64,
                screen_vision_enabled=screen_vision_enabled)
            self.root.after(0, self._handle_qwen_result, cleaned, raw)
        except Exception as exc:    # noqa: BLE001
            self.root.after(0, self._handle_qwen_error, exc)

    def _handle_qwen_result(self, cleaned: str, raw: str) -> None:
        write_debug_log("qwen_response", raw)
        if not cleaned:
            self._post_error_line(
                "Qwen returned an empty response. Raw output saved "
                "to qwen_debug.log.")
        else:
            self.history.append({"role": "assistant", "content": cleaned})
            self._trim_history()
            self._post_assistant_line(cleaned)
        self._end_call()

    def _handle_qwen_error(self, exc: BaseException) -> None:
        msg = self._format_qwen_error(exc)
        write_debug_log("qwen_error",
                        f"{type(exc).__name__}: {exc}\n"
                        f"{traceback.format_exc()}")
        self._post_error_line(msg)
        if self.history and self.history[-1].get("role") == "user":
            self.history.pop()
        self._end_call()

    def _format_qwen_error(self, exc: BaseException) -> str:
        if isinstance(exc, requests.exceptions.ConnectionError):
            return (f"Couldn't connect to Ollama at {QWEN_ENDPOINT}.\n"
                    f"  • Make sure Ollama is running "
                    f"('ollama serve' in a terminal)\n"
                    f"  • Make sure {QWEN_MODEL} is pulled "
                    f"('ollama pull {QWEN_MODEL}')")
        if isinstance(exc, requests.exceptions.Timeout):
            return (f"Qwen took longer than {QWEN_TIMEOUT}s to reply. "
                    f"Ollama may be stuck — try restarting it.")
        if isinstance(exc, requests.exceptions.HTTPError):
            resp = getattr(exc, "response", None)
            body = ""
            if resp is not None:
                try:
                    body = resp.text[:500]
                except Exception:    # noqa: BLE001
                    body = ""
            return f"Ollama HTTP error: {exc}\n{body}"
        if isinstance(exc, json.JSONDecodeError):
            return ("Ollama returned a non-JSON response. See "
                    "qwen_debug.log for raw output.")
        return f"Qwen call failed: {type(exc).__name__}: {exc}"


# ═══════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    # Use the TkinterDnD-aware root when the lib is installed so
    # drop_target_register works on child widgets. Falls back to a
    # plain tk.Tk() otherwise. If TkinterDnD itself fails to init
    # (rare — usually a tkdnd binary issue) we also fall back.
    root: tk.Tk
    try:
        if HAS_DND:
            try:
                root = TkinterDnD.Tk()
            except Exception:    # noqa: BLE001
                root = tk.Tk()
        else:
            root = tk.Tk()
    except tk.TclError as e:
        print(f"Failed to start tkinter: {e}", file=sys.stderr)
        return 2

    app: ISeeApp | None = None
    try:
        app = ISeeApp(root)
    except Exception:
        traceback.print_exc()
        try:
            messagebox.showerror(
                "Startup error",
                "iSee failed to start. See console for details.")
        except tk.TclError:
            pass
        return 1

    def _on_close():
        # Persist whatever's on screen before exiting.
        if app is not None:
            try:
                app._persist_current_conversation()
            except Exception:    # noqa: BLE001
                # Don't let a save error block app exit.
                traceback.print_exc()
            # Save current window geometry so the next launch
            # restores it.
            try:
                geom = root.geometry()
                if geom:
                    app.settings["window_geometry"] = geom
                    save_settings(app.settings)
            except Exception:    # noqa: BLE001
                pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)

    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
