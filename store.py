"""Remembered options and saved conversations, all kept on this computer.

Everything lives in one folder the user can open, back up or delete. Nothing
is ever sent anywhere, and conversations are only saved if the user turns it on.
"""

import json
import re
from datetime import datetime
from pathlib import Path

DEFAULT_SETTINGS = {
    "system_prompt": "You are a helpful assistant.",
    "max_new": 1024,        # answer length limit, in tokens
    "temperature": 0.7,
    "speculative": True,
    "save_chats": False,    # saving conversations is opt-in
}


def _right_type(value, default):
    """True when a value read from the settings file has a sensible type."""
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, (int, float)):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, type(default))


def _title_of(messages):
    """A short title from the first line of the first thing the user said."""
    first = next((m["content"] for m in messages if m["role"] == "user"), "").strip()
    return first.splitlines()[0][:80] if first else "Conversation"


def _valid_chat(data):
    """True when parsed JSON has the shape of a saved conversation."""
    try:
        return (isinstance(data["title"], str) and isinstance(data["created"], str)
                and all(m["role"] in ("user", "assistant") and isinstance(m["content"], str)
                        for m in data["messages"]))
    except (KeyError, TypeError):
        return False


def _to_markdown(data):
    lines = [f"# {data['title']}", "",
             f"Saved {data['updated'].replace('T', ' ')}, model {data['model'] or 'unknown'}", ""]
    for m in data["messages"]:
        lines += ["**You**" if m["role"] == "user" else "**Wiseacre**", "", m["content"], ""]
    return "\n".join(lines)


class Store:
    def __init__(self, root):
        self.root = Path(root).expanduser()
        self.settings_file = self.root / "settings.json"
        self.chat_dir = self.root / "chats"

    # -- options ----------------------------------------------------------

    def load_settings(self):
        settings = dict(DEFAULT_SETTINGS)
        try:
            saved = json.loads(self.settings_file.read_text(encoding="utf-8"))
            for key, default in DEFAULT_SETTINGS.items():
                if key in saved and _right_type(saved[key], default):
                    settings[key] = saved[key]
        except (OSError, ValueError, TypeError):
            pass  # missing or damaged file: use the defaults
        return settings

    def save_settings(self, settings):
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.settings_file.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        except OSError as e:
            print(f"Could not save options: {e}", flush=True)

    # -- conversations ----------------------------------------------------

    def save_chat(self, filename, messages, model=""):
        """Write a conversation, or update the one saved earlier. Returns its file name.
        A readable .md copy is written next to the .json file the app reopens."""
        self.chat_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now().isoformat(timespec="seconds")
        earlier = self._read(filename)
        title = _title_of(messages)
        if not earlier:
            filename = self._new_filename(title)

        data = {"title": title, "created": earlier["created"] if earlier else now,
                "updated": now, "model": model, "messages": messages}
        path = self.chat_dir / filename
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        path.with_suffix(".md").write_text(_to_markdown(data), encoding="utf-8")
        return filename

    def list_chats(self):
        """(label, file name) pairs for a dropdown, newest first."""
        choices = []
        for path in sorted(self.chat_dir.glob("*.json"), reverse=True):
            data = self._read(path.name)
            if data:
                when = datetime.fromisoformat(data["created"]).strftime("%b %d, %H:%M")
                choices.append((f"{when}   {data['title']}", path.name))
        return choices

    def load_chat(self, filename):
        data = self._read(filename)
        return data["messages"] if data else None

    # -- internals --------------------------------------------------------

    def _new_filename(self, title):
        """A readable, unique name such as 2026-11-09_12-30_write-three-problems.json"""
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "chat"
        base = f"{datetime.now():%Y-%m-%d_%H-%M}_{slug}"
        name, n = f"{base}.json", 2
        while (self.chat_dir / name).exists():
            name, n = f"{base}-{n}.json", n + 1
        return name

    def _read(self, filename):
        """One saved conversation, or None if it is missing, damaged or not ours."""
        if not filename or Path(filename).name != filename or not filename.endswith(".json"):
            return None  # only plain file names inside the chats folder are allowed
        try:
            data = json.loads((self.chat_dir / filename).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not _valid_chat(data):
            return None
        try:
            datetime.fromisoformat(data["created"])
        except ValueError:
            return None
        return data
