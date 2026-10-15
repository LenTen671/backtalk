# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hold-to-talk — a global key listener.

HOLD the key -> mic opens. RELEASE -> mic closes and the utterance is
processed. The button IS the voice-activity detector, which is why this
mode is speaker-safe with no headphones: the mic simply isn't open while
the assistant talks, unless you press the key — and pressing while it
talks interrupts it.

THE KEY-REPEAT TRAP (the bug that kills every naive build): the OS fires
on_press events CONTINUOUSLY while a key is held. Without the held-state
filter below, every repeat reads as a fresh press and keeps cancelling
the reply before it can speak.

AND THE HALF THAT TRAP HIDES: some keyboards send auto-repeat as full
DOWN/UP PAIRS rather than the repeated DOWN-only stream. Filtering the
presses and trusting every release then breaks the OTHER way -- a single
hold is chopped into dozens of ~50ms recordings, each too short to
transcribe, and the whole thing is SILENT. No exception, no log line,
nothing to search for; it simply reads as "the microphone does not work".
Measured in the field on a Logitech MX Mechanical through a Bolt
receiver: one 2.6-second hold produced 186 key events and about fifty
recordings. So a release is never trusted on sight -- see is_held().

macOS needs Input Monitoring permission for the hosting terminal
(System Settings -> Privacy & Security -> Input Monitoring). Windows
works out of the box. X11 Linux sessions need the user in the `input`
group or an X11 session pynput can hook.

WAYLAND HAS NO GLOBAL KEY HOOK FOR EITHER PYNPUT BACKEND. pynput's
default Linux backend talks to an X server; under a native Wayland
compositor (Hyprland, Sway, ...) ordinary windows are native Wayland
clients whose keystrokes never reach the X server, even when XWayland
is running for compatibility (DISPLAY is set), so the X11 backend sees
NOTHING — not a wrong key, no events at all. pynput's other backend
(uinput) fails a different way: it shells out to `dumpkeys` to load the
keymap, which needs a real virtual-console file descriptor that a
graphical session doesn't have, root or not. Both dead ends on a
Wayland desktop. So on Linux + Wayland, this module reads the physical
keyboard device directly through /dev/input via `evdev` instead — the
same permission (`input` group membership) most distros already grant
desktop users, no root required.
"""
import os
import select
import sys
import threading
import time

from pynput import keyboard

_WAYLAND = sys.platform.startswith("linux") and (
    os.environ.get("XDG_SESSION_TYPE") == "wayland"
    or bool(os.environ.get("WAYLAND_DISPLAY"))
)

if _WAYLAND:
    import evdev
    from evdev import ecodes

# Friendly config names -> pynput's names. pynput calls the right option
# key alt_r, not right_alt; the docs speak human, this map translates.
# (Field-caught: right_alt silently fell back to home, which Mac
# laptops cannot press, so the voice looked healthy and never fired.)
_ALIASES = {
    "right_alt": "alt_r", "left_alt": "alt_l",
    "right_option": "alt_r", "left_option": "alt_l",
    "right_ctrl": "ctrl_r", "left_ctrl": "ctrl_l",
    "right_cmd": "cmd_r", "left_cmd": "cmd_l",
    "right_shift": "shift_r", "left_shift": "shift_l",
}


def resolve_key(name: str):
    """'home' / 'f13' / 'right_alt' / any single character -> pynput key."""
    name = (name or "home").strip().lower()
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    name = _ALIASES.get(name, name)
    try:
        return getattr(keyboard.Key, name)
    except AttributeError:
        print(f"[ptt] unknown key {name!r} — falling back to 'home'",
              flush=True)
        return keyboard.Key.home


# The canonical (post-alias) names above, and a few extras, mapped to
# evdev's KEY_* names for the Wayland backend.
_EVDEV_NAMES = {
    "alt_r": "KEY_RIGHTALT", "alt_l": "KEY_LEFTALT",
    "ctrl_r": "KEY_RIGHTCTRL", "ctrl_l": "KEY_LEFTCTRL",
    "cmd_r": "KEY_RIGHTMETA", "cmd_l": "KEY_LEFTMETA",
    "shift_r": "KEY_RIGHTSHIFT", "shift_l": "KEY_LEFTSHIFT",
    "home": "KEY_HOME", "end": "KEY_END",
    "insert": "KEY_INSERT", "delete": "KEY_DELETE",
    "page_up": "KEY_PAGEUP", "page_down": "KEY_PAGEDOWN",
    "tab": "KEY_TAB", "caps_lock": "KEY_CAPSLOCK",
    "esc": "KEY_ESC", "space": "KEY_SPACE",
}
for _n in range(1, 25):
    _EVDEV_NAMES[f"f{_n}"] = f"KEY_F{_n}"


def resolve_evdev_code(name: str):
    """Same friendly names as resolve_key(), but -> an evdev KEY_* code."""
    name = (name or "home").strip().lower()
    if len(name) == 1 and name.isalnum():
        target = f"KEY_{name.upper()}"
    else:
        target = _EVDEV_NAMES.get(_ALIASES.get(name, name))
    if target and hasattr(ecodes, target):
        return getattr(ecodes, target)
    print(f"[ptt] unknown key {name!r} — falling back to 'home'",
          flush=True)
    return ecodes.KEY_HOME


class _PynputPTTListener:
    """macOS, Windows, and X11 Linux sessions: pynput's global hook works
    fine here, so this is the original implementation, unchanged."""

    # How long a release must stand unchallenged before it is believed.
    # Comfortably longer than any keyboard's auto-repeat period (measured
    # at ~50ms on the hardware that exposed this; Windows' fastest setting
    # is ~30ms) and short enough that letting go still feels instant.
    RELEASE_GRACE = 0.12

    def __init__(self, key="home"):
        self._key = resolve_key(key) if isinstance(key, str) else key
        self._held = False
        self._release_t = None          # a release awaiting confirmation
        self._press_evt = threading.Event()
        self._listener = keyboard.Listener(on_press=self._on_press,
                                           on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, k):
        if k != self._key:
            return
        # A press cancels any pending release: that release was auto-repeat,
        # not a human letting go.
        self._release_t = None
        if not self._held:                      # filter key-repeat
            self._held = True
            self._press_evt.set()

    def _on_release(self, k):
        if k == self._key:
            # PROVISIONAL. Believed only if no press follows; see _settle().
            self._release_t = time.monotonic()

    def _settle(self):
        """Commit a release that has stood unchallenged for the grace window."""
        r = self._release_t
        if self._held and r is not None and \
                time.monotonic() - r >= self.RELEASE_GRACE:
            self._held = False
            self._release_t = None

    def wait_press(self):
        """Block until the key goes DOWN (one event per physical press)."""
        # Settled on a loop, not once. A release landing after the last
        # is_held() poll leaves _held provisionally True, and a single
        # settle-then-wait would then block forever: the next press is
        # filtered as key-repeat, so nothing ever sets the event again.
        while True:
            self._settle()
            if self._press_evt.wait(timeout=self.RELEASE_GRACE):
                self._press_evt.clear()
                return

    def is_held(self) -> bool:
        self._settle()
        return self._held


class _EvdevPTTListener:
    """Linux + Wayland: read the physical keyboard device(s) directly.

    evdev reports DOWN (1), UP (0), and REPEAT (2) as distinct, explicit
    values — unlike pynput's X11 stream, there's no ambiguity to filter
    on press. A release grace period is still kept, defensively, in case
    a wireless receiver re-derives its own down/up pairs instead of
    relying on the kernel's repeat timer (see the module docstring).
    """

    RELEASE_GRACE = 0.12

    def __init__(self, key="home"):
        self._code = resolve_evdev_code(key) if isinstance(key, str) else key
        self._held = False
        self._release_t = None
        self._press_evt = threading.Event()
        self._stop = threading.Event()
        self._devices = self._find_keyboards()
        if not self._devices:
            print("[ptt] no readable keyboard device found under "
                  "/dev/input — add this user to the 'input' group "
                  "(then log out and back in) and try again.",
                  flush=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _find_keyboards():
        """Every /dev/input device that can send a letter key — this
        filters out the lid switch, trackpad, and audio jack, which all
        show up under /dev/input too but never report KEY_A."""
        found = []
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
                if ecodes.KEY_A in dev.capabilities().get(ecodes.EV_KEY, []):
                    found.append(dev)
                else:
                    dev.close()
            except (OSError, PermissionError):
                continue
        return found

    def _run(self):
        if not self._devices:
            return
        fd_map = {d.fd: d for d in self._devices}
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select(list(fd_map), [], [], 0.2)
            except (OSError, ValueError):
                return
            for fd in ready:
                try:
                    for ev in fd_map[fd].read():
                        self._handle(ev)
                except (OSError, BlockingIOError):
                    continue

    def _handle(self, ev):
        if ev.type != ecodes.EV_KEY or ev.code != self._code:
            return
        if ev.value == 1:                        # DOWN
            self._release_t = None
            if not self._held:                   # filter key-repeat
                self._held = True
                self._press_evt.set()
        elif ev.value == 0:                       # UP, provisional
            self._release_t = time.monotonic()
        # value == 2 (REPEAT) needs no handling: already held.

    def _settle(self):
        r = self._release_t
        if self._held and r is not None and \
                time.monotonic() - r >= self.RELEASE_GRACE:
            self._held = False
            self._release_t = None

    def wait_press(self):
        while True:
            self._settle()
            if self._press_evt.wait(timeout=self.RELEASE_GRACE):
                self._press_evt.clear()
                return

    def is_held(self) -> bool:
        self._settle()
        return self._held


class PTTListener:
    """Dispatches to the evdev backend under Linux + Wayland, and to the
    original pynput backend everywhere else (macOS, Windows, X11 Linux)."""

    def __new__(cls, key="home"):
        if _WAYLAND:
            return _EvdevPTTListener(key)
        return _PynputPTTListener(key)
