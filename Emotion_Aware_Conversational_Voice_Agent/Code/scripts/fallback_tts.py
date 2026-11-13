"""
fallback_tts.py - the agent's emergency voice (GPU demo build, step 8b).

Used when CSM-1B (gpu/tts_gpu.py) is not loaded or fails: the operating system's built-in voice through pyttsx3.
No torch and nothing from gpu/, so it works on any machine. If pyttsx3 itself fails (not installed, no voice,
no audio device), the text is printed instead. speak() never raises.

The caller strips voice tags first (gpu/tts_gpu.speak_with_fallback does); this file reads the text as given.

Usage:  python scripts/fallback_tts.py "Hello, this is the fallback voice."
"""

import sys


def _print_text(text, error) -> None:
    """Last resort: show the reply on screen. Even a console that cannot print some characters must not raise."""
    line = f"[voice unavailable: {type(error).__name__}] {text}"
    try:
        print(line, flush=True)
    except Exception:  # noqa: BLE001  (e.g. a console encoding that cannot show an emoji)
        try:
            sys.stdout.write(line.encode("ascii", "replace").decode("ascii") + "\n")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass


def speak(text) -> str:
    """Speak text with pyttsx3. Returns the path used: "pyttsx3", "text" (printed instead) or "none" (empty text).
    Never raises."""
    try:
        text = " ".join(str(text if text is not None else "").split())
    except Exception:  # noqa: BLE001  (an object whose str() fails)
        text = ""
    if not text:
        return "none"
    try:
        import pyttsx3   # imported here so a missing pyttsx3 means "print the text", not an import error

        engine = pyttsx3.init()
        try:
            engine.say(text)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:  # noqa: BLE001
                pass
        return "pyttsx3"
    except Exception as e:  # noqa: BLE001
        _print_text(text, e)
        return "text"


if __name__ == "__main__":
    print(f"path: {speak(' '.join(sys.argv[1:]) or 'This is the fallback voice.')}")
