"""Translate Thadou-Kuki <-> English with the local Ollama model.

Ollama's built-in Qwen3 renderer overrides the Modelfile TEMPLATE and makes the model
emit stray <think>/<tool_call> tags, so this talks to /api/generate in raw mode and
builds the exact prompt format used in training.

  python translate.py "God loves you."                 # English -> Thadou (default)
  python translate.py -r "Kapa le kanu chu inn ah aum uve."   # Thadou -> English
  python translate.py -m thadou "..."                  # pick model (default thadou-v2)
  python translate.py                                  # interactive
"""
import argparse, json, re, sys, time, urllib.request

SYS = "You are an expert translator between Thadou-Kuki (Thado Chin) and English."
JUNK = re.compile(r"</?(think|tool_call)>", re.I)


def translate(text, to_english=False, model="thadou-v2", host="http://localhost:11434"):
    head = "Thadou-Kuki to English:" if to_english else "English to Thadou-Kuki:"
    prompt = (f"<|im_start|>system\n{SYS}<|im_end|>\n"
              f"<|im_start|>user\n{head}\n\n{text.strip()}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    body = json.dumps({"model": model, "prompt": prompt, "raw": True, "stream": False,
                       "options": {"temperature": 0.2, "repeat_penalty": 1.1,
                                   "num_predict": 256, "stop": ["<|im_end|>", "<|im_start|>"]}}).encode()
    req = urllib.request.Request(f"{host}/api/generate", body, {"Content-Type": "application/json"})
    for attempt in range(3):  # Ollama 500s while swapping a model in/out of VRAM
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return JUNK.sub("", json.load(r).get("response", "")).strip()
        except urllib.error.HTTPError:
            if attempt == 2:
                raise
            time.sleep(5)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="*")
    p.add_argument("-r", "--to-english", action="store_true", help="Thadou -> English")
    p.add_argument("-m", "--model", default="thadou-v2")
    a = p.parse_args()
    if a.text:
        print(translate(" ".join(a.text), a.to_english, a.model))
    else:
        d = "Thadou -> English" if a.to_english else "English -> Thadou"
        print(f"{a.model} | {d} | blank line to quit")
        while (line := input("> ").strip()):
            print(translate(line, a.to_english, a.model))
