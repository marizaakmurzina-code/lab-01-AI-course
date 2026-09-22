"""Part 2 -- the measurement, on Gemini. Needs GEMINI_API_KEY in the environment.

1. Token counts for every text in :mod:`texts`, in every language
   (count_tokens: free, the model never runs).
2. With --call: one real request per language (system prompt + complaint), so the
   answer length is measured too. The input tokens of that request are then the
   exact number Google bills, and Part 3 prices exactly that.

Results are written to measurements.json for Part 3.

Run:
    python3 part2_measure.py           # count tokens only, no answers
    python3 part2_measure.py --call    # also answer the complaint in en, ru, kk
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from google import genai
from google.genai import errors, types

from prices import DEFAULT_MODEL, MODELS
from texts import CORPUS, LANGUAGES

OUTPUT_PATH = Path(__file__).with_name("measurements.json")

#: Gemini "thinking" tokens count against this limit and are billed as output,
#: so it has to clear thinking plus a full answer. finish_reason MAX_TOKENS means
#: an answer was cut off and that measurement is unusable.
MAX_TOKENS = 4096


def retry(call):
    """Run call(); on rate-limit / server errors wait 30 s and try again (3 attempts)."""
    for attempt in range(3):
        try:
            return call()
        except errors.APIError as exc:
            if getattr(exc, "code", None) not in (429, 500, 503) or attempt == 2:
                raise
            print(f"  API error {exc.code}, waiting 30 s ...")
            time.sleep(30)


def count_tokens(client, model_id: str, text: str) -> int:
    """Return the number of input tokens ``text`` costs on ``model_id``."""
    return retry(lambda: client.models.count_tokens(model=model_id, contents=text)).total_tokens


def count_request_tokens(client, model_id: str, lang: str) -> int:
    """ESTIMATE of one request's input tokens: system prompt + complaint, counted together.

    With --call this is replaced by the exact billed number from the real request.
    """
    return count_tokens(
        client, model_id, CORPUS["system_prompt"][lang] + "\n\n" + CORPUS["complaint"][lang]
    )


def one_real_request(client, model_id: str, lang: str) -> dict:
    """Send one request, print the answer and its billed usage, return the numbers."""
    r = retry(lambda: client.models.generate_content(
        model=model_id,
        contents=CORPUS["complaint"][lang],
        config=types.GenerateContentConfig(
            system_instruction=CORPUS["system_prompt"][lang], max_output_tokens=MAX_TOKENS
        ),
    ))
    u = r.usage_metadata
    answer = u.candidates_token_count or 0
    thinking = getattr(u, "thoughts_token_count", 0) or 0  # billed as output, never shown
    finish = r.candidates[0].finish_reason if r.candidates else "NONE"
    finish = getattr(finish, "name", finish)
    print(f"  finish_reason: {finish}")
    try:
        print("  --- answer ---\n  " + (r.text or "(no text)").replace("\n", "\n  "))
    except Exception:
        print("  (no text)")
    print(f"  billed: {u.prompt_token_count} in, {answer + thinking} out "
          f"({answer} answer + {thinking} thinking)")
    if "MAX_TOKENS" in str(finish).upper():
        print(f"  NOTE: answer was cut off at max_output_tokens={MAX_TOKENS}.")
    return {"input_tokens": u.prompt_token_count, "output_tokens": answer + thinking,
            "answer_tokens": answer, "thinking_tokens": thinking, "finish_reason": str(finish)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS),
                        help=f"which model to measure on (default: {DEFAULT_MODEL})")
    parser.add_argument("--call", action="store_true",
                        help="also answer the complaint in each language")
    args = parser.parse_args()
    model_id = MODELS[args.model].model_id

    try:
        client = genai.Client()  # reads GEMINI_API_KEY
    except Exception as exc:
        print(f"could not build a client: {exc}\nset GEMINI_API_KEY first.", file=sys.stderr)
        return 1

    counts, request_tokens, billed = {}, {}, {}
    try:
        print(f"counting tokens on {model_id} (free, no model run)")
        for item_id, versions in CORPUS.items():
            counts[item_id] = {lang: count_tokens(client, model_id, versions[lang]) for lang in LANGUAGES}
            print(f"  {item_id:<14}", " ".join(f"{l}={counts[item_id][l]}" for l in LANGUAGES))
        request_tokens = {lang: count_request_tokens(client, model_id, lang) for lang in LANGUAGES}
        print(f"  {'request':<14}", " ".join(f"{l}={request_tokens[l]}" for l in LANGUAGES),
              "(system + complaint, ESTIMATE)")

        if args.call:
            print(f"\nanswering the same complaint on {model_id}, in each language:")
            for lang in LANGUAGES:
                print(f"\n[{lang}]")
                billed[lang] = one_real_request(client, model_id, lang)
                request_tokens[lang] = billed[lang]["input_tokens"]  # exact, replaces the estimate
            print(f"\n  {'request':<14}", " ".join(f"{l}={request_tokens[l]}" for l in LANGUAGES),
                  "(billed by Google, exact)")
    except errors.APIError as exc:
        print(f"API error {getattr(exc, 'code', '?')}: {exc}\n"
              "(400/403: wrong key or unsupported region)", file=sys.stderr)
        return 1

    payload = {"model": args.model, "model_id": model_id, "token_counts": counts,
               "request_tokens": request_tokens, "one_request_billed": billed or None}
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUTPUT_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
