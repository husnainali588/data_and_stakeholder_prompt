"""
Fetches call transcripts from Retell AI API (DEV key first, PROD fallback),
then scores each transcript for Engagement (0-10) and Data Collection (0-9).

Input  : call_ids.csv        — CSV with a single column: call_id
Output : scored_transcripts.csv — call_id, transcript, engagement_score, data_collection_score, scoring_error
"""

import csv
import os
import json
import time
import requests
import argparse
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE      = "call_ids.csv"
OUTPUT_FILE     = "scored_transcripts.csv"
CALL_ID_COL     = "call_id"

RETELL_BASE_URL = "https://api.retellai.com/v2/get-call"
RETELL_DELAY    = 0.3    # seconds between Retell API calls

OPENROUTER_URL  = "https://openrouter.ai/api/v1/chat/completions"
MODEL           = "openai/gpt-5-mini"
MAX_WORKERS     = 5      # parallel threads for scoring


# ── Load API keys ─────────────────────────────────────────────────────────────
def load_keys():
    dev_key  = os.getenv("RETELL_DEV_API_KEY",  "").strip()
    prod_key = os.getenv("RETELL_PROD_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if not dev_key:
        raise ValueError("RETELL_DEV_API_KEY not set in .env file.")
    if not prod_key:
        raise ValueError("RETELL_PROD_API_KEY not set in .env file.")
    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY not set in .env file.")

    return dev_key, prod_key, openrouter_key


# ── Load rubric files ─────────────────────────────────────────────────────────
def load_rubric(filename: str) -> dict:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"{filename} not found. Make sure it's in the same directory.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Retell: fetch transcript ──────────────────────────────────────────────────
def fetch_from_retell(call_id: str, api_key: str) -> tuple:
    """Returns (transcript_text, status_code)."""
    try:
        resp = requests.get(
            f"{RETELL_BASE_URL}/{call_id}",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=15
        )
        return resp.json().get("transcript", ""), resp.status_code
    except requests.exceptions.RequestException as e:
        return "", None


def get_transcript(call_id: str, dev_key: str, prod_key: str) -> tuple:
    """Try DEV key first, fall back to PROD on 404. Returns (transcript, found)."""
    transcript, status = fetch_from_retell(call_id, dev_key)
    if status == 200:
        return transcript, True

    if status == 404:
        transcript, status = fetch_from_retell(call_id, prod_key)
        if status == 200:
            return transcript, True

    return "", False


# ── OpenRouter: score transcript ──────────────────────────────────────────────
def call_api(rubric: dict, transcript: str, score_range: int, openrouter_key: str) -> dict:
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": json.dumps(rubric)},
                    {"role": "user",   "content": f"Score this call transcript:\n\n{transcript}"}
                ],
                "temperature": 0,
            },
            timeout=60
        )
    except requests.exceptions.Timeout:
        return {"score": None, "success": False, "error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        return {"score": None, "success": False, "error": f"Request failed: {str(e)}"}

    if response.status_code != 200:
        try:
            error_detail = response.json().get("error", {})
            error_msg = error_detail.get("message", response.text[:300]) if isinstance(error_detail, dict) else str(error_detail)
        except Exception:
            error_msg = response.text[:300]
        return {"score": None, "success": False, "error": f"API {response.status_code}: {error_msg}"}

    score_text = response.json()["choices"][0]["message"]["content"].strip()

    try:
        score = int(score_text)
        if 0 <= score <= score_range:
            return {"score": score, "success": True}
        else:
            return {"score": None, "success": False, "error": f"Score out of range: {score_text}"}
    except ValueError:
        return {"score": None, "success": False, "error": f"Invalid response: {score_text}"}


def score_transcript(transcript: str, engagement_rubric: dict, data_rubric: dict, openrouter_key: str) -> dict:
    """Score one transcript with both rubrics in parallel."""
    if not transcript or not transcript.strip():
        return {
            "engagement_score":      None,
            "data_collection_score": None,
            "success": False,
            "error":   "Empty transcript"
        }

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_eng  = executor.submit(call_api, engagement_rubric, transcript, 10, openrouter_key)
        future_data = executor.submit(call_api, data_rubric,       transcript,  9, openrouter_key)
        eng  = future_eng.result()
        data = future_data.result()

    return {
        "engagement_score":      eng["score"],
        "data_collection_score": data["score"],
        "success":               eng["success"] and data["success"],
        "error":                 eng.get("error") or data.get("error")
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("\n" + "=" * 60)
    print("  RETELL TRANSCRIPT FETCHER + SCORER")
    print("=" * 60)

    # Load keys
    dev_key, prod_key, openrouter_key = load_keys()
    print("✓ API keys loaded (Retell DEV + PROD, OpenRouter)")

    # Load rubrics
    engagement_rubric = load_rubric("scoring_rubric.json")
    data_rubric       = load_rubric("data_collection_rubric.json")
    print("✓ Rubric files loaded")

    # Load call IDs
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f"'{INPUT_FILE}' not found. Make sure it's in the same directory.")

    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader   = csv.DictReader(f)
        call_ids = [row[CALL_ID_COL].strip() for row in reader if row.get(CALL_ID_COL, "").strip()]

    total = len(call_ids)
    print(f"✓ Loaded {total} call ID(s) from '{INPUT_FILE}'\n")

    # ── Step 1: Fetch all transcripts sequentially ────────────────────────────
    print(f"STEP 1 — Fetching transcripts from Retell...\n")

    transcript_map = {}   # call_id → transcript text

    for i, call_id in enumerate(call_ids, 1):
        print(f"  [{i}/{total}] {call_id} ...", end=" ", flush=True)
        transcript, found = get_transcript(call_id, dev_key, prod_key)
        transcript_map[call_id] = transcript
        print("✓" if found else "✗ not found")
        time.sleep(RETELL_DELAY)

    # ── Step 2: Score all transcripts in parallel ─────────────────────────────
    print(f"\nSTEP 2 — Scoring transcripts (parallel, {MAX_WORKERS} workers)...\n")

    score_map = {}   # call_id → score result

    def score_one(call_id):
        transcript = transcript_map.get(call_id, "")
        return call_id, score_transcript(transcript, engagement_rubric, data_rubric, openrouter_key)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(score_one, cid): cid for cid in call_ids}

        done = 0
        for future in as_completed(futures):
            call_id, result = future.result()
            score_map[call_id] = result
            done += 1
            status = (
                f"Engagement: {result['engagement_score']}/10  |  Data: {result['data_collection_score']}/9"
                if result["success"]
                else f"FAILED — {result['error']}"
            )
            print(f"  [{done}/{total}] {call_id} → {status}")

    # ── Step 3: Write output CSV in original order ────────────────────────────
    success = 0
    failed  = []

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["call_id", "transcript", "engagement_score", "data_collection_score", "scoring_error"])
        writer.writeheader()

        for call_id in call_ids:
            result     = score_map.get(call_id, {})
            transcript = transcript_map.get(call_id, "")

            writer.writerow({
                "call_id":              call_id,
                "transcript":           transcript,
                "engagement_score":     result.get("engagement_score")      if result.get("engagement_score")      is not None else "ERROR",
                "data_collection_score": result.get("data_collection_score") if result.get("data_collection_score") is not None else "ERROR",
                "scoring_error":        result.get("error") or ""
            })

            if result.get("success"):
                success += 1
            else:
                failed.append(call_id)

    print(f"\n{'=' * 60}")
    print(f"  DONE — {success}/{total} scored successfully")
    if failed:
        print(f"  Failed : {failed}")
    print(f"  Saved to: {OUTPUT_FILE}")
    print(f"{'=' * 60}\n")


# ── Manual mode ───────────────────────────────────────────────────────────────
def manual_mode():
    print("\n" + "=" * 60)
    print("  TRANSCRIPT SCORER — MANUAL MODE")
    print("=" * 60)
    print("Paste your transcript below.")
    print("When done, type END on a new line and press Enter:")
    print("-" * 60)

    _, _, openrouter_key  = load_keys()
    engagement_rubric     = load_rubric("scoring_rubric.json")
    data_rubric           = load_rubric("data_collection_rubric.json")

    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)

    transcript = "\n".join(lines).strip()
    if not transcript:
        print("\nNo transcript entered. Exiting.")
        return

    print("\nScoring...")
    result = score_transcript(transcript, engagement_rubric, data_rubric, openrouter_key)

    print("\n" + "=" * 60)
    if result["success"]:
        print(f"  Engagement Score      : {result['engagement_score']} / 10")
        print(f"  Data Collection Score : {result['data_collection_score']} / 9")
    else:
        print(f"  ERROR: {result['error']}")
        if result["engagement_score"] is not None:
            print(f"  Engagement Score      : {result['engagement_score']} / 10")
        if result["data_collection_score"] is not None:
            print(f"  Data Collection Score : {result['data_collection_score']} / 9")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Retell transcripts and score them.")
    parser.add_argument("--manual", action="store_true", help="Score a single transcript manually")
    args = parser.parse_args()

    try:
        if args.manual:
            manual_mode()
        else:
            run()
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error: {e}\n")
    except KeyboardInterrupt:
        print("\n\nStopped by user.\n")
