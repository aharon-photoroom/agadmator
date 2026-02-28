"""Step 0.1: Fetch agadmator library metadata.

The agadmator-library repo stores one JSON file per video in db/.
Each file is named {youtubeVideoId}.json and contains:
  - videoSnippet: YouTube metadata (title, description, publishedAt)
  - videoContentDetails: duration
  - videoGame[]: games covered (playerWhite, playerBlack, pgn, fen, date)
  - chessCom / chess365 / chesstempoCom / lichessMasters: cross-references
  - lichessGameId[]: Lichess game IDs
  - lichessGameEval[]: full per-move engine analysis
  - stockfishEval[]: final position evaluation
"""

import json
import logging
import re
import subprocess
from pathlib import Path

import requests
from tqdm import tqdm

from agadmator.config import (
    LIBRARY_DB_DIR,
    LIBRARY_RAW_BASE,
    AGADMATOR_LIBRARY_REPO,
    RAW_DIR,
)

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com/repos/agadmator-library/agadmator-library.github.io"


def _clone_db_sparse(dest: Path):
    """Clone only the db/ directory using git sparse-checkout."""
    dest.mkdir(parents=True, exist_ok=True)
    repo_dir = dest / "_repo"

    if (repo_dir / "db").exists():
        log.info("Library repo already cloned at %s", repo_dir / "db")
        return repo_dir / "db"

    log.info("Sparse-cloning agadmator-library (db/ only)...")
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--sparse",
         AGADMATOR_LIBRARY_REPO, str(repo_dir)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", "db"],
        cwd=str(repo_dir), check=True, capture_output=True, text=True,
    )
    log.info("Cloned db/ directory to %s", repo_dir / "db")
    return repo_dir / "db"


def _list_db_files_via_api() -> list[str]:
    """List all db/*.json filenames via GitHub API (tree endpoint)."""
    resp = requests.get(f"{GITHUB_API}/git/trees/master?recursive=1", timeout=60)
    resp.raise_for_status()
    tree = resp.json()["tree"]
    return [
        item["path"]
        for item in tree
        if item["path"].startswith("db/") and item["path"].endswith(".json")
    ]


def _fetch_single_json(path: str) -> dict | None:
    """Fetch a single db/*.json file via raw GitHub URL."""
    url = f"{LIBRARY_RAW_BASE}/{path}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        log.debug("Failed to fetch %s: %s", path, e)
    return None


def _parse_duration_iso(duration: str) -> int:
    """Parse ISO 8601 duration (PT17M58S) to seconds."""
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration or "")
    if not match:
        return 0
    h, m, s = (int(x or 0) for x in match.groups())
    return h * 3600 + m * 60 + s


def normalize_video(raw: dict) -> dict:
    """Normalize a raw db/*.json entry to a consistent schema."""
    video_id = raw.get("_id", "")
    snippet = raw.get("videoSnippet", {})
    details = raw.get("videoContentDetails", {})
    games_raw = raw.get("videoGame", [])
    chess365 = raw.get("chess365", {})
    chesstempo = raw.get("chesstempoCom", {})
    lichess_evals = raw.get("lichessGameEval", [])
    lichess_ids = raw.get("lichessGameId", [])

    # Extract game info (most videos have 1 game, some have multiple)
    games = []
    for g in games_raw:
        games.append({
            "white": g.get("playerWhite", ""),
            "black": g.get("playerBlack", ""),
            "pgn": g.get("pgn", ""),
            "fen": g.get("fen", ""),
            "date": g.get("date", ""),
        })

    # Best event/tournament name (prefer chesstempo > chess365)
    event = (
        chesstempo.get("event")
        or chess365.get("tournament")
        or ""
    )

    # Best opening info
    opening_name = chesstempo.get("openingName", "")
    eco = chesstempo.get("eco") or chess365.get("eco") or ""
    if not opening_name and lichess_evals:
        opening = lichess_evals[0].get("opening", {})
        opening_name = opening.get("name", "")
        eco = eco or opening.get("eco", "")

    # Result
    result = (
        chess365.get("result")
        or chesstempo.get("result", "")
        or ""
    )
    # Normalize chesstempo format (w/b/d → 1-0/0-1/1/2-1/2)
    if result == "w":
        result = "1-0"
    elif result == "b":
        result = "0-1"
    elif result == "d":
        result = "1/2-1/2"

    # Lichess game IDs for fetching annotated PGN
    lic_ids = [entry.get("id", "") for entry in lichess_ids if entry.get("id")]

    # Per-move analysis from Lichess (if available)
    has_analysis = bool(lichess_evals and lichess_evals[0].get("analysis"))

    return {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "published_at": snippet.get("publishedAt", ""),
        "duration_seconds": _parse_duration_iso(details.get("duration", "")),
        "games": games,
        "event": event,
        "opening": opening_name,
        "eco": eco,
        "result": result,
        "lichess_ids": lic_ids,
        "has_lichess_analysis": has_analysis,
        # Keep raw cross-references for enrichment
        "_lichess_evals": lichess_evals if has_analysis else [],
    }


def fetch_metadata(output_path: str | None = None):
    """Fetch and save all agadmator game metadata.

    Strategy:
      1. Try sparse git clone (fastest, gets all 4855 files at once)
      2. Fallback: list files via GitHub API + fetch individually
    """
    output = Path(output_path) if output_path else RAW_DIR / "metadata.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Try sparse clone first
    db_dir = None
    try:
        db_dir = _clone_db_sparse(LIBRARY_DB_DIR)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log.warning("Sparse clone failed (%s), falling back to API", e)

    videos = []

    if db_dir and db_dir.exists():
        # Parse local JSON files
        json_files = sorted(db_dir.glob("*.json"))
        log.info("Parsing %d local db files...", len(json_files))
        for f in tqdm(json_files, desc="Parsing metadata"):
            try:
                with open(f) as fh:
                    raw = json.load(fh)
                videos.append(normalize_video(raw))
            except (json.JSONDecodeError, Exception) as e:
                log.debug("Failed to parse %s: %s", f.name, e)
    else:
        # Fetch via API
        log.info("Listing db files via GitHub API...")
        db_paths = _list_db_files_via_api()
        log.info("Found %d db files, fetching...", len(db_paths))
        for path in tqdm(db_paths, desc="Fetching metadata"):
            raw = _fetch_single_json(path)
            if raw:
                videos.append(normalize_video(raw))

    # Filter to entries with at least one game with PGN
    has_pgn = [v for v in videos if any(g["pgn"] for g in v["games"])]
    log.info(
        "Total videos: %d, with PGN: %d, with Lichess analysis: %d",
        len(videos),
        len(has_pgn),
        sum(1 for v in videos if v["has_lichess_analysis"]),
    )

    # Save full metadata (without bulky _lichess_evals in the index)
    index = []
    for v in videos:
        entry = {k: v for k, v in v.items() if not k.startswith("_")}
        index.append(entry)

    with open(output, "w") as f:
        json.dump(index, f, indent=2)
    log.info("Saved metadata index (%d videos) to %s", len(index), output)

    # Save lichess evals separately (they're large)
    evals_output = output.parent / "lichess_evals.json"
    evals = {
        v["video_id"]: v["_lichess_evals"]
        for v in videos if v["_lichess_evals"]
    }
    with open(evals_output, "w") as f:
        json.dump(evals, f)
    log.info("Saved Lichess evals (%d videos) to %s", len(evals), evals_output)

    return videos
