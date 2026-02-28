"""CLI entry point for the agadmator pipeline."""

import argparse
import logging
import sys


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Agadmator-style chess video generation pipeline"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    # Pipeline status
    sub.add_parser("status", help="Show data pipeline status")

    # Step 0: Data collection
    p_meta = sub.add_parser("fetch-metadata", help="Fetch agadmator library metadata")
    p_meta.add_argument("--output", default=None)

    p_audio = sub.add_parser("download-audio", help="Download audio from YouTube")
    p_audio.add_argument("--metadata", default=None)
    p_audio.add_argument("--limit", type=int, default=None)
    p_audio.add_argument("--workers", type=int, default=8, help="Concurrent downloads")

    p_vocals = sub.add_parser("isolate-vocals", help="Isolate vocals with Demucs")
    p_vocals.add_argument("--input-dir", default=None)
    p_vocals.add_argument("--output-dir", default=None)

    p_transcribe = sub.add_parser("transcribe", help="Transcribe with faster-whisper")
    p_transcribe.add_argument("--input-dir", default=None)
    p_transcribe.add_argument("--model", default="large-v3")

    p_pgn = sub.add_parser("collect-pgn", help="Extract/collect PGN files")
    p_pgn.add_argument("--metadata", default=None)

    p_align = sub.add_parser("align", help="Align PGN moves with transcripts")
    p_align.add_argument("--transcripts-dir", default=None)
    p_align.add_argument("--pgn-dir", default=None)

    # Step 1: LLM
    sub.add_parser("prepare-llm-data", help="Prepare LLM training data")

    p_llm_train = sub.add_parser("train-llm", help="Fine-tune LLM with QLoRA")
    p_llm_train.add_argument("--config", default="configs/default.yaml")
    p_llm_train.add_argument("--phase", choices=["pretrain", "style"], required=True)

    p_generate = sub.add_parser("generate-commentary", help="Generate commentary from PGN")
    p_generate.add_argument("pgn_file")
    p_generate.add_argument("--model-path", required=True)
    p_generate.add_argument("--output", default=None)

    # Step 2: TTS
    sub.add_parser("prepare-tts-data", help="Prepare TTS training data")

    p_tts_train = sub.add_parser("train-tts", help="Fine-tune TTS model")
    p_tts_train.add_argument("--config", default="configs/default.yaml")

    p_speak = sub.add_parser("synthesize", help="Generate speech from transcript")
    p_speak.add_argument("transcript_file")
    p_speak.add_argument("--reference-audio", required=True)
    p_speak.add_argument("--output", default="output.wav")

    # Step 3 & 4: Render + compose
    p_render = sub.add_parser("render-board", help="Render board animation")
    p_render.add_argument("pgn_file")
    p_render.add_argument("--timestamps", required=True)

    p_compose = sub.add_parser("compose-video", help="Compose final video")
    p_compose.add_argument("--board-video", required=True)
    p_compose.add_argument("--audio", required=True)
    p_compose.add_argument("--output", default="output.mp4")

    # Full pipeline
    p_full = sub.add_parser("generate-video", help="Full pipeline: PGN → video")
    p_full.add_argument("pgn_file")
    p_full.add_argument("--llm-model", required=True)
    p_full.add_argument("--reference-audio", required=True)
    p_full.add_argument("--output", default="output.mp4")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    _dispatch(args)


def _dispatch(args):
    if args.command == "status":
        _show_status()
    elif args.command == "fetch-metadata":
        from agadmator.data.metadata import fetch_metadata
        fetch_metadata(args.output)
    elif args.command == "download-audio":
        from agadmator.data.audio import download_audio
        from agadmator.config import RAW_DIR
        meta = args.metadata or str(RAW_DIR / "metadata.json")
        download_audio(meta, args.limit, args.workers)
    elif args.command == "isolate-vocals":
        from agadmator.data.vocals import isolate_vocals
        isolate_vocals(args.input_dir, args.output_dir)
    elif args.command == "transcribe":
        from agadmator.data.transcribe import transcribe_all
        transcribe_all(args.input_dir, args.model)
    elif args.command == "collect-pgn":
        from agadmator.data.pgn_collect import collect_pgn
        from agadmator.config import RAW_DIR
        meta = args.metadata or str(RAW_DIR / "metadata.json")
        collect_pgn(meta)
    elif args.command == "align":
        from agadmator.data.align import align_all
        align_all(args.transcripts_dir, args.pgn_dir)
    elif args.command == "prepare-llm-data":
        from agadmator.llm.prepare_data import prepare_llm_data
        prepare_llm_data()
    elif args.command == "train-llm":
        from agadmator.llm.train import train_llm
        train_llm(args.config, args.phase)
    elif args.command == "generate-commentary":
        from agadmator.llm.generate import generate_commentary
        generate_commentary(args.pgn_file, args.model_path, args.output)
    elif args.command == "prepare-tts-data":
        from agadmator.tts.prepare_data import prepare_tts_data
        prepare_tts_data()
    elif args.command == "train-tts":
        from agadmator.tts.train import train_tts
        train_tts(args.config)
    elif args.command == "synthesize":
        from agadmator.tts.synthesize import synthesize
        synthesize(args.transcript_file, args.reference_audio, args.output)
    elif args.command == "render-board":
        from agadmator.render.board import render_board_video
        render_board_video(args.pgn_file, args.timestamps)
    elif args.command == "compose-video":
        from agadmator.compose.video import compose_video
        compose_video(args.board_video, args.audio, args.output)
    elif args.command == "generate-video":
        from agadmator.compose.pipeline import full_pipeline
        full_pipeline(
            args.pgn_file, args.llm_model, args.reference_audio, args.output
        )


def _show_status():
    """Show the current state of each pipeline stage."""
    from agadmator.config import (
        LIBRARY_DB_DIR, RAW_DIR, AUDIO_DIR, VOCALS_DIR,
        TRANSCRIPTS_DIR, PGN_DIR, ALIGNED_DIR, TTS_SEGMENTS_DIR,
        PROCESSED_DIR,
    )
    import json

    def count(d, pattern="*"):
        if d.exists():
            return len(list(d.glob(pattern)))
        return 0

    meta_path = RAW_DIR / "metadata.json"
    meta_count = 0
    with_pgn = 0
    with_analysis = 0
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta_count = len(meta)
        with_pgn = sum(1 for v in meta if any(g.get("pgn") for g in v.get("games", [])))
        with_analysis = sum(1 for v in meta if v.get("has_lichess_analysis"))

    db_files = count(LIBRARY_DB_DIR / "_repo" / "db", "*.json")
    llm_data = PROCESSED_DIR / "llm_training"

    print("=== Agadmator Pipeline Status ===\n")
    print(f"  Library DB files:     {db_files:>5}")
    print(f"  Metadata index:       {meta_count:>5}  (with PGN: {with_pgn}, with evals: {with_analysis})")
    print(f"  Audio files:          {count(AUDIO_DIR, '*.wav'):>5}")
    print(f"  Vocal files:          {count(VOCALS_DIR, '*.wav'):>5}")
    print(f"  Transcripts:          {count(TRANSCRIPTS_DIR, '*.json'):>5}")
    print(f"  PGN files:            {count(PGN_DIR, '*.pgn'):>5}")
    print(f"  Aligned pairs:        {count(ALIGNED_DIR, '*.json'):>5}")
    print(f"  TTS segments:         {count(TTS_SEGMENTS_DIR, '*.wav'):>5}")
    print(f"  LLM training files:   {count(llm_data, '*.jsonl'):>5}")
    print()

    # Suggest next step
    if db_files == 0 and meta_count == 0:
        print("  Next: agadmator fetch-metadata")
    elif meta_count > 0 and count(PGN_DIR, "*.pgn") == 0:
        print("  Next: agadmator collect-pgn")
    elif meta_count > 0 and count(AUDIO_DIR, "*.wav") == 0:
        print("  Next: agadmator download-audio --limit 10  (start small)")
    elif count(AUDIO_DIR, "*.wav") > 0 and count(VOCALS_DIR, "*.wav") == 0:
        print("  Next: agadmator isolate-vocals")
    elif count(VOCALS_DIR, "*.wav") > 0 and count(TRANSCRIPTS_DIR, "*.json") == 0:
        print("  Next: agadmator transcribe")
    elif count(TRANSCRIPTS_DIR, "*.json") > 0 and count(ALIGNED_DIR, "*.json") == 0:
        print("  Next: agadmator align")
    elif count(ALIGNED_DIR, "*.json") > 0 and count(llm_data, "*.jsonl") == 0:
        print("  Next: agadmator prepare-llm-data")
    else:
        print("  All data stages have outputs. Ready for training.")


if __name__ == "__main__":
    main()
