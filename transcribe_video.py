#!/usr/bin/env python3
"""
Automatically generated transcripts for a video file or a directory of video files
using faster-whisper.
"""
import argparse
from pathlib import Path
import time
import sys
import re
from typing import List
import subprocess
from faster_whisper import WhisperModel

model: WhisperModel | None = None


def extract_audio(video_path: str, audio_path: str):
    command = f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}"'
    subprocess.call(command, shell=True)


def _remove_ums(text: str) -> str:
    words = re.split(r"\b", text)
    for i, word in enumerate(words):
        if word.lower() in ["um", "uh"]:
            comma_before = i > 0 and words[i - 1] in [", ", ","]
            comma_after = i < len(words) - 1 and words[i + 1] in [", ", ","]
            words[i] = " " if comma_before and comma_after else ""
            if comma_before:
                words[i - 1] = ""
            if comma_after:
                words[i + 1] = ""
                if i < len(words) - 2 and word in ["Um", "Uh"]:
                    words[i + 2] = words[i + 2].capitalize()
    return "".join(words)


def transcribe_audio(
    audio_path: Path,
    output_path: Path,
    language: str = "en",
    device: str = "cuda",
    initial_prompt: str | None = None,
    remove_ums: bool = True,
):
    global model
    if model is None:
        print("Loading model...")
        model = WhisperModel("medium", device=device)

    default_prompts = {
        "de": "Herzlich willkommen zur Vorlesung! Legen wir los.",
        "en": "Welcome back to the lecture. Let's get started.",
    }

    print("Opening audio file...")
    segments, _ = model.transcribe(
        str(audio_path),
        language=language,
        initial_prompt=initial_prompt or default_prompts.get(language, ""),
    )

    print("Transcribing...")
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            text = segment.text
            if remove_ums:
                text = _remove_ums(text)
            line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {text}"
            print(line)
            f.write(f"{line}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe one or multiple video files."
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the video file or a directory containing video files.",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Path to save the extracted audio file. Only allowed for single video files.",
    )
    parser.add_argument(
        "--transcription_path",
        type=str,
        default=None,
        help="Path to save the transcription file. Only allowed for single video files.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language of the audio files. Defaults to 'en'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run the model on. Defaults to 'cuda'.",
    )
    parser.add_argument(
        "--initial_prompt",
        type=str,
        default=None,
        help="Initial prompt for the model. Default is chosen based on language.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate even existing audio or transcription files. The default behavior is to skip generation if the files already exist.",
    )
    parser.add_argument(
        "--keep-ums",
        action="store_true",
        help="Do not automatically remove 'um' and 'uh' from the transcript.",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path)
    video_files: List[Path] = []
    audio_files: List[Path] = []
    transcription_files: List[Path] = []
    if video_path.is_dir():
        video_files = list(video_path.glob("*.mp4"))
        audio_files = [video_file.with_suffix(".mp3") for video_file in video_files]
        transcription_files = [
            video_file.with_suffix(".txt") for video_file in video_files
        ]
    elif video_path.is_file():
        video_files = [video_path]
        audio_files = [args.audio_path or video_path.with_suffix(".mp3")]
        transcription_files = [
            args.transcription_path or video_path.with_suffix(".txt")
        ]
    else:
        print(f"Invalid video path: {args.video_path}")
        sys.exit(1)

    for video_file, audio_file, transcription_file in zip(
        video_files, audio_files, transcription_files
    ):
        skip_audio: bool = False
        if audio_file.exists():
            if args.regenerate:
                print(f"Audio file already exists at {audio_file}. Regenerating...")
            else:
                print(f"Audio file already exists at {audio_file}. Skipping...")
                skip_audio = True
        if not skip_audio:
            extract_audio(video_file, audio_file)

        skip_transcription: bool = False
        if transcription_file.exists():
            if args.regenerate:
                print(
                    f"Transcription file already exists at {transcription_file}. Regenerating..."
                )
            else:
                print(
                    f"Transcription file already exists at {transcription_file}. Skipping..."
                )
                skip_transcription = True
        if not skip_transcription:
            start = time.time()
            transcribe_audio(
                audio_file,
                transcription_file,
                language=args.language,
                device=args.device,
                initial_prompt=args.initial_prompt,
                remove_ums=not args.keep_ums,
            )
            elapsed = time.time() - start
            print(f"Transcription took {elapsed} seconds.")

    print("Done!")


if __name__ == "__main__":
    main()
