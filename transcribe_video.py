import argparse
import subprocess
from faster_whisper import WhisperModel
from pathlib import Path
import time


def extract_audio(video_path: str, audio_path: str):
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path}"
    subprocess.call(command, shell=True)


def transcribe_audio(
    audio_path: str,
    output_path: str,
    language: str = "de",
    device: str = "cuda",
    initial_prompt: str = "Herzlich willkommen zur Vorlesung! Legen wir los.",
):
    print("Loading model...")
    model = WhisperModel("medium", device=device)

    print("Opening audio file...")
    segments, info = model.transcribe(
        audio_path,
        language=language,
        initial_prompt=initial_prompt,
    )

    print("Transcribing...")
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
            print(line)
            f.write(f"{line}\n")


def main():
    parser = argparse.ArgumentParser(description="Transcribe a video file.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Path to save the extracted audio file.",
    )
    parser.add_argument(
        "--transcription_path",
        type=str,
        default=None,
        help="Path to save the transcription file.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="de",
        help="Language of the audio file. Defaults to 'de'.",
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
        default="Herzlich willkommen zur Vorlesung! Legen wir los.",
        help="Initial prompt for the model. Defaults to 'Herzlich willkommen zur Vorlesung! Legen wir los.'.",
    )
    args = parser.parse_args()

    audio_path = args.audio_path or ".".join(args.video_path.split(".")[:-1]) + ".mp3"
    transcription_path = (
        args.transcription_path or ".".join(args.video_path.split(".")[:-1]) + ".txt"
    )

    skip_audio: bool = False
    if Path(audio_path).exists():
        print(f"Audio file already exists at {audio_path}.")
        skip_audio = input("Skip audio extraction? [y/N] ").lower() == "y"
    if not skip_audio:
        extract_audio(args.video_path, audio_path)

    start = time.time()
    transcribe_audio(
        audio_path,
        transcription_path,
        language=args.language,
        device=args.device,
        initial_prompt=args.initial_prompt,
    )
    elapsed = time.time() - start
    print(f"Transcription took {elapsed} seconds.")
    print("Done!")


if __name__ == "__main__":
    main()
