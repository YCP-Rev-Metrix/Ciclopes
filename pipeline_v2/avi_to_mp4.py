import subprocess
import sys


def convert(input_path: str, output_path: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        output_path,
    ]
    subprocess.run(cmd, check=True)
    print(f"Converted: {input_path} -> {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input.avi> <output.mp4>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
