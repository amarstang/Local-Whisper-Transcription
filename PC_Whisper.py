# | Model name | Parameters | Relative speed (CPU) | Accuracy  | Typical use                  |
# | ---------- | ---------- | -------------------- | --------- | ---------------------------- |
# | `tiny`     | ~39M       | Very fast            | Low       | Quick drafts, low importance |
# | `base`     | ~74M       | Fast                 | Medium    | Lightweight tasks            |
# | `small`    | ~244M      | Moderate             | Good      | **Best default**             |
# | `medium`   | ~769M      | Slow                 | Very good | Higher accuracy              |
# | `large`    | ~1550M     | Very slow            | Best      | Maximum quality              |

# compute_type="int8"       # best for CPU
# compute_type="float16"    # GPU
# compute_type="int8_float16"  # hybrid

import os
import gc

from huggingface_hub import login
from faster_whisper import WhisperModel
from pathlib import Path
from tqdm import tqdm

# Disable Hugging Face symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Option 1: Hardcoded token
# Replace the placeholder with your own Hugging Face token.
login(token="Use your own Hugging Face token here")

# Option 2: Environment variable (recommended)
# login(token=os.getenv("HF_TOKEN"))



INPUT_DIR = Path(r"C:\Users\your_username\path\to\audio")
MODEL_NAME = "large"
AUDIO_PATTERN = "*.m4a"
BATCH_SIZE = 4 # Number of audio segments processed together - Reduce this value if the GPU runs out of memory.
FALLBACK_BATCH_SIZE = 2 # Smaller batch size used if the initial batch fails.


# Initialize the Whisper model using the GPU.
model = WhisperModel(
    MODEL_NAME,
    device="cuda",
    compute_type="float16"
)

# Find and sort all matching audio files in the input directory.
audio_files = sorted(INPUT_DIR.glob(AUDIO_PATTERN))

# Transcribe each audio file individually.
for audio_file in tqdm(audio_files, desc="Files", unit="file"):
    print(f"Processing: {audio_file.name}")

    try:
        # Attempt transcription using the configured batch size.
        segments, info = model.transcribe(
            str(audio_file),
            language="da",
            vad_filter=True,
            batch_size=BATCH_SIZE
        )

    except TypeError:
        # Some faster-whisper versions may not support batch_size.
        print("Batching not supported, retrying without it...")

        segments, info = model.transcribe(
            str(audio_file),
            language="da",
            vad_filter=True
        )

    except RuntimeError:
        # Retry with a smaller batch if the initial batch causes a runtime error,
        # for example due to insufficient GPU memory.
        print(f"Retrying {audio_file.name} with a smaller batch...")

        try:
            segments, info = model.transcribe(
                str(audio_file),
                language="da",
                vad_filter=True,
                batch_size=FALLBACK_BATCH_SIZE
            )

        except TypeError:
            # Fall back to transcription without batching.
            segments, info = model.transcribe(
                str(audio_file),
                language="da",
                vad_filter=True
            )

    # Collect non-empty transcription segments.
    transcript_parts = []

    for segment in segments:
        if segment.text and segment.text.strip():
            transcript_parts.append(segment.text.strip())

    # Combine all segments into one transcript.
    transcript = " ".join(transcript_parts)

    # Save the transcript next to the original audio file as a .txt file.
    output_file = audio_file.with_suffix(".txt")
    output_file.write_text(transcript, encoding="utf-8")

    print(f"Saved: {output_file}")

    # Run Python's garbage collector to help prevent memory buildup
    # when processing multiple files.
    gc.collect()