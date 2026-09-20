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

from huggingface_hub import login
from faster_whisper import WhisperModel
from pathlib import Path
from tqdm import tqdm

import os
import gc

# Option 1: Hardcoded token
# Replace the placeholder with your own Hugging Face token.
login(token="Use your own Hugging Face token here")

# Option 2: Environment variable (recommended)
# login(token=os.getenv("HF_TOKEN"))

# Replace with the path to your input directory.
INPUT_DIR = Path(r"C:\Users\your_username\path\to\audio")
MODEL_SIZE = "large"   # change to "medium" if needed

model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

audio_files = sorted(INPUT_DIR.glob("*.m4a"))

for audio_file in tqdm(audio_files, desc="Files", unit="file"):
    segments, info = model.transcribe(
        str(audio_file),
        language="da",
        vad_filter=True
    )

    text_parts = []
    segment_count = 0

    for seg in segments:
        if seg.text and seg.text.strip():
            text_parts.append(seg.text.strip())
    print()  # newline after current file

    transcript = " ".join(text_parts).strip()
    out_file = audio_files.with_suffix(".txt")
    out_file.write_text(transcript, encoding="utf-8")
    print(f"Saved: {out_file}")    
    
    gc.collect()