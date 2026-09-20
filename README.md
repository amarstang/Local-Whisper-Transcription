# Local Whisper Transcription

Python scripts for transcribing audio files locally using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper).

These scripts were originally used for privacy-sensitive transcription where
the audio needed to remain on the local machine rather than being uploaded to
an external transcription service.

Local processing helped support the privacy and data-handling requirements of
the workflow, including GDPR considerations.

> **Disclaimer:** Whisper transcription is not 100% accurate. The generated text should be reviewed and corrected where necessary. The scripts are intended to handle most of the transcription work and provide a useful starting point, not a guaranteed final transcript.

## Features

- Local audio transcription using Whisper
- GPU-accelerated transcription using CUDA
- Support for `.m4a` audio files
- Voice Activity Detection (VAD)
- Configurable Whisper model size
- Batch processing of multiple audio files
- Automatic `.txt` output
- Optional batching for machines with more GPU resources
- Small utility for formatting generated transcripts

## Repository Contents

### `PC_Whisper.py`

GPU-oriented transcription script intended for a more powerful machine.

It:

- Uses `faster-whisper`
- Runs the Whisper model using CUDA and `float16`
- Processes all matching audio files in a directory
- Supports configurable batch sizes
- Retries with a smaller batch if necessary
- Writes each transcript to a `.txt` file next to the original audio file

### `Laptop_Whisper.py`

Simpler version of the transcription script.

It also performs local transcription using CUDA but does not use the additional
batch-size fallback logic found in the PC version.

### `PythonNewLine.py`

Small text-processing utility that inserts a newline after each period in a
generated transcript.

This can make long transcription output easier to read or process further.

## Requirements

The scripts use:

- Python
- `faster-whisper`
- `huggingface_hub`
- `tqdm`

Install the Python dependencies with:

```bash
pip install faster-whisper huggingface-hub tqdm
```

GPU transcription additionally requires a compatible NVIDIA GPU and the
required CUDA libraries.

## Hugging Face Authentication

The scripts contain two possible approaches for supplying a Hugging Face token.

### Environment variable - recommended

Set your token as an environment variable:

```text
HF_TOKEN
```

Then use:

```python
login(token=os.getenv("HF_TOKEN"))
```

### Hardcoded token

A token can also be supplied directly:

```python
login(token="YOUR_HUGGING_FACE_TOKEN")
```

Do not commit a real access token to a public repository.

## Configuration

Set the directory containing the audio files:

```python
INPUT_DIR = Path(r"C:\Users\your_username\path\to\audio")
```

Select the Whisper model:

```python
MODEL_NAME = "large"
```

Select the input file type:

```python
AUDIO_PATTERN = "*.m4a"
```

For the batched version, the batch sizes can also be adjusted:

```python
BATCH_SIZE = 4
FALLBACK_BATCH_SIZE = 2
```

Reduce the batch size if GPU memory is limited.

## Running the Transcription

Run one of the transcription scripts:

```bash
python PC_Whisper.py
```

or:

```bash
python Laptop_Whisper.py
```

Each matching audio file is transcribed and saved as a `.txt` file in the same
directory.

For example:

```text
recording.m4a
```

produces:

```text
recording.txt
```

## Privacy

The transcription itself is performed locally using `faster-whisper`.

This approach was chosen for a workflow involving privacy-sensitive audio so
that source audio did not need to be sent to a third-party transcription
service.

Depending on the configuration, model files and authentication may still
involve communication with Hugging Face, for example when downloading model
weights. The audio transcription itself is performed on the local machine.

This repository should not by itself be considered a guarantee of GDPR
compliance. GDPR compliance depends on the complete data-processing workflow,
including storage, access control, retention, lawful basis, and other
organizational and technical measures.

## Notes

The scripts currently specify Danish as the transcription language:

```python
language="da"
```

Change this value if another language is required.

The GPU configuration currently uses:

```python
device="cuda"
compute_type="float16"
```

For CPU-based execution, the model configuration would need to be changed
accordingly.
