# FutuGen Audio Transcription & Classification Pipeline

## Introduction

This project implements a real-time audio processing pipeline that transcribes conversations and automatically classifies them as either **private** or **topic-based** discussions. Built as a technical assessment for FutuGen, it demonstrates a streaming architecture for audio intelligence applications.

### What It Does

1. **Streams audio** in configurable chunks with overlap (default: 2.5s chunks, 0.5s overlap)
2. **Transcribes** each chunk using OpenAI Whisper with word-level timestamps
3. **Detects speakers** using heuristic silence-gap analysis
4. **Classifies conversations incrementally** - updates classification after each chunk using semantic embeddings
5. **Extracts metadata** including topics, sentiment, privacy signals, and participant count

---

## Architecture

The system follows a **streaming async pipeline** pattern:

```
AudioStreamer → TranscriptionEngine → ClassifierEngine
     ↓               ↓                      ↓
  AudioChunk → TranscriptionRecord → ClassificationResult
```

### Core Components

**1. AudioStreamer** (`src/audio_streamer/streamer.py`)
- Loads WAV files and converts to mono 16kHz
- Splits into overlapping chunks with Unix timestamps
- Base64 encodes audio for JSON serialization
- Outputs `AudioChunk` objects

**2. TranscriptionEngine** (`src/transcription/engine.py`)
- Transcribes audio chunks using local Whisper model
- Applies sliding window context (0.5s from previous chunk)
- Provides word-level timestamps and confidence scores
- Detects speaker changes (naive: >1s silence = speaker switch)
- Retry logic with exponential backoff (3 retries, 60s timeout)
- Outputs `TranscriptionRecord` objects to `transcription.jsonl`

**3. ClassifierEngine** (`src/classifier/engine.py`)
- Uses `sentence-transformers` for semantic similarity
- Classifies as "private" or "topic_based" via cosine similarity
- Extracts topics through keyword matching
- Detects privacy signals and analyzes sentiment
- Outputs `ClassificationResult` to `classification.json`

**4. AudioPipeline** (`src/pipeline.py`)
- Orchestrates the entire flow
- Streams → Transcribes → Classifies → Saves results

### Configuration

Uses **Pydantic models** with YAML config (`configs/pipeline_config.yaml`):

```yaml
project:
  output_dir: "output"
  data_dir: "data"

audio_streamer:
  chunk_duration: 2.5        # seconds
  overlap: 0.5               # seconds
  target_sample_rate: 16000  # Hz

transcription:
  model_name: "base"         # tiny/base/small/medium/large
  context_duration: 0.5      # seconds

classifier:
  model_name: "all-MiniLM-L6-v2"
  class_descriptions:
    private: "Personal, confidential conversation..."
    topic_based: "Professional, business discussion..."
```

---

## How to Run

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd FutureGen-Task

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Prerequisites:** Python 3.10+, FFmpeg, ~2GB RAM

### Run Pipeline

The CLI requires both an **audio file** and a **config file** as parameters.

**Basic usage:**
```bash
python scripts/main.py <audio_file.wav> --config <config.yaml>
```

**Example (quiet mode - no logs):**
```bash
python scripts/main.py data/short_mono_16k.wav --config configs/pipeline_config.yaml
```

**With INFO logging (-v):**
```bash
python scripts/main.py data/short_mono_16k.wav -c configs/pipeline_config.yaml -v
```

**With DEBUG logging (-vv):**
```bash
python scripts/main.py data/short_mono_16k.wav -c configs/pipeline_config.yaml -vv
```

**With custom session ID:**
```bash
python scripts/main.py audio.wav -c configs/pipeline_config.yaml -s "meeting_20240329" -v
```

**View help:**
```bash
python scripts/main.py --help
```

**Command structure:**
- `audio_file` (REQUIRED): Path to WAV audio file
- `--config` / `-c` (REQUIRED): Path to YAML config file
- `--session-id` / `-s` (optional): Custom session identifier (auto-generated from filename if not provided)
- `--verbose` / `-v` (optional): Verbosity level (no `-v`: quiet, `-v`: INFO logs, `-vv`: DEBUG logs)

**Exit codes:**
- `0` = Success
- `1` = Runtime error
- `2` = Invalid arguments (e.g., file not found)

**Output behavior:**
- **Console logs**: Real-time processing logs (INFO level) showing progress through chunks
- **Streaming output**: Transcription records appended to `transcription.jsonl` as each chunk is processed (append-only)
- **Incremental classification**: `classification.json` updated after each chunk (live updates as conversation progresses)

Output files are saved to `output/` directory (configurable in YAML):
- `transcription.jsonl` - Transcription records streamed incrementally (one per chunk)
- `classification.json` - Live classification updates (overwrites with latest after each chunk)

### Output Schema

**TranscriptionRecord:**
```json
{
  "record_id": "uuid",
  "chunk_id": "uuid",
  "session_id": "string",
  "speaker_id": "SPEAKER_01",
  "text": "transcribed text",
  "language": "en",
  "timestamp_start": 1700000000.0,
  "timestamp_end": 1700000002.5,
  "confidence": 0.94,
  "words": [{"word": "text", "start": 0.0, "end": 0.5}]
}
```

**ClassificationResult:**
```json
{
  "classification_id": "uuid",
  "session_id": "string",
  "classification": "topic_based",
  "confidence": 0.87,
  "topics": ["topic1", "topic2"],
  "privacy_signals": [],
  "dominant_topic": "topic1",
  "sentiment": "positive",
  "participants_count": 2,
  "total_duration_s": 187.4,
  "model_used": "whisper-base + all-MiniLM-L6-v2"
}
```

### Configuration Tuning

**Faster processing:**
```yaml
transcription:
  model_name: "tiny"
```

**Better accuracy:**
```yaml
transcription:
  model_name: "medium"
```

**Adjust chunking:**
```yaml
audio_streamer:
  chunk_duration: 5.0
  overlap: 1.0
```

### Testing

```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
./lint_script.sh          # Format and lint
```

---

## Implementation Details & Design Decisions

### Whisper Integration: Local vs API

**Choice: Local Whisper Model**

I chose local Whisper over OpenAI API for privacy, cost, and latency. Audio data stays on-device, no per-request charges, and faster processing without network round-trips. Trade-off is ~2GB RAM requirement and local model storage.

**Implementation:**
- Word-level timestamps from Whisper output
- Sliding window: 0.5s context overlap between chunks
- Retry logic: 3 attempts with exponential backoff (60s timeout)
- Confidence scores from `avg_logprob` and `no_speech_prob`
- Streaming append-only output to `transcription.jsonl`

### Speaker Diarization

**Choice: Silence-gap heuristic**

I implemented naive speaker detection: gaps > 1.0s trigger speaker switch. This is simple and fast but not production-ready. Cannot distinguish multiple speakers accurately or handle overlapping speech. For production, use `pyannote.audio` for proper voice-based diarization.

### Classifier Approach

**Choice: Sentence-Transformers + Heuristics (Approach 3)**

I used semantic embeddings (`all-MiniLM-L6-v2`) with cosine similarity for classification. This balances accuracy, speed, and resource usage.

**Why not alternatives:**
- LLM-based (GPT-4): Too expensive and slow for real-time processing
- Classical NLP (TF-IDF): Too rigid, misses semantic meaning
- Embeddings: Fast (100ms), lightweight (80MB), works on CPU

**Architecture:**
```
Text → Embeddings → Cosine Similarity → Classification (private/topic_based)
     + Keyword matching for topics
     + Word counting for sentiment
```

**Limitations:**
- Topic extraction uses basic keyword matching (misses synonyms)
- Sentiment analysis is word-counting only (no context)
- Binary classification (no fine-grained categories)

---

## Known Limitations

1. **Test suite**: E2E tests load real models (~3s overhead). Need better mocking.

2. **Speaker detection**: Naive silence-based heuristic. Use `pyannote.audio` for production.

3. **Classification**: Simplistic keyword/sentiment logic. Fine-tune embeddings on labeled data or add LLM fallback for better accuracy.

4. **Audio formats**: WAV only. Manual conversion required for MP3/M4A.

5. **Streaming**: Processes complete files only. No real-time live audio support.

---

