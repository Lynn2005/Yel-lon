import os
import re
import json
import time
import shutil
import hashlib
import subprocess
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime
import requests
import streamlit as st

st.set_page_config(page_title="Lynn Recap · One Click", page_icon="🎬", layout="wide")
st.markdown("<style>" + Path("style.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)

ROOT = Path("jobs")
ROOT.mkdir(exist_ok=True)
STEPS = [
    (5, "Validating video", "validation"),
    (10, "Extracting audio", "audio"),
    (20, "Transcribing movie", "transcript"),
    (30, "Creating original SRT", "original_srt"),
    (40, "Translating into Burmese", "translation"),
    (50, "Understanding story and writing recap", "recap"),
    (60, "Creating recap subtitles", "recap_srt"),
    (70, "Generating AI voice", "voice"),
    (80, "Syncing voice and subtitles", "sync"),
    (90, "Rendering final video", "render"),
    (98, "Validating final output", "final_validation"),
]

def run_cmd(args, timeout=None):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError((p.stderr or "Processing command failed")[-1800:])
    return p.stdout

def safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)[:120] or "movie.mp4"

def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default if default is not None else {}

def update_status(job, step, progress, message, completed=None, state="processing"):
    status = read_json(job / "status.json", {})
    status.update({"jobId": job.name, "status": state, "step": step, "progress": progress,
                   "message": message, "updatedAt": datetime.utcnow().isoformat() + "Z"})
    if completed is not None:
        status["completed"] = completed
    save_json(job / "status.json", status)

def srt_time(seconds):
    ms = max(0, int(round(float(seconds) * 1000)))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def make_srt(segments):
    rows = []
    last_end = 0.0
    for i, seg in enumerate(segments, 1):
        start = max(0.0, float(seg.get("start", 0)))
        end = max(start + 0.15, float(seg.get("end", start + 1)))
        # Correct invalid/overlapping timestamps instead of writing a broken SRT.
        start = max(start, last_end)
        end = max(end, start + 0.15)
        text = str(seg.get("text", "")).strip()
        if text:
            rows.append(f"{len(rows)+1}\n{srt_time(start)} --> {srt_time(end)}\n{text}")
            last_end = end
    return "\n\n".join(rows) + ("\n" if rows else "")

def parse_video(path):
    raw = run_cmd(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    data = json.loads(raw)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float(data.get("format", {}).get("duration") or 0)
    if not video or duration <= 0:
        raise ValueError("Video file could not be processed.")
    if not audio:
        raise ValueError("This video has no readable audio track.")
    return {"duration": duration, "width": video.get("width"), "height": video.get("height"),
            "fps": video.get("r_frame_rate"), "videoCodec": video.get("codec_name"),
            "audioCodec": audio.get("codec_name"), "size": path.stat().st_size,
            "audioAvailable": True}

def extract_audio(video, out):
    run_cmd(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-map", "0:a:0",
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)], timeout=1800)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError("Audio extraction returned an empty file.")

def groq_transcribe(audio, key, model="whisper-large-v3-turbo"):
    # Groq's transcription upload limit requires chunking long audio.
    size = audio.stat().st_size
    max_bytes = 24 * 1024 * 1024
    chunks = []
    if size <= max_bytes:
        chunks = [audio]
    else:
        duration = float(run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                  "-of", "default=noprint_wrappers=1:nokey=1", str(audio)]).strip())
        chunk_seconds = max(120, int(duration * max_bytes / size * 0.82))
        pattern = audio.parent / "stt_chunk_%03d.mp3"
        run_cmd(["ffmpeg", "-y", "-v", "error", "-i", str(audio), "-f", "segment",
                 "-segment_time", str(chunk_seconds), "-c:a", "libmp3lame", "-b:a", "48k", str(pattern)], timeout=1800)
        chunks = sorted(audio.parent.glob("stt_chunk_*.mp3"))
    all_segments, full_text, offset = [], [], 0.0
    for chunk in chunks:
        payload = None
        for attempt in range(3):
            try:
                with open(chunk, "rb") as f:
                    r = requests.post("https://api.groq.com/openai/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {key}"},
                        files={"file": (chunk.name, f)},
                        data={"model": model, "response_format": "verbose_json", "timestamp_granularities[]": "segment"},
                        timeout=600)
                if r.status_code in (429, 500, 502, 503, 504):
                    if attempt < 2:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                r.raise_for_status()
                payload = r.json()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
        if payload:
            full_text.append(payload.get("text", ""))
            for seg in payload.get("segments", []):
                all_segments.append({"start": float(seg.get("start", 0)) + offset,
                                     "end": float(seg.get("end", 0)) + offset,
                                     "text": seg.get("text", "").strip()})
            if len(chunks) > 1:
                offset += float(run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                         "-of", "default=noprint_wrappers=1:nokey=1", str(chunk)]).strip())
    return {"text": "\n".join(full_text).strip(), "segments": all_segments}

def gemini_text(prompt, key, model="gemini-3.8-flash"):
    # Gemini 2.0 Flash has been shut down; use currently supported model IDs.
    models = [model, "gemini-3.7-flash", "gemini-3.5-flash-lite"]
    last_error = ""
    for chosen in dict.fromkeys(models):
        for attempt in range(3):
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{chosen}:generateContent",
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=180)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                data = r.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception as e:
                last_error = str(e)
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
    raise RuntimeError("AI text generation failed. Check API key, model access and quota. " + last_error[:300])

def split_chunks(text, limit=7000):
    paras = re.split(r"(?<=[.!?။])\s+|\n+", text.strip())
    chunks, current = [], ""
    for para in paras:
        if not para:
            continue
        if len(current) + len(para) + 1 > limit and current:
            chunks.append(current)
            current = ""
        current = (current + " " + para).strip()
    if current:
        chunks.append(current)
    return chunks

def translate_burmese(transcript_data, key, model):
    segments = transcript_data.get("segments", [])
    if not segments:
        text = translate_burmese_text(transcript_data.get("text", ""), key, model)
        return text, []
    translated_segments = []
    # Translate batches while preserving the original timestamps for a usable Burmese SRT.
    batch_size = 80
    for start in range(0, len(segments), batch_size):
        batch = segments[start:start + batch_size]
        payload = [{"id": start + i + 1, "text": str(s.get("text", "")).strip()} for i, s in enumerate(batch)]
        prompt = ("Translate each movie dialogue segment into natural, concise spoken Myanmar Burmese. "
                  "Return ONLY a valid JSON array of {id,text} objects. Preserve every id, timestamp alignment, "
                  "character name, meaning, and segment; do not add commentary.\n"
                  + json.dumps(payload, ensure_ascii=False))
        raw = gemini_text(prompt, key, model)
        try:
            match = re.search(r"\[.*\]", raw, re.S)
            values = json.loads(match.group(0) if match else raw)
            mapping = {int(x["id"]): str(x["text"]).strip() for x in values}
        except Exception:
            # Repair malformed JSON by translating the batch as plain text.
            plain = gemini_text("Translate this movie transcript into natural Myanmar Burmese. Return only translation.\n\n" +
                                "\n".join(x["text"] for x in payload), key, model).splitlines()
            mapping = {x["id"]: (plain[i] if i < len(plain) else x["text"]) for i, x in enumerate(payload)}
        for i, seg in enumerate(batch, start + 1):
            translated_segments.append({"start": float(seg.get("start", 0)), "end": float(seg.get("end", 0)),
                                        "text": mapping.get(i, str(seg.get("text", "")).strip())})
    return "\n".join(x["text"] for x in translated_segments), translated_segments

def translate_burmese_text(transcript, key, model):
    chunks = split_chunks(transcript, 7000)
    translated = []
    for i, chunk in enumerate(chunks, 1):
        prompt = ("Translate this movie dialogue transcript into natural spoken Myanmar Burmese. "
                  "Preserve character names, meaning, story order, and every important detail. "
                  "Do not summarize, invent events, add explanations, or omit dialogue. Return only Burmese translation.\n\n"
                  f"CHUNK {i}/{len(chunks)}:\n{chunk}")
        translated.append(gemini_text(prompt, key, model))
    return "\n\n".join(translated)

def generate_recap(transcript, translation, length, style, key, model):
    word_targets = {"Auto": "choose a suitable length based on story complexity",
                    "1–2 Minutes": "about 250–350 Burmese words",
                    "3–5 Minutes": "about 500–800 Burmese words",
                    "5–10 Minutes": "about 1000–1500 Burmese words",
                    "10–15 Minutes": "about 1500–2200 Burmese words",
                    "Custom": "a concise but complete recap"}
    # Map-reduce summary avoids sending a full feature-length transcript in one request.
    source = translation or transcript
    chunks = split_chunks(source, 6500)
    mini = []
    for i, chunk in enumerate(chunks, 1):
        mini.append(gemini_text(
            "Summarize this section of a movie transcript in Burmese. Preserve events in order, character names, relationships, motives and reveals. Do not invent information. Return only the summary.\n"
            f"SECTION {i}/{len(chunks)}:\n{chunk}", key, model))
    master = "\n".join(mini)
    prompt = (f"Write a natural Myanmar Burmese movie recap for voice narration, style: {style}; target: {word_targets[length]}. "
              "Tell the story in chronological order: beginning, conflict, development, twist, climax and ending. "
              "No unnecessary intro, repeated sentences, fake information, invented scenes, headings, markdown, emojis or stage directions. "
              "Preserve consistent character names and include the actual ending supported by the source. Return only clean Burmese narration.\n\n"
              f"ORIGINAL TRANSCRIPT EXCERPT:\n{transcript[:12000]}\n\n"
              f"BURMESE TRANSLATION / CHUNK SUMMARIES:\n{master[:45000]}")
    result = gemini_text(prompt, key, model)
    return re.sub(r"[\*#\[\]<>]", "", result).strip()

def make_recap_segments(script, chars=25):
    # Split on sentence boundaries first; long sentences are wrapped at word boundaries.
    sentences = [x.strip() for x in re.split(r"(?<=[။.!?])\s+|\n+", script) if x.strip()]
    segments = []
    for sentence in sentences:
        words = sentence.split()
        lines, line = [], ""
        for word in words:
            if len(line) + len(word) + (1 if line else 0) > chars and line:
                lines.append(line)
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            lines.append(line)
        for j in range(0, len(lines), 2):
            segments.append({"text": "\n".join(lines[j:j+2])})
    return segments

async def edge_chunk(text, path, voice, speed):
    import edge_tts
    rate = f"{int((float(speed) - 1) * 100):+d}%"
    await edge_tts.Communicate(text, voice, rate=rate).save(str(path))

def generate_voice(script, job, speed, voice):
    voice_dir = job / "voice"
    voice_dir.mkdir(exist_ok=True)
    chunks = [x.strip() for x in re.split(r"(?<=[။.!?])\s+|\n+", script) if x.strip()]
    if not chunks:
        raise ValueError("Recap script is empty.")
    made = []
    for i, chunk in enumerate(chunks, 1):
        out = voice_dir / f"segment_{i:04d}.mp3"
        if not out.exists() or out.stat().st_size == 0:
            try:
                asyncio.run(edge_chunk(chunk, out, voice, speed))
            except Exception:
                # Retry the failed chunk only.
                time.sleep(1)
                asyncio.run(edge_chunk(chunk, out, voice, speed))
        made.append(out)
    concat_file = voice_dir / "concat.txt"
    concat_file.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in made), encoding="utf-8")
    full = job / "voice_full.mp3"
    run_cmd(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat_file),
             "-c:a", "libmp3lame", "-q:a", "3", str(full)], timeout=1800)
    return full, chunks

def wrap_subtitle_text(text, max_chars=25, max_lines=2):
    """Wrap Burmese subtitle text into short, readable lines."""
    text = re.sub(r"\\s+", " ", str(text)).strip()
    if not text:
        return ""
    max_chars = max(12, min(35, int(max_chars)))
    words = text.split(" ")
    lines, line = [], ""
    for word in words:
        # Keep Burmese phrase chunks together where possible; split only unusually long chunks.
        pieces = [word[i:i + max_chars] for i in range(0, len(word), max_chars)] if len(word) > max_chars else [word]
        for piece in pieces:
            candidate = (line + " " + piece).strip()
            if len(candidate) > max_chars and line:
                lines.append(line)
                line = piece
            else:
                line = candidate
    if line:
        lines.append(line)
    # More than two lines makes subtitles cover too much of the image; use separate timed cues.
    return lines

def normalize_srt_file(path, max_chars=25):
    """Reflow an existing recap SRT into short two-line cues before re-rendering."""
    if not path.exists():
        return
    raw = path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        return
    blocks = re.split(r"\n\s*\n", raw)
    output = []
    def parse_time(value):
        hh, mm, rest = value.split(":")
        ss, ms = rest.split(",")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000
    for block in blocks:
        rows = block.splitlines()
        if len(rows) < 3 or "-->" not in rows[1]:
            continue
        try:
            start_raw, end_raw = [x.strip() for x in rows[1].split("-->", 1)]
            start, end = parse_time(start_raw), parse_time(end_raw)
        except (ValueError, IndexError):
            continue
        lines = wrap_subtitle_text(" ".join(rows[2:]), max_chars=max_chars)
        groups = [lines[i:i + 2] for i in range(0, len(lines), 2)]
        if not groups:
            continue
        for i, group in enumerate(groups):
            cue_start = start + (end - start) * i / len(groups)
            cue_end = start + (end - start) * (i + 1) / len(groups)
            output.append({"start": cue_start, "end": max(cue_start + 0.15, cue_end),
                           "text": "\n".join(group)})
    if output:
        path.write_text(make_srt(output), encoding="utf-8")

def create_timed_srt(chunks, voice_file, out, max_chars=25):
    # Segment duration is measured from the generated audio, not estimated from text.
    durations = []
    for i in range(1, len(chunks) + 1):
        p = voice_file.parent / "voice" / f"segment_{i:04d}.mp3"
        d = float(run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                           "-of", "default=noprint_wrappers=1:nokey=1", str(p)]).strip())
        durations.append(max(0.2, d))
    t, segments = 0.0, []
    for text, duration in zip(chunks, durations):
        lines = wrap_subtitle_text(text, max_chars=max_chars)
        if not lines:
            t += duration + 0.15
            continue
        # Divide long sentences into timed cues so no subtitle occupies more than two lines.
        cue_groups = [lines[i:i + 2] for i in range(0, len(lines), 2)]
        usable = max(0.2, duration)
        for idx, group in enumerate(cue_groups):
            cue_start = t + usable * idx / len(cue_groups)
            cue_end = t + usable * (idx + 1) / len(cue_groups)
            segments.append({"start": cue_start, "end": max(cue_start + 0.15, cue_end),
                             "text": "\\n".join(group)})
        t += duration + 0.15
    out.write_text(make_srt(segments), encoding="utf-8")

def font_path():
    # Prefer the Myanmar fonts supplied for this project, then use system fallbacks.
    project_fonts = Path(__file__).resolve().parent / "fonts"
    candidates = [
        project_fonts / "ဧက ၀၃ - Regular.ttf",
        project_fonts / "ဧက ၀၈ - Regular.ttf",
        Path("/usr/share/fonts/truetype/noto/NotoSansMyanmar-Regular.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansMyanmar-VF.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSerifMyanmar-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    return next((str(p) for p in candidates if p.exists()), None)

def subtitle_font_name():
    path = font_path()
    if path:
        filename = Path(path).name
        if filename == "ဧက ၀၃ - Regular.ttf":
            return "A ka 03"
        if filename == "ဧက ၀၈ - Regular.ttf":
            return "A ka 08"
    return "Noto Sans Myanmar"

def render_video(video, voice, srt, output, ratio, subtitle_on, bgm_on=False,
                 blur_on=False, blur_strength=10, blur_x=25, blur_y=25, blur_w=35, blur_h=25,
                 blur_style="Gaussian", mirror=False, logo_on=False, logo_path=None,
                 logo_position="Top right", logo_size=15, bgm_path=None, bgm_volume=20):
    """Render narration video with independently optional region blur, logo and background music."""
    args = ["ffmpeg", "-y", "-i", str(video), "-i", str(voice)]
    use_logo = bool(logo_on and logo_path and Path(logo_path).exists())
    use_bgm = bool(bgm_on and bgm_path and Path(bgm_path).exists())
    if use_logo:
        args += ["-i", str(logo_path)]
    logo_index = 2 if use_logo else None
    if use_bgm:
        args += ["-i", str(bgm_path)]
    bgm_index = (3 if use_logo else 2) if use_bgm else None

    vf = []
    if ratio == "9:16 · Reels/Shorts":
        vf.append("scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280")
    elif ratio == "16:9 · YouTube":
        vf.append("scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2")
    elif ratio == "1:1 · Square":
        vf.append("scale=720:720:force_original_aspect_ratio=increase,crop=720:720")
    if mirror:
        vf.append("hflip")

    # Blur the original image first, then burn Burmese subtitles on top so the new
    # subtitles stay sharp and readable instead of being blurred with the source.
    sub_filter = None
    if subtitle_on and srt and srt.exists():
        sub = str(srt.resolve()).replace("\\", "/").replace(":", r"\\:").replace("'", r"\\'")
        style = "FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=58"
        if font_path():
            style += f",FontName={subtitle_font_name()}"
            style += f",fontsdir={str(Path(font_path()).parent).replace(chr(92), chr(47))}"
        sub_filter = f"subtitles='{sub}':force_style='{style}'"

    graph = []
    base_chain = ",".join(vf) if vf else "null"
    if blur_on:
        # Coordinates and dimensions are percentages of the output frame; clamp to frame bounds.
        x = max(0, min(95, int(blur_x)))
        y = max(0, min(95, int(blur_y)))
        w = max(5, min(100 - x, int(blur_w)))
        h = max(5, min(100 - y, int(blur_h)))
        blur_strength = max(1, min(30, int(blur_strength)))
        if blur_style == "Pixelate":
            blur_filter = f"scale=iw/12:ih/12:flags=neighbor,scale=iw*12:ih*12:flags=neighbor"
        else:
            blur_filter = f"boxblur={blur_strength}:2"
        graph.append(f"[0:v]{base_chain},split=2[clean][blurinput]")
        graph.append(
            f"[blurinput]crop=w=iw*{w}/100:h=ih*{h}/100:x=iw*{x}/100:y=ih*{y}/100,"
            f"{blur_filter}[blurred]"
        )
        graph.append(f"[clean][blurred]overlay=x=W*{x}/100:y=H*{y}/100:shortest=1[region]")
        current = "region"
    else:
        graph.append(f"[0:v]{base_chain}[base]")
        current = "base"

    if sub_filter:
        graph.append(f"[{current}]{sub_filter}[subtitled]")
        current = "subtitled"

    if use_logo:
        positions = {
            "Top left": "20:20",
            "Top right": "W-w-20:20",
            "Bottom left": "20:H-h-20",
            "Bottom right": "W-w-20:H-h-20",
            "Center": "(W-w)/2:(H-h)/2",
        }
        pos = positions.get(logo_position, "W-w-20:20")
        graph.append(f"[{logo_index}:v][{current}]scale2ref=w=main_w*{max(5, min(50, int(logo_size)))}/100:h=-1[logo][ref]")
        graph.append(f"[ref][logo]overlay={pos}[withlogo]")
        current = "withlogo"

    if use_bgm:
        volume = max(0, min(100, int(bgm_volume))) / 100
        graph.append(f"[1:a]volume=1[voiceaudio]")
        graph.append(f"[{bgm_index}:a]volume={volume}[musicaudio]")
        graph.append("[voiceaudio][musicaudio]amix=inputs=2:duration=first:dropout_transition=2,loudnorm=I=-16:TP=-1.5:LRA=11[aout]")
        audio_map = "[aout]"
    else:
        audio_map = "1:a:0"

    args += ["-filter_complex", ";".join(graph), "-map", f"[{current}]", "-map", audio_map,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-c:a", "aac", "-b:a", "192k"]
    if not use_bgm:
        args += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
    args += ["-shortest", "-movflags", "+faststart", str(output)]
    run_cmd(args, timeout=7200)

def create_live_edit_frame(video, srt, output, ratio, subtitle_on,
                           blur_on=False, blur_strength=10, blur_x=25, blur_y=25,
                           blur_w=35, blur_h=25, blur_style="Gaussian", mirror=False,
                           preview_time=3):
    """Render one lightweight still frame so Live Edit controls can be previewed immediately."""
    vf = []
    if ratio == "9:16 · Reels/Shorts":
        vf.append("scale=360:640:force_original_aspect_ratio=increase,crop=360:640")
    elif ratio == "16:9 · YouTube":
        vf.append("scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2")
    elif ratio == "1:1 · Square":
        vf.append("scale=360:360:force_original_aspect_ratio=increase,crop=360:360")
    if mirror:
        vf.append("hflip")
    base_chain = ",".join(vf) if vf else "null"
    graph = []
    if blur_on:
        x = max(0, min(95, int(blur_x)))
        y = max(0, min(95, int(blur_y)))
        w = max(5, min(100 - x, int(blur_w)))
        h = max(5, min(100 - y, int(blur_h)))
        strength = max(1, min(30, int(blur_strength)))
        blur_filter = (f"scale=iw/12:ih/12:flags=neighbor,scale=iw*12:ih*12:flags=neighbor"
                       if blur_style == "Pixelate" else f"boxblur={strength}:2")
        graph.append(f"[0:v]{base_chain},split=2[clean][blurinput]")
        graph.append(f"[blurinput]crop=w=iw*{w}/100:h=ih*{h}/100:x=iw*{x}/100:y=ih*{y}/100,{blur_filter}[blurred]")
        graph.append(f"[clean][blurred]overlay=x=W*{x}/100:y=H*{y}/100:shortest=1[region]")
        current = "region"
    else:
        graph.append(f"[0:v]{base_chain}[base]")
        current = "base"
    if subtitle_on and srt and Path(srt).exists():
        sub = str(Path(srt).resolve()).replace("\\", "/").replace(":", r"\\:").replace("'", r"\\'")
        style = "FontSize=12,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=24"
        if font_path():
            style += f",FontName={subtitle_font_name()}"
            style += f",fontsdir={str(Path(font_path()).parent).replace(chr(92), chr(47))}"
        graph.append(f"[{current}]subtitles='{sub}':force_style='{style}'[subtitled]")
        current = "subtitled"
    run_cmd(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(max(0, float(preview_time))), "-i", str(video),
             "-filter_complex", ";".join(graph), "-map", f"[{current}]",
             "-frames:v", "1", "-q:v", "4", str(output)], timeout=120)
    return output

def validate_final(path):
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError("Final rendering failed.")
    info = json.loads(run_cmd(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))
    streams = info.get("streams", [])
    if not any(s.get("codec_type") == "video" for s in streams) or not any(s.get("codec_type") == "audio" for s in streams):
        raise RuntimeError("Final rendering failed.")
    if float(info.get("format", {}).get("duration") or 0) <= 0:
        raise RuntimeError("Final rendering failed.")

st.markdown("""
<div class="hero">
  <div class="hero-kicker">LYNN RECAP STUDIO · AI WORKFLOW</div>
  <div class="hero-title">Turn any movie into a<br/>Myanmar recap video.</div>
  <div class="hero-copy">Upload your movie, choose your preferred style, and let the pipeline handle transcription, Burmese translation, narration, subtitles, and final rendering.</div>
  <div class="pill-row"><span class="pill">✦ One-click processing</span><span class="pill">🇲🇲 Myanmar narration</span><span class="pill">🎞️ MP4 export</span></div>
</div>
""", unsafe_allow_html=True)
with st.sidebar:
    st.header("🔑 API Settings")
    from streamlit_local_storage import LocalStorage
    local_storage = LocalStorage()

    # Each browser-storage component needs its own stable Streamlit key.
    # The value can be None on the first render while the browser component loads.
    if "saved_groq_key" not in st.session_state:
        stored_groq = local_storage.getItem("yel_lon_groq_api_key")
        st.session_state.saved_groq_key = stored_groq or os.getenv("GROQ_API_KEY", "")
    if "saved_gemini_key" not in st.session_state:
        stored_gemini = local_storage.getItem("yel_lon_gemini_api_key")
        st.session_state.saved_gemini_key = stored_gemini or os.getenv("GEMINI_API_KEY", "")

    st.markdown("[Get Groq API key ↗](https://console.groq.com/keys)")
    st.text_input("Groq API Key", type="password", key="groq_key_input",
                  value=st.session_state.saved_groq_key)
    st.markdown("[Get Gemini API key ↗](https://aistudio.google.com/app/apikey)")
    st.text_input("Gemini API Key", type="password", key="gemini_key_input",
                  value=st.session_state.saved_gemini_key)
    if st.button("💾 Save API Keys", use_container_width=True):
        groq_to_save = st.session_state.get("groq_key_input", "").strip()
        gemini_to_save = st.session_state.get("gemini_key_input", "").strip()
        local_storage.setItem("yel_lon_groq_api_key", groq_to_save, key="save_groq_api_key")
        local_storage.setItem("yel_lon_gemini_api_key", gemini_to_save, key="save_gemini_api_key")
        st.session_state.saved_groq_key = groq_to_save
        st.session_state.saved_gemini_key = gemini_to_save
        st.success("Saved in this browser. Reloading the page should keep these values.")
    groq_key = st.session_state.get("groq_key_input", st.session_state.saved_groq_key).strip()
    gemini_key = st.session_state.get("gemini_key_input", st.session_state.saved_gemini_key).strip()
    st.caption("Stored in this browser only. Do not save keys on a shared device, and never put keys in public GitHub code.")
    st.divider()
    st.subheader("⚙️ Recap Settings")
    recap_length = st.selectbox("Recap Length", ["Auto", "1–2 Minutes", "3–5 Minutes", "5–10 Minutes", "10–15 Minutes", "Custom"], index=2)
    recap_style = st.selectbox("Recap Style", ["Natural Storytelling", "Cinematic", "Suspense", "Casual", "Fast paced"])
    voice_choice = st.selectbox("Voice", ["Natural Female", "Natural Male"])
    voice_speed = st.select_slider("Voice Speed", options=["0.8", "0.9", "1.0", "1.1", "1.2"], value="1.0")
    ratio = st.selectbox("Aspect Ratio", ["Original", "9:16 · Reels/Shorts", "16:9 · YouTube", "1:1 · Square"])
    subtitle_on = st.toggle("Burn Burmese subtitles into video", value=True)
    line_chars = st.selectbox("Subtitle characters per line", [20, 25, 30, 35], index=1)
    model = st.selectbox("Gemini Model", ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite"], index=0)
    with st.expander("Advanced Settings"):
        st.caption("STT: Groq Whisper · LLM: Gemini · TTS: Edge TTS fallback")
        st.caption("FFmpeg/FFprobe are checked at runtime.")
        st.caption("Streamlit Community Cloud storage is temporary; jobs can be lost after restart.")

st.markdown("## 01 · Upload your movie")
st.caption("MP4, MKV, MOV, WEBM or AVI · Configure API keys and output preferences in the sidebar first.")
video_file = st.file_uploader("Drop your movie here or browse files", type=["mp4", "mkv", "mov", "webm", "avi"], help="Choose a movie file with a readable audio track.")
if video_file:
    st.success(f"Selected: {video_file.name} · {video_file.size / 1024 / 1024:.1f} MB")
    st.video(video_file)

if "current_job" not in st.session_state:
    st.session_state.current_job = ""
# Restore the most recent job after a browser refresh / Streamlit session reset.
# The editor was hidden because current_job only lived in session_state.
if not st.session_state.current_job:
    try:
        recoverable_jobs = sorted(
            (p for p in ROOT.iterdir() if p.is_dir() and (p / "metadata.json").exists()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if recoverable_jobs:
            st.session_state.current_job = recoverable_jobs[0].name
    except OSError:
        pass
if "job_error" not in st.session_state:
    st.session_state.job_error = ""

st.markdown("## 02 · Generate your recap")
st.caption("One click runs the complete workflow from audio extraction to final MP4.")
start = st.button("✦  GENERATE MY RECAP", type="primary", use_container_width=True,
                  disabled=not (video_file and groq_key and gemini_key))
if start:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        st.error("FFmpeg/FFprobe မတွေ့ပါ။ packages.txt ကိုစစ်ပြီး app ကို reboot လုပ်ပါ။")
    else:
        digest = hashlib.sha256(video_file.getvalue()).hexdigest()[:12]
        job_id = "job_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + digest
        job = ROOT / job_id
        job.mkdir(parents=True, exist_ok=True)
        st.session_state.current_job = job_id
        st.session_state.job_error = ""
        input_path = job / ("input" + (Path(safe_name(video_file.name)).suffix.lower() or ".mp4"))
        if not input_path.exists():
            input_path.write_bytes(video_file.getvalue())
        settings = {"recapLength": recap_length, "style": recap_style, "voice": voice_choice,
                    "speed": voice_speed, "ratio": ratio, "subtitles": subtitle_on, "lineChars": line_chars,
                    "model": model, "videoHash": digest}
        save_json(job / "metadata.json", {"jobId": job_id, "filename": safe_name(video_file.name), "settings": settings,
                                          "createdAt": datetime.utcnow().isoformat() + "Z"})
        completed = []
        progress = st.progress(0, text="Starting job...")
        current = st.empty()
        try:
            def stage(step, pct, message, artifact=None):
                nonlocal_progress = pct
                progress.progress(nonlocal_progress, text=f"{nonlocal_progress}% · {message}")
                current.info(f"🎬 {message}")
                update_status(job, step, pct, message, completed)
            stage("validation", 5, "Validating video")
            info = parse_video(input_path)
            save_json(job / "video_info.json", info)
            completed.append("validation")
            stage("audio", 10, "Extracting and normalizing audio")
            audio = job / "audio.wav"
            if not audio.exists():
                extract_audio(input_path, audio)
            completed.append("audio")
            stage("transcript", 20, "Transcribing movie with Whisper")
            transcript_path = job / "transcript.txt"
            transcript_json = job / "transcript.json"
            if not transcript_json.exists():
                data = groq_transcribe(audio, groq_key)
                save_json(transcript_json, data)
                transcript_path.write_text(data.get("text", ""), encoding="utf-8")
            data = read_json(transcript_json)
            transcript = transcript_path.read_text(encoding="utf-8") if transcript_path.exists() else data.get("text", "")
            completed.append("transcript")
            stage("original_srt", 30, "Creating original SRT")
            original_srt = job / "original.srt"
            if not original_srt.exists():
                original_srt.write_text(make_srt(data.get("segments", [])), encoding="utf-8")
            completed.append("original_srt")
            stage("translation", 40, "Translating dialogue into Burmese")
            translation_path = job / "burmese_translation.txt"
            translated_json_path = job / "burmese_segments.json"
            if not translation_path.exists() or not translated_json_path.exists():
                translation, translated_segments = translate_burmese(data, gemini_key, model)
                translation_path.write_text(translation, encoding="utf-8")
                save_json(translated_json_path, translated_segments)
            else:
                translation = translation_path.read_text(encoding="utf-8")
                translated_segments = read_json(translated_json_path, [])
            burmese_srt_path = job / "burmese.srt"
            if translated_segments and not burmese_srt_path.exists():
                burmese_srt_path.write_text(make_srt(translated_segments), encoding="utf-8")
            completed.append("translation")
            stage("recap", 50, "Understanding story and writing recap")
            recap_path = job / "recap.txt"
            if not recap_path.exists():
                recap_path.write_text(generate_recap(transcript, translation, recap_length, recap_style, gemini_key, model), encoding="utf-8")
            recap = recap_path.read_text(encoding="utf-8")
            if not recap.strip():
                raise RuntimeError("Recap script generation returned empty text.")
            completed.append("recap")
            stage("recap_srt", 60, "Creating recap subtitle segments")
            recap_srt = job / "recap.srt"
            # Final timed SRT is generated after actual TTS durations; this preliminary file is still useful for stage recovery.
            if not recap_srt.exists():
                prelim = []
                t = 0.0
                for seg in make_recap_segments(recap, line_chars):
                    prelim.append({"start": t, "end": t + 3.5, "text": seg["text"]})
                    t += 3.5
                recap_srt.write_text(make_srt(prelim), encoding="utf-8")
            completed.append("recap_srt")
            stage("voice", 70, "Generating Burmese AI voice")
            voice_file = job / "voice_full.mp3"
            voice_name = "my-MM-NilarNeural" if voice_choice == "Natural Female" else "my-MM-ThihaNeural"
            chunks_path = job / "voice_chunks.json"
            if not voice_file.exists():
                voice_file, voice_chunks = generate_voice(recap, job, voice_speed, voice_name)
                save_json(chunks_path, voice_chunks)
            else:
                voice_chunks = read_json(chunks_path, [])
                if not voice_chunks:
                    voice_chunks = [x.strip() for x in re.split(r"(?<=[။.!?])\s+|\n+", recap) if x.strip()]
            completed.append("voice")
            stage("sync", 80, "Syncing subtitles to actual voice timing")
            create_timed_srt(voice_chunks, voice_file, recap_srt, line_chars)
            completed.append("sync")
            stage("render", 90, "Rendering final MP4 with narration")
            final_path = job / "final.mp4"
            render_video(input_path, voice_file, recap_srt, final_path, ratio, subtitle_on,
                         blur_strength=0, mirror=False, logo_path=None)
            completed.append("render")
            stage("final_validation", 98, "Validating final video")
            validate_final(final_path)
            completed.append("final_validation")
            update_status(job, "complete", 100, "Recap complete", completed, "complete")
            progress.progress(100, text="100% · Complete")
            current.success("✅ YOUR RECAP IS READY")
        except Exception as e:
            st.session_state.job_error = str(e)
            update_status(job, "failed", int(read_json(job / "status.json", {}).get("progress", 0)),
                          "Processing failed. Retry the failed step after checking settings.", completed, "failed")
            st.error("❌ Processing failed. Completed files have been kept in this job folder.")
            st.code(str(e)[:1800])
            st.caption("Streamlit Community Cloud က restart ဖြစ်လျှင် local job files ပျောက်နိုင်သည်။ Long movies အတွက် persistent disk ပါသော host လိုနိုင်သည်။")

job_id = st.session_state.current_job
if job_id:
    job = ROOT / job_id
    if job.exists():
        status = read_json(job / "status.json", {})
        if status:
            st.divider()
            st.subheader("📊 Job Status")
            st.progress(min(100, int(status.get("progress", 0))) / 100, text=f"{status.get('progress', 0)}% · {status.get('message', '')}")
            st.caption(f"Job ID: {job_id} · Status: {status.get('status', 'unknown')}")
        final_path = job / "final.mp4"
        edit_srt_path = job / "recap.srt"
        st.markdown("## 🎛️ Live Edit Studio")
        st.caption("မူရင်းစာတန်းကို Blur လုပ်ပြီး မြန်မာစာတန်းကို အပေါ်က ကြည်လင်စွာတင်နိုင်ပါတယ်။ ချိန်ညှိပြီး Preview ထုတ်ပါ။")
        input_video_path = next((p for p in job.iterdir() if p.name.startswith("input") and p.is_file()), None)
        preview_col, guide_col = st.columns([3, 2])
        with preview_col:
            st.markdown("### 🎞️ Live Preview")
            st.caption("Effect controls ပြောင်းတာနဲ့ preview ပုံ update ဖြစ်မယ်။ အောက်က slider နဲ့ စမ်းကြည့်မယ့်အချိန်ကို ရွေးနိုင်ပါတယ်။")
            preview_time = st.slider(
                "🎞️ Preview time (seconds)", min_value=0, max_value=60, value=3, step=1,
                key=f"{job_id}_live_preview_time",
                help="ဗီဒီယိုထဲက ဘယ်အချိန် frame ကို စစ်ကြည့်မလဲ ရွေးပါ။"
            )
            if input_video_path:
                try:
                    frame_path = job / f"live_edit_frame_{int(preview_time)}.jpg"
                    create_live_edit_frame(
                        input_video_path, edit_srt_path, frame_path, ratio,
                        st.session_state.get(f"{job_id}_edit_subtitle", True),
                        blur_on=st.session_state.get(f"{job_id}_blur_on", False),
                        blur_strength=st.session_state.get(f"{job_id}_blur_strength", 10),
                        blur_x=st.session_state.get(f"{job_id}_blur_x", 25),
                        blur_y=st.session_state.get(f"{job_id}_blur_y", 25),
                        blur_w=st.session_state.get(f"{job_id}_blur_w", 35),
                        blur_h=st.session_state.get(f"{job_id}_blur_h", 25),
                        blur_style=st.session_state.get(f"{job_id}_blur_style", "Gaussian"),
                        mirror=st.session_state.get(f"{job_id}_edit_mirror", False),
                        preview_time=preview_time,
                    )
                    st.image(str(frame_path), use_container_width=True)
                except Exception as preview_error:
                    st.warning("Live still preview မထုတ်နိုင်သေးပါ။ Apply edits နှိပ်ပြီး full preview စမ်းနိုင်ပါတယ်။")
                    st.caption(str(preview_error)[:300])
        with guide_col:
            st.info("မူရင်းစာတန်းရှိတဲ့နေရာကို Blur X/Y နဲ့ Width/Height ချိန်ပါ။ မြန်မာစာတန်းကို Blur မဖြစ်ဘဲ အပေါ်မှာ ထပ်တင်ထားတာကို preview မှာကြည့်ပါ။ နောက်ဆုံးဗီဒီယိုအတွက် **Apply edits & render preview** ကိုနှိပ်ပါ။")
        if edit_srt_path.exists():
            current_srt = edit_srt_path.read_text(encoding="utf-8")
            edited_srt_text = st.text_area(
                "📝 Subtitle Live Edit (SRT format)", value=current_srt,
                height=260, key=f"{job_id}_live_srt_editor",
                help="စာသားနဲ့ timestamp ကို SRT format အတိုင်း ပြင်ပါ။"
            )
        else:
            edited_srt_text = ""
            st.info("Recap SRT မရှိသေးပါ။ Recap ထုတ်ပြီးမှ စာတန်းပြင်နိုင်ပါတယ်။")
        st.markdown("### 🎚️ Effect Switches")
        effect_a, effect_b, effect_c = st.columns(3)
        with effect_a:
            edit_subtitle_on = st.toggle("📝 Subtitles ON/OFF", value=True, key=f"{job_id}_edit_subtitle")
            blur_on = st.toggle("🌫️ Region Blur ON/OFF", value=False, key=f"{job_id}_blur_on")
        with effect_b:
            mirror_on = st.toggle("🪞 Mirror ON/OFF", value=False, key=f"{job_id}_edit_mirror")
            logo_on = st.toggle("🏷️ Logo ON/OFF", value=False, key=f"{job_id}_logo_on")
        with effect_c:
            bgm_on = st.toggle("🎵 Background Music ON/OFF", value=False, key=f"{job_id}_bgm_on")

        # One adjustment panel is visible at a time. Switches remain independent,
        # so enabled effects persist even when another panel is selected.
        adjust_options = ["📝 Text Adjust", "🌫️ Blur Adjust", "🏷️ Logo Adjust"]
        if f"{job_id}_active_adjust_panel" not in st.session_state:
            st.session_state[f"{job_id}_active_adjust_panel"] = "📝 Text Adjust"
        active_adjust = st.radio(
            "Adjust panel",
            adjust_options,
            horizontal=True,
            key=f"{job_id}_active_adjust_panel",
            label_visibility="collapsed",
        )

        if active_adjust == "📝 Text Adjust":
            st.markdown("#### 📝 Text Adjust")
            st.caption("စာတန်းကို ဖွင့်/ပိတ်ခြင်းနဲ့ SRT စာသားပြင်ခြင်းကို ဒီ panel မှာလုပ်ပါ။")
            st.caption("Subtitle ဖွင့်/ပိတ် control ကို အပေါ်က Effect Switches မှာထားထားပါတယ်။")
            st.caption("အောက်က Subtitle Live Edit (SRT format) မှာ စာသားနဲ့ timestamp ကို ပြင်နိုင်ပါတယ်။")

        # Keep effect state separate from the active panel: hiding a panel does not disable its effect.
        if blur_on and active_adjust == "🌫️ Blur Adjust":
            st.markdown("#### 🌫️ Blur Adjust")
            blur_x, blur_y = st.columns(2)
            with blur_x:
                blur_left = st.slider("Blur X position (%)", 0, 95, 25, key=f"{job_id}_blur_x")
                blur_width = st.slider("Blur width (%)", 5, 100, 35, key=f"{job_id}_blur_w")
            with blur_y:
                blur_top = st.slider("Blur Y position (%)", 0, 95, 25, key=f"{job_id}_blur_y")
                blur_height = st.slider("Blur height (%)", 5, 100, 25, key=f"{job_id}_blur_h")
            blur_strength = st.slider("Blur strength", 1, 30, 10, key=f"{job_id}_blur_strength")
            blur_style = st.selectbox("Blur style", ["Gaussian", "Pixelate"], key=f"{job_id}_blur_style")
            st.caption("X/Y က ဧရိယာရဲ့ ဘယ်ဘက်အပေါ်ထောင့်၊ width/height က အရွယ်အစား (%) ဖြစ်ပါတယ်။")
        else:
            blur_left = st.session_state.get(f"{job_id}_blur_x", 25)
            blur_top = st.session_state.get(f"{job_id}_blur_y", 25)
            blur_width = st.session_state.get(f"{job_id}_blur_w", 35)
            blur_height = st.session_state.get(f"{job_id}_blur_h", 25)
            blur_strength = st.session_state.get(f"{job_id}_blur_strength", 10)
            blur_style = st.session_state.get(f"{job_id}_blur_style", "Gaussian")

        logo_file = None
        logo_position = st.session_state.get(f"{job_id}_edit_logo_position", "Top right")
        logo_size = st.session_state.get(f"{job_id}_logo_size", 15)
        if logo_on and active_adjust == "🏷️ Logo Adjust":
            st.markdown("#### 🏷️ Logo Adjust")
            logo_position = st.selectbox("Logo position", ["Top right", "Top left", "Bottom right", "Bottom left", "Center"],
                                         key=f"{job_id}_edit_logo_position")
            logo_size = st.slider("Logo size (% of video width)", 5, 50, 15, key=f"{job_id}_logo_size")
            logo_file = st.file_uploader("Logo image (PNG/JPG)", type=["png", "jpg", "jpeg"],
                                         key=f"{job_id}_edit_logo_upload")
        elif logo_on:
            # Preserve uploaded logo and settings while its adjustment panel is hidden.
            logo_file = st.session_state.get(f"{job_id}_edit_logo_upload")
        bgm_file = None
        bgm_volume = st.session_state.get(f"{job_id}_bgm_volume", 20)
        if bgm_on:
            st.markdown("#### 🎵 Background Music Adjust")
            bgm_file = st.file_uploader("Background music (MP3/WAV/M4A)", type=["mp3", "wav", "m4a", "aac", "ogg"],
                                         key=f"{job_id}_bgm_upload")
            bgm_volume = st.slider("Music volume (%)", 0, 100, 20, key=f"{job_id}_bgm_volume")
            st.caption("0% = music အသံမကြား၊ 20% = နောက်ခံသံအနိမ့်။")

        if st.button("💾 Save subtitle edits", key=f"{job_id}_save_srt_edits", use_container_width=True):
            if edited_srt_text.strip():
                edit_srt_path.write_text(edited_srt_text.strip() + "\n", encoding="utf-8")
                st.success("စာတန်းပြင်ဆင်ချက် သိမ်းပြီးပါပြီ။")
                st.rerun()
            else:
                st.error("SRT စာသားအလွတ် မဖြစ်ရပါ။")
        if st.button("🎬 Apply edits & render preview", type="primary",
                     key=f"{job_id}_render_edits", use_container_width=True):
            try:
                if edited_srt_text.strip() and edit_srt_path.exists():
                    edit_srt_path.write_text(edited_srt_text.strip() + "\n", encoding="utf-8")
                if edit_subtitle_on:
                    normalize_srt_file(edit_srt_path, max_chars=25)
                logo_path = None
                if logo_on and logo_file is not None:
                    logo_path = job / "custom_logo.png"
                    logo_path.write_bytes(logo_file.getvalue())
                bgm_path = None
                if bgm_on and bgm_file is not None:
                    bgm_path = job / ("background_music" + Path(bgm_file.name).suffix.lower())
                    bgm_path.write_bytes(bgm_file.getvalue())
                edited_output = job / "final_edited.mp4"
                with st.spinner("Applying enabled effects and rendering preview..."):
                    render_video(
                        job / next(p.name for p in job.iterdir() if p.name.startswith("input") and p.is_file()),
                        job / "voice_full.mp3", edit_srt_path, edited_output, ratio,
                        edit_subtitle_on, blur_on=blur_on, blur_strength=blur_strength,
                        blur_x=blur_left, blur_y=blur_top, blur_w=blur_width, blur_h=blur_height,
                        blur_style=blur_style, mirror=mirror_on, logo_on=logo_on, logo_path=logo_path,
                        logo_position=logo_position, logo_size=logo_size,
                        bgm_on=bgm_on, bgm_path=bgm_path, bgm_volume=bgm_volume
                    )
                    validate_final(edited_output)
                st.success("Live Edit preview ready!")
                st.video(edited_output.read_bytes())
                st.download_button("⬇️ Download edited MP4", edited_output.read_bytes(),
                                   "final_edited.mp4", "video/mp4", key=f"{job_id}_download_edited")
            except Exception as e:
                st.error("Live Edit မအောင်မြင်ပါ။")
                st.code(str(e)[:1200])
        if final_path.exists() and final_path.stat().st_size:
            st.success("✅ RECAP COMPLETE")
            st.video(final_path.read_bytes())
            st.download_button("🎬 DOWNLOAD FINAL MP4", final_path.read_bytes(), "final.mp4", "video/mp4", use_container_width=True)
        download_files = [
            ("Original Transcript", "transcript.txt", "text/plain"),
            ("Original SRT", "original.srt", "application/x-subrip"),
            ("Burmese Translation", "burmese_translation.txt", "text/plain"),
            ("Burmese SRT", "burmese.srt", "application/x-subrip"),
            ("Recap Script", "recap.txt", "text/plain"),
            ("Recap SRT", "recap.srt", "application/x-subrip"),
            ("AI Voice", "voice_full.mp3", "audio/mpeg"),
        ]
        with st.expander("📥 Download files"):
            for label, filename, mime in download_files:
                p = job / filename
                if p.exists() and p.stat().st_size:
                    st.download_button(f"⬇️ {label}", p.read_bytes(), filename, mime, key=f"{job_id}_{filename}", use_container_width=True)

st.divider()
st.caption("Lynn Recap · API keys ကို public GitHub code ထဲ မထည့်ပါနှင့်။")
