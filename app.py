import os, io, re, json, shutil, subprocess, tempfile, zipfile
from pathlib import Path
import streamlit as st
import requests

st.set_page_config(page_title="Lynn Recap", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root{color-scheme:dark}
.stApp{background:radial-gradient(ellipse at 85% 0%,#1c2b50 0,transparent 38%),#090d18;color:#f5f7ff}
.block-container{max-width:1200px;padding-top:1.5rem}
section[data-testid="stSidebar"]{background:#101728}
div[data-testid="stMetric"],div[data-testid="stFileUploader"]{background:#111a2b;border:1px solid #293752;border-radius:16px;padding:14px}
.stButton>button{border-radius:12px;min-height:44px;font-weight:700}
h1,h2,h3{letter-spacing:-.4px}
.small-note{color:#93a2bb;font-size:.9rem}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Lynn Recap")
st.caption("Movie Recap Studio · Upload → Transcript → Recap → Voice → MP4")
with st.sidebar:
    st.header("🔑 API Settings")
    groq_key = st.text_input("Groq API Key", type="password", help="Whisper transcript အတွက် Groq API key")
    gemini_key = st.text_input("Gemini API Key", type="password", help="မြန်မာ recap script အတွက် Gemini API key")
    st.caption("Key များကို ဒီ app ထဲမှာ အမြဲတမ်းသိမ်းမထားပါ။")
    st.divider()
    st.subheader("⚙️ Output Settings")
    recap_length = st.selectbox("Recap အရှည်", ["Short (1–3 min)", "Medium (3–5 min)", "Long (5–10 min)"], index=1)
    voice_speed = st.select_slider("Voice speed", options=["0.85", "0.95", "1.0", "1.1", "1.2"], value="1.0")
    ratio = st.selectbox("Video ratio", ["9:16 · Reels/TikTok", "16:9 · YouTube", "1:1 · Square"])
    subtitle_mode = st.selectbox("Subtitle", ["Burn into video", "SRT file only", "No subtitles"])
    st.caption("အသုံးပြုမှုအတွက် Groq / Gemini API key လိုအပ်နိုင်ပါတယ်။")

def run_cmd(args):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or "Command failed")[-2500:])
    return p.stdout

def srt_time(seconds):
    ms=int(seconds*1000); h=ms//3600000; ms%=3600000; m=ms//60000; ms%=60000; s=ms//1000; ms%=1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def transcript_to_srt(segments):
    return "\n\n".join(f"{i+1}\n{srt_time(float(x.get('start',0)))} --> {srt_time(float(x.get('end',0)))}\n{x.get('text','').strip()}" for i,x in enumerate(segments))

def groq_transcribe(file_path, key):
    size=os.path.getsize(file_path)
    if size > 24*1024*1024:
        raise ValueError("Audio file is larger than 24 MB. Long videos need chunking, which this version does not yet do. Use a shorter video.")
    with open(file_path,"rb") as f:
        r=requests.post("https://api.groq.com/openai/v1/audio/transcriptions",
          headers={"Authorization":f"Bearer {key}"},
          files={"file":(Path(file_path).name,f,"audio/mpeg")},
          data={"model":"whisper-large-v3-turbo","response_format":"verbose_json","timestamp_granularities[]":["segment"]},timeout=300)
    if not r.ok: raise RuntimeError(f"Groq API error {r.status_code}: {r.text[:1000]}")
    return r.json()

def gemini_recap(text, key, length):
    words={"Short (1–3 min)":"400–600 words","Medium (3–5 min)":"700–1000 words","Long (5–10 min)":"1200–1800 words"}[length]
    prompt=f"""You are an expert Burmese movie recap writer. Based only on the transcript below, write a natural, engaging Myanmar (Burmese) language movie recap script of {words}. Keep the plot in chronological order, preserve character names when known, do not invent scenes, do not include headings or notes, and write conversational narration suitable for voice-over. Transcript may be incomplete; don't make unsupported claims.\n\nTRANSCRIPT:\n{text[:65000]}"""
    models=["gemini-2.5-flash","gemini-2.0-flash"]
    errors=[]
    for model in models:
        try:
            r=requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
              json={"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"temperature":0.65}},timeout=120)
            if r.ok:
                data=r.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            errors.append(f"{model}: {r.status_code} {r.text[:300]}")
        except Exception as e: errors.append(str(e))
    raise RuntimeError("Gemini request failed. " + " | ".join(errors))

def edge_voice(text, out_path, speed):
    import asyncio, edge_tts
    async def make():
        rate=f"{int((float(speed)-1)*100):+d}%"
        await edge_tts.Communicate(text, "my-MM-NilarNeural", rate=rate).save(out_path)
    asyncio.run(make())

def ffmpeg_render(video, audio, output, ratio):
    vf=None
    if ratio.startswith("9:16"): vf="scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280"
    elif ratio.startswith("16:9"): vf="scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720"
    elif ratio.startswith("1:1"): vf="scale=720:720:force_original_aspect_ratio=increase,crop=720:720"
    cmd=["ffmpeg","-y","-i",video,"-i",audio,"-map","0:v:0","-map","1:a:0","-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac","-shortest"]
    if vf: cmd += ["-vf",vf]
    cmd += ["-movflags","+faststart",output]
    run_cmd(cmd)

st.markdown("### ① Movie Video Upload")
video_file=st.file_uploader("Video file", type=["mp4","mov","mkv","webm"], help="Demo host တွင် video အရွယ်အစားနှင့် processing time ကန့်သတ်ချက်ရှိနိုင်ပါတယ်။")
if video_file:
    st.success(f"ရွေးထားသည်: {video_file.name} · {video_file.size/1024/1024:.1f} MB")
    st.video(video_file)
st.markdown("### ② Subtitle (Optional)")
srt_file=st.file_uploader("ကိုယ်ပိုင် SRT ရှိလျှင် တင်နိုင်ပါတယ်",type=["srt","txt"])
col1,col2=st.columns(2)
with col1:
    st.markdown("### ③ Transcript & Recap")
    do_transcript=st.button("📝 Extract audio + Original SRT",use_container_width=True,disabled=not (video_file and groq_key))
    do_recap=st.button("🇲🇲 Create Burmese recap script",use_container_width=True,disabled=not (groq_key and gemini_key))
with col2:
    st.markdown("### ④ Voice & Final Video")
    voice_upload=st.file_uploader("ကိုယ်ပိုင် voice file (optional)",type=["mp3","wav","m4a"])
    do_render=st.button("🎞️ Generate AI voice + Final MP4",use_container_width=True,disabled=not video_file)

if "transcript" not in st.session_state: st.session_state.transcript=""
if "srt" not in st.session_state: st.session_state.srt=""
if "recap" not in st.session_state: st.session_state.recap=""

if do_transcript:
    try:
        if not shutil.which("ffmpeg"): raise RuntimeError("FFmpeg မတွေ့ပါ။ packages.txt ထည့်ထားပြီး Streamlit app ကို reboot လုပ်ကြည့်ပါ။")
        with st.status("Audio ထုတ်ပြီး Transcript ဖန်တီးနေသည်…",expanded=True) as status:
            with tempfile.TemporaryDirectory() as td:
                vp=os.path.join(td,video_file.name)
                with open(vp,"wb") as f:f.write(video_file.getbuffer())
                ap=os.path.join(td,"audio.mp3")
                run_cmd(["ffmpeg","-y","-i",vp,"-vn","-ac","1","-ar","16000","-b:a","64k",ap])
                st.write("Audio extracted. Sending to Groq Whisper…")
                data=groq_transcribe(ap,groq_key)
                st.session_state.transcript=data.get("text","")
                st.session_state.srt=transcript_to_srt(data.get("segments",[]))
                status.update(label="Transcript ပြီးပါပြီ",state="complete")
        st.success("Original transcript ready.")
    except Exception as e: st.error(str(e))

if st.session_state.transcript:
    st.text_area("Original Transcript",key="transcript_edit",value=st.session_state.transcript,height=220)
    st.download_button("⬇️ Download original transcript TXT",st.session_state.transcript.encode(),"original_transcript.txt","text/plain")
if st.session_state.srt:
    st.download_button("⬇️ Download original SRT",st.session_state.srt.encode(),"original_subtitles.srt","application/x-subrip")

if do_recap:
    try:
        source=st.session_state.get("transcript_edit",st.session_state.transcript)
        if not source: raise ValueError("အရင်ဆုံး Original Transcript ထုတ်ပေးပါ။")
        with st.spinner("Gemini က မြန်မာ recap script ဖန်တီးနေသည်…"):
            st.session_state.recap=gemini_recap(source,gemini_key,recap_length)
        st.success("မြန်မာ recap script အဆင်သင့်ဖြစ်ပါပြီ။")
    except Exception as e: st.error(str(e))
if st.session_state.recap:
    st.text_area("မြန်မာ Recap Script",value=st.session_state.recap,height=260,key="recap_edit")
    st.download_button("⬇️ Download recap script",st.session_state.recap.encode(),"myanmar_recap.txt","text/plain")

if do_render:
    try:
        recap=st.session_state.get("recap_edit",st.session_state.recap)
        if not recap and not voice_upload: raise ValueError("အရင်ဆုံး မြန်မာ recap script ဖန်တီးပါ သို့မဟုတ် ကိုယ်ပိုင် voice တင်ပါ။")
        if not shutil.which("ffmpeg"): raise RuntimeError("FFmpeg မတွေ့ပါ။")
        with tempfile.TemporaryDirectory() as td:
            vp=os.path.join(td,video_file.name)
            with open(vp,"wb") as f:f.write(video_file.getbuffer())
            ap=os.path.join(td,"voice.mp3")
            if voice_upload:
                with open(ap,"wb") as f:f.write(voice_upload.getbuffer())
            else:
                if not recap: raise ValueError("Recap script မရှိပါ။")
                with st.spinner("မြန်မာ AI Voice ထုတ်နေသည်…"): edge_voice(recap,ap,voice_speed)
            out=os.path.join(td,"lynn_recap_final.mp4")
            with st.spinner("Final video render လုပ်နေသည်…"): ffmpeg_render(vp,ap,out,ratio)
            raw=Path(out).read_bytes()
            st.success("Final MP4 အဆင်သင့်ဖြစ်ပါပြီ။")
            st.video(raw)
            st.download_button("⬇️ Download Final MP4",raw,"lynn_recap_final.mp4","video/mp4",use_container_width=True)
    except Exception as e: st.error(str(e))

st.divider()
st.caption("Lynn Recap · API keys ကို public GitHub code ထဲ မထည့်ပါနှင့်။ ဒီ Streamlit session ထဲမှာသာ ထည့်သုံးပါ။")
