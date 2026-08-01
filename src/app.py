import os
import shutil
import uuid
import threading
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List

from src.highlight_engine import HighlightEngine
from src.video_cropper import VideoCropper
from src.caption_generator import CaptionGenerator
from src.voiceover_engine import VoiceoverEngine

app = FastAPI(title="Global Highlights - Auto-Clip Engine (Local-Upgraded)")

# Setup directories
UPLOAD_DIR = os.path.abspath("uploads")
CLIPS_DIR = os.path.abspath("clips")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

templates = Jinja2Templates(directory="src/templates")

# In-memory storage for current active video and analysis results
app_state = {
    "current_video_path": "",
    "current_video_name": "",
    "highlights": []
}

# Thread-safe in-memory job status storage
export_jobs = {}

@app.get("/")
async def home():
    """Serves the main dashboard page."""
    template_path = os.path.join("src", "templates", "index.html")
    if os.path.exists(template_path):
        return FileResponse(template_path, media_type="text/html")
    return {"message": "Global Highlights Auto-Clip Engine dashboard. Please place index.html in src/templates/"}

@app.post("/upload")
def upload_video(file: UploadFile = File(...)):
    """Handles uploading a long-form video file."""
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".mp4", ".mov", ".mkv", ".avi"]:
            raise HTTPException(status_code=400, detail="Unsupported video format. Please upload an MP4, MOV, MKV, or AVI video.")
            
        file_id = str(uuid.uuid4())[:8]
        safe_filename = f"{file_id}_{file.filename}"
        dest_path = os.path.join(UPLOAD_DIR, safe_filename)
        
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        app_state["current_video_path"] = dest_path
        app_state["current_video_name"] = file.filename
        app_state["highlights"] = []
        
        return {
            "status": "success",
            "filename": file.filename,
            "saved_path": dest_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/demo")
def load_demo():
    """Generates a synthetic demo video instantly if the user doesn't have an upload."""
    demo_path = os.path.join(UPLOAD_DIR, "demo_video.mp4")
    
    if not os.path.exists(demo_path):
        print("Creating programmatically generated demo video...")
        try:
            import cv2
            import math
            import numpy as np
            from gtts import gTTS
            from moviepy import AudioFileClip, ImageClip
            
            demo_text = (
                "Welcome to the Global Highlights test video. In this video, "
                "we have highly energetic talking moments! Look at this incredible movement "
                "on the screen right now. This is a total game changer for video creators! "
                "You can see how auto cropping tracks speaking characters smoothly, "
                "burning beautiful captions on the fly. Amazing hacks for Shorts and Reels!"
            )
            temp_demo_audio = "temp_demo_audio.mp3"
            tts = gTTS(text=demo_text, lang='en')
            tts.save(temp_demo_audio)
            
            audio_clip = AudioFileClip(temp_demo_audio)
            audio_dur = audio_clip.duration
            
            width, height = 1280, 720
            fps = 24
            total_frames = int(audio_dur * fps)
            
            temp_silent_video = "temp_demo_silent.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_silent_video, fourcc, fps, (width, height))
            
            face_x = width // 2
            face_y = height // 2
            
            for f in range(total_frames):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                for grid_x in range(0, width, 80):
                    cv2.line(frame, (grid_x, 0), (grid_x, height), (30, 30, 30), 1)
                for grid_y in range(0, height, 80):
                    cv2.line(frame, (0, grid_y), (width, grid_y), (30, 30, 30), 1)
                
                time_sec = f / fps
                offset_x = int(math.sin(time_sec * 1.5) * 350)
                curr_face_x = face_x + offset_x
                
                cv2.circle(frame, (curr_face_x, face_y), 90, (147, 20, 255), -1)
                cv2.circle(frame, (curr_face_x - 30, face_y - 20), 15, (255, 255, 255), -1)
                cv2.circle(frame, (curr_face_x + 30, face_y - 20), 15, (255, 255, 255), -1)
                cv2.circle(frame, (curr_face_x - 30, face_y - 20), 6, (0, 0, 0), -1)
                cv2.circle(frame, (curr_face_x + 30, face_y - 20), 6, (0, 0, 0), -1)
                
                mouth_height = max(2, int(10 + math.sin(time_sec * 8) * 15))
                cv2.ellipse(frame, (curr_face_x, face_y + 30), (25, mouth_height), 0, 0, 180, (0, 0, 255), -1)
                
                cv2.putText(frame, "Global Highlights Demo Source (16:9)", (50, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, "TRACKING TARGET ACTIVE", (curr_face_x - 150, face_y - 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                video_writer.write(frame)
                
            video_writer.release()
            
            from moviepy import VideoFileClip
            with VideoFileClip(temp_silent_video) as silent_video:
                demo_video_clip = silent_video.with_audio(audio_clip)
                demo_video_clip.write_videofile(
                    demo_path,
                    codec="libx264",
                    audio_codec="aac",
                    logger=None
                )
                
            for temp_f in [temp_demo_audio, temp_silent_video]:
                if os.path.exists(temp_f):
                    os.remove(temp_f)
            print("Demo video created successfully!")
        except Exception as e:
            print("Error creating synthetic demo video:", e)
            raise HTTPException(status_code=500, detail=f"Failed to generate demo clip: {str(e)}")
            
    app_state["current_video_path"] = demo_path
    app_state["current_video_name"] = "Demo_Video_16_9.mp4"
    app_state["highlights"] = []
    
    return {
        "status": "success",
        "filename": "Demo_Video_16_9.mp4",
        "saved_path": demo_path
    }

class AnalyzeRequest(BaseModel):
    use_whisper: bool = True
    whisper_model: str = "tiny"
    use_ollama: bool = False
    ollama_model: str = "llama3.2:3b"
    ollama_host: str = "http://localhost:11434"

@app.post("/analyze")
def analyze_video(req: AnalyzeRequest):
    """Runs the HighlightEngine with local-upgraded features (faster-whisper and Ollama semantic pass)."""
    video_path = app_state["current_video_path"]
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(
            status_code=400, 
            detail="No active video loaded. Please upload a video or load the demo. (If you just did that and still see this, the server may have restarted — try again.)"
        )
        
    try:
        engine = HighlightEngine(video_path)
        highlights = engine.detect_highlights(
            top_n=5,
            use_whisper=req.use_whisper,
            whisper_model=req.whisper_model,
            use_ollama=req.use_ollama,
            ollama_model=req.ollama_model,
            ollama_host=req.ollama_host
        )
        
        app_state["highlights"] = highlights
        
        return {
            "status": "success",
            "video_name": app_state["current_video_name"],
            "duration": engine.duration,
            "highlights": highlights
        }
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

class ExportRequest(BaseModel):
    clip_index: int
    track_faces: bool = True
    caption_style: str = "Reels"
    enable_voiceover: bool = False
    voiceover_text: str = ""
    accent_tld: str = "com"
    use_kokoro: bool = False
    kokoro_voice: str = "af_bella"

def perform_background_export(job_id: str, req: ExportRequest, video_path: str, clip_data: dict):
    """Background worker task that updates progress dynamically for live polling."""
    try:
        start_time = clip_data["start"]
        end_time = clip_data["end"]
        transcript_text = clip_data["text"]
        
        clip_id = str(uuid.uuid4())[:6]
        cropped_path = os.path.join(CLIPS_DIR, f"crop_{clip_id}.mp4")
        final_output_path = os.path.join(CLIPS_DIR, f"highlight_{clip_id}.mp4")
        srt_path = os.path.join(CLIPS_DIR, f"highlight_{clip_id}.srt")
        
        # 1. Video cropping with face-tracking callback
        def update_progress_callback(percent):
            # Scale face tracking crop progress to range [5%, 50%]
            scaled = 5 + int(percent * 0.45)
            export_jobs[job_id]["progress"] = scaled
            
        export_jobs[job_id]["stage"] = "Step 1/3: Running face-tracking vertical crop with smooth glide..."
        export_jobs[job_id]["progress"] = 5
        
        cropper = VideoCropper(tflite_model_path="blaze_face_short_range.tflite")
        crop_success = cropper.crop_to_vertical(
            video_path, cropped_path, start_time, end_time, 
            track_faces=req.track_faces, update_progress=update_progress_callback
        )
        if not crop_success or not os.path.exists(cropped_path):
            raise Exception("Failed to crop video to vertical format using local models.")
            
        # 2. Voiceover and Audio Ducking
        export_jobs[job_id]["stage"] = "Step 2/3: Generating custom narration & mixing with background volume ducking..."
        export_jobs[job_id]["progress"] = 55
        
        voiceover_output_path = os.path.join(CLIPS_DIR, f"voice_{clip_id}.mp4")
        if req.enable_voiceover and req.voiceover_text.strip():
            vo_engine = VoiceoverEngine()
            vo_success = vo_engine.overlay_voiceover_on_video(
                cropped_path, req.voiceover_text, voiceover_output_path, 
                duck_ratio=0.25, use_kokoro=req.use_kokoro, kokoro_voice=req.kokoro_voice
            )
            if vo_success and os.path.exists(voiceover_output_path):
                os.remove(cropped_path)
                os.rename(voiceover_output_path, cropped_path)
                
        # 3. Dynamic Karaoke Subtitles & SRT Generation
        export_jobs[job_id]["stage"] = "Step 3/3: Running phonetic timing estimators and burning animated subtitles..."
        export_jobs[job_id]["progress"] = 75
        
        caption_gen = CaptionGenerator()
        words_timing = caption_gen.estimate_word_timings(transcript_text, 0, end_time - start_time)
        caption_gen.generate_srt(words_timing, srt_path)
        
        export_jobs[job_id]["progress"] = 85
        
        if req.caption_style != "None":
            cap_success = caption_gen.burn_captions_opencv(
                cropped_path, final_output_path, words_timing, style_type=req.caption_style
            )
            if not cap_success or not os.path.exists(final_output_path):
                shutil.copy(cropped_path, final_output_path)
        else:
            shutil.copy(cropped_path, final_output_path)
            
        if os.path.exists(cropped_path):
            os.remove(cropped_path)
            
        final_filename = os.path.basename(final_output_path)
        srt_filename = os.path.basename(srt_path)
        
        # Complete
        export_jobs[job_id] = {
            "status": "completed",
            "stage": "Render complete! Ready for download.",
            "progress": 100,
            "result": {
                "download_url": f"/download/{final_filename}",
                "srt_url": f"/download/{srt_filename}",
                "clip_filename": final_filename,
                "srt_filename": srt_filename,
                "duration": end_time - start_time,
                "start": start_time,
                "end": end_time
            },
            "error": None
        }
    except Exception as e:
        print(f"Background export error: {e}")
        export_jobs[job_id] = {
            "status": "failed",
            "stage": "Render failed",
            "progress": 0,
            "result": None,
            "error": str(e)
        }

@app.post("/export")
def export_clip(req: ExportRequest, background_tasks: BackgroundTasks):
    """
    Spawns an asynchronous background thread task to render the clip,
    immediately returning a job_id for real-time progress polling.
    """
    video_path = app_state["current_video_path"]
    highlights = app_state["highlights"]
    
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=400, detail="No active video loaded. Please upload or reload demo.")
        
    if not highlights or req.clip_index >= len(highlights):
        raise HTTPException(status_code=400, detail="Invalid clip selection index.")
        
    job_id = str(uuid.uuid4())[:8]
    export_jobs[job_id] = {
        "status": "processing",
        "stage": "Initializing render task...",
        "progress": 0,
        "result": None,
        "error": None
    }
    
    clip_data = highlights[req.clip_index]
    
    # Offload the CPU-intensive render task to a background thread task
    background_tasks.add_task(
        perform_background_export, job_id, req, video_path, clip_data
    )
    
    return {
        "status": "queued",
        "job_id": job_id
    }

@app.get("/export/status/{job_id}")
def get_export_status(job_id: str):
    """Returns the current real-time progress percentage and stage message for a render job."""
    if job_id not in export_jobs:
        raise HTTPException(status_code=404, detail="Requested render job not found.")
    return export_jobs[job_id]

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Serves the generated highlight clip MP4 or SRT file for download."""
    file_path = os.path.join(CLIPS_DIR, filename)
    if not os.path.exists(file_path):
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Requested file not found.")
            
    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=filename
    )
