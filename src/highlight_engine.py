import os
import wave
import json
import re
import urllib.request
import time
import numpy as np
import speech_recognition as sr
from moviepy import VideoFileClip
import cv2

class HighlightEngine:
    def __init__(self, video_path):
        self.video_path = video_path
        self.duration = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.last_timing = {}
        
        # Get video metadata
        try:
            with VideoFileClip(video_path) as clip:
                self.duration = clip.duration
                self.fps = clip.fps if clip.fps else 30
                self.width = clip.size[0]
                self.height = clip.size[1]
        except Exception as e:
            print(f"Error loading video metadata: {e}")
            
    def extract_audio(self, output_wav_path, sample_rate=16000):
        """Extracts audio from video file to a WAV file."""
        print(f"Extracting audio from {self.video_path} to {output_wav_path}...")
        try:
            with VideoFileClip(self.video_path) as clip:
                if clip.audio is not None:
                    clip.audio.write_audiofile(
                        output_wav_path,
                        fps=sample_rate,
                        nbytes=2,
                        codec='pcm_s16le',
                        logger=None
                    )
                    return True
                else:
                    print("No audio track found in video.")
                    return False
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return False

    def analyze_audio_energy(self, wav_path, segment_duration=1.0):
        """Calculates RMS energy for each segment of the audio."""
        if not os.path.exists(wav_path):
            return []
            
        try:
            with wave.open(wav_path, 'rb') as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                
                # Read all audio frames
                raw_data = wf.readframes(n_frames)
                
                # Convert to numpy array
                if sampwidth == 1:
                    dtype = np.uint8
                    bias = 128
                elif sampwidth == 2:
                    dtype = np.int16
                    bias = 0
                else:
                    dtype = np.int32
                    bias = 0
                    
                audio_data = np.frombuffer(raw_data, dtype=dtype).astype(np.float32) - bias
                
                # If stereo, average channels
                if n_channels > 1:
                    audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)
                
                # Normalize audio
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data /= max_val
                    
                # Compute RMS in chunks
                chunk_size = int(sample_rate * segment_duration)
                num_chunks = int(len(audio_data) / chunk_size)
                
                rms_values = []
                for i in range(num_chunks):
                    chunk = audio_data[i * chunk_size : (i + 1) * chunk_size]
                    rms = np.sqrt(np.mean(chunk**2))
                    rms_values.append(float(rms)) # Cast numpy float to native float
                    
                return rms_values
        except Exception as e:
            print(f"Error analyzing audio energy: {e}")
            return []

    def transcribe_audio_segments(self, wav_path, segment_duration=15):
        """Transcribes the audio in chunks using SpeechRecognition."""
        if not os.path.exists(wav_path):
            return []
            
        recognizer = sr.Recognizer()
        transcripts = []
        
        try:
            with sr.AudioFile(wav_path) as source:
                total_duration = source.DURATION
                start = 0
                while start < total_duration:
                    end = min(start + segment_duration, total_duration)
                    try:
                        audio_chunk = recognizer.record(source, duration=segment_duration)
                        text = recognizer.recognize_google(audio_chunk, language="en-US")
                        transcripts.append({
                            "start": float(start),
                            "end": float(end),
                            "text": text
                        })
                    except sr.UnknownValueError:
                        transcripts.append({
                            "start": float(start),
                            "end": float(end),
                            "text": ""
                        })
                    except sr.RequestError as e:
                        print(f"Google speech API request failed: {e}")
                        transcripts.append({
                            "start": float(start),
                            "end": float(end),
                            "text": ""
                        })
                    except Exception as e:
                        print(f"Transcription chunk error: {e}")
                        transcripts.append({
                            "start": float(start),
                            "end": float(end),
                            "text": ""
                        })
                    start += segment_duration
            return transcripts
        except Exception as e:
            print(f"Error initializing SpeechRecognition: {e}")
            return []

    def transcribe_with_faster_whisper(self, wav_path, whisper_model="small"):
        """Transcribes the entire audio file using local faster-whisper model.
        Tries GPU (CUDA) first for speed, falls back to CPU automatically if
        no compatible GPU/cuDNN is available."""
        print(f"Running local faster-whisper ({whisper_model})...")
        try:
            from faster_whisper import WhisperModel
            try:
                model = WhisperModel(whisper_model, device="cuda", compute_type="float16")
                print("faster-whisper running on GPU (CUDA).")
            except Exception as gpu_err:
                print(f"GPU unavailable for faster-whisper ({gpu_err}), using CPU instead.")
                model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
            segments, info = model.transcribe(wav_path, beam_size=5)
            
            transcripts = []
            for s in segments:
                transcripts.append({
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": s.text.strip()
                })
            print("Local faster-whisper transcription complete!")
            return transcripts
        except Exception as e:
            print(f"faster-whisper local model error: {e}. Falling back to SpeechRecognition API.")
            return self.transcribe_audio_segments(wav_path, segment_duration=15)

    def analyze_visual_motion(self, max_frames_to_check=500):
        """
        Computes representative frame-to-frame motion scores spread evenly
        across the entire video timeline. Returns a list of dicts: {"time": t, "score": s}.
        """
        print("Analyzing visual motion peaks across full duration...")
        motion_scores = []
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return []
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30 # standard fallback
                
            if total_frames <= 0:
                cap.release()
                return []
                
            # Spread indices evenly from start to end (at most max_frames_to_check)
            if total_frames <= max_frames_to_check:
                indices = list(range(total_frames))
            else:
                indices = [int(i * (total_frames - 1) / (max_frames_to_check - 1)) for i in range(max_frames_to_check)]
                
            prev_gray = None
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (160, 90))
                
                time_sec = float(idx / fps)
                
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    mean_diff = float(np.mean(diff))
                    motion_scores.append({
                        "time": time_sec,
                        "score": mean_diff
                    })
                else:
                    motion_scores.append({
                        "time": time_sec,
                        "score": 0.0
                    })
                    
                prev_gray = gray
                
            cap.release()
            return motion_scores
        except Exception as e:
            print(f"Error in visual analysis: {e}")
            return []

    def score_highlights_via_llm(self, candidates, provider="Ollama", host="http://localhost:11434", model="llama3.2:3b", api_key=None):
        """
        Sends the top highlight candidates to local Ollama or an OpenAI-compatible endpoint
        for semantic virality analysis. Blends results and degrades gracefully.
        """
        print(f"Running LLM pass: Provider={provider}, Model={model}, Host={host}")
        
        simplified_candidates = [
            {
                "index": i,
                "start": cand["start"],
                "end": cand["end"],
                "text": cand["text"]
            }
            for i, cand in enumerate(candidates)
        ]
        
        system_prompt = (
            "You are a viral social media editor. Analyze the transcript segments of this video "
            "and assign a 'semantic_score' between 0 and 100 based on humor, hook potential, insight, "
            "and suitability for TikTok/Shorts. Provide a short 'reason' for each.\n\n"
            "Respond ONLY with a valid JSON array of objects, each containing exact keys: 'index' (int), "
            "'semantic_score' (int), and 'reason' (string). Do not output any introductory or concluding text."
        )
        
        user_content = f"Candidates JSON:\n{json.dumps(simplified_candidates, indent=2)}"
        
        try:
            if provider.lower() == "ollama":
                url = f"{host}/api/generate"
                payload = {
                    "model": model,
                    "prompt": f"{system_prompt}\n\n{user_content}",
                    "stream": False,
                    "options": {
                        "temperature": 0.2
                    }
                }
                headers = {"Content-Type": "application/json"}
                
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    resp_data = json.loads(response.read().decode("utf-8"))
                    response_text = resp_data.get("response", "").strip()
            else:
                # OpenAI-compatible endpoints (standard Chat completions format)
                url = host if host.endswith("/chat/completions") else f"{host.rstrip('/')}/chat/completions"
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    "temperature": 0.2
                }
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                    
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    resp_data = json.loads(response.read().decode("utf-8"))
                    response_text = resp_data["choices"][0]["message"]["content"].strip()
            
            # Clean response text in case LLM wraps it in markdown blocks
            json_match = re.search(r"\[\s*\{.*\}\s*\]", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
                
            scores_list = json.loads(response_text)
            scores_lookup = {item["index"]: item for item in scores_list}
            
            for i, cand in enumerate(candidates):
                if i in scores_lookup:
                    info = scores_lookup[i]
                    sem_score = float(info.get("semantic_score", cand["score"]))
                    reason = info.get("reason", "AI evaluated score")
                    
                    blended_score = round((cand["score"] * 0.4) + (sem_score * 0.6), 2)
                    cand["score"] = blended_score
                    cand["hook_title"] = f"★ {cand['hook_title']}"
                    cand["explanation"] = reason
                    cand["breakdown"]["llm_semantic_score"] = sem_score
                    print(f"  - Clip #{i+1} LLM re-score: {cand['score']}% (Reason: {reason})")
                else:
                    cand["explanation"] = "LLM ranking skipped (Index mismatch)"
                    cand["breakdown"]["llm_semantic_score"] = cand["score"]
                    
            print("LLM semantic pass completed successfully!")
            
        except Exception as e:
            print(f"LLM integration warning/offline: {e}. Gracefully falling back to heuristic-only scoring.")
            for cand in candidates:
                cand["explanation"] = f"LLM offline fallback. Heuristic rating active."
                cand["breakdown"]["llm_semantic_score"] = 0.0

    def detect_highlights(self, min_clip_duration=15, max_clip_duration=30, top_n=5, 
                          use_whisper=True, whisper_model="small", 
                          llm_provider="Off", ollama_model="llama3.2:3b", ollama_host="http://localhost:11434",
                          openai_base_url="http://localhost:20128/v1", openai_model="oc/deepseek-v4-flash-free",
                          openai_api_key=None):
        """
        Combines audio energy, motion detection, speech rate, and optional LLM analysis (Ollama or OpenAI compatible)
        to identify and rank high-virality highlights. Incorporates stage-level performance timing instrumentation.
        """
        print("Starting Upgraded Highlight Detection Pipeline...")
        t_start = time.perf_counter()
        
        # 1 & 2. Audio Extraction and Energy Profiling
        t0 = time.perf_counter()
        temp_wav = "temp_extraction.wav"
        audio_extracted = self.extract_audio(temp_wav)
        
        rms_energy = []
        if audio_extracted:
            rms_energy = self.analyze_audio_energy(temp_wav, segment_duration=1.0)
        t_audio = time.perf_counter() - t0
            
        # 3. Transcription Phase
        t1 = time.perf_counter()
        transcripts = []
        if audio_extracted:
            if use_whisper:
                transcripts = self.transcribe_with_faster_whisper(temp_wav, whisper_model=whisper_model)
            else:
                transcripts = self.transcribe_audio_segments(temp_wav, segment_duration=15)
        t_transcribe = time.perf_counter() - t1
            
        # 4. Motion Profiling (Spread over entire duration)
        t2 = time.perf_counter()
        motion_scores = self.analyze_visual_motion()
        t_motion = time.perf_counter() - t2
        
        # Clean up temp wave
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except:
                pass

        # 5. Candidate Generation & Heuristic Scoring
        t3 = time.perf_counter()
        candidates = []
        step = 10
        window_size = 20
        
        virality_words = {
            "wow", "crazy", "unbelievable", "amazing", "shocker", "funny", "secret",
            "look", "insane", "hacks", "omg", "lol", "wait", "epic", "huge", "never",
            "always", "stop", "mind-blowing", "important", "alert", "danger", "dead",
            "laugh", "joke", "funny", "fail", "win", "scary", "exposed", "truth", "hack"
        }
        
        video_dur = self.duration if self.duration > 0 else 60
        
        for start in range(0, int(video_dur) - window_size, step):
            end = start + window_size
            
            # (a) Audio Energy Score (0 to 100)
            audio_sub = rms_energy[start:end] if len(rms_energy) >= end else []
            if audio_sub:
                avg_rms = float(np.mean(audio_sub))
                peak_rms = float(np.max(audio_sub))
                audio_score = min((avg_rms * 40 + peak_rms * 60) * 100, 100)
            else:
                audio_score = 50
                
            # (b) Lexical & Speech Score (0 to 100)
            segment_text = ""
            speech_rate = 0
            lexical_score = 0
            
            # Find transcripts that overlap with this window
            overlapping_texts = []
            for t in transcripts:
                if (t["start"] < end) and (t["end"] > start):
                    overlapping_texts.append(t["text"])
            
            segment_text = " ".join([t for t in overlapping_texts if t]).strip()
            
            if segment_text:
                words = segment_text.lower().split()
                speech_rate = len(words) / window_size
                density_score = min((speech_rate / 3.0) * 100, 100)
                
                matched_words = [w for w in words if w in virality_words]
                word_score = min(len(matched_words) * 20, 100)
                lexical_score = (density_score * 0.4) + (word_score * 0.6)
            else:
                lexical_score = 40
                
            # (c) Visual Motion Score (0 to 100) - Mapped to Timeline dict
            if motion_scores:
                motion_sub = [item["score"] for item in motion_scores if start <= item["time"] <= end]
                if motion_sub:
                    motion_score = min(float(np.mean(motion_sub)) * 10, 100)
                else:
                    motion_score = 50
            else:
                motion_score = 50
                
            # (d) Hook Quality (0 to 100)
            hook_sub = rms_energy[start:start+3] if len(rms_energy) >= start+3 else []
            hook_score = min(float(np.max(hook_sub)) * 150, 100) if hook_sub else 50
            
            final_score = (
                (audio_score * 0.35) + 
                (lexical_score * 0.40) + 
                (motion_score * 0.15) + 
                (hook_score * 0.10)
            )
            
            final_score = round(float(final_score), 2)
            
            words_list = segment_text.split()
            if len(words_list) >= 4:
                hook_title = " ".join(words_list[:4]) + "..."
            else:
                hook_title = f"Clip Highlight at {start}s"
                
            candidates.append({
                "start": start,
                "end": end,
                "score": final_score,
                "text": segment_text if segment_text else "[No dialogue detected]",
                "hook_title": hook_title,
                "explanation": "Heuristic scoring mode active.",
                "breakdown": {
                    "audio_energy": round(audio_score, 1),
                    "lexical_virality": round(lexical_score, 1),
                    "visual_motion": round(motion_score, 1),
                    "hook_strength": round(hook_score, 1)
                }
            })
            
        # 6. Deduplicate candidates
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        unique_highlights = []
        
        for cand in candidates:
            overlap = False
            for accepted in unique_highlights:
                intersect_start = max(cand["start"], accepted["start"])
                intersect_end = min(cand["end"], accepted["end"])
                if intersect_end > intersect_start:
                    overlap_dur = intersect_end - intersect_start
                    if overlap_dur / window_size > 0.30:
                        overlap = True
                        break
            if not overlap:
                unique_highlights.append(cand)
                
        top_candidates = unique_highlights[:top_n]
        t_heuristic = time.perf_counter() - t3
        
        # 7. Apply optional LLM semantic re-ranking pass on top candidates
        t_llm_start = time.perf_counter()
        if llm_provider != "Off" and len(top_candidates) > 0:
            if llm_provider == "Ollama":
                self.score_highlights_via_llm(
                    top_candidates, 
                    provider="Ollama", 
                    host=ollama_host, 
                    model=ollama_model
                )
            elif llm_provider == "OpenAI":
                self.score_highlights_via_llm(
                    top_candidates, 
                    provider="OpenAI", 
                    host=openai_base_url, 
                    model=openai_model, 
                    api_key=openai_api_key
                )
        t_llm = time.perf_counter() - t_llm_start
            
        # Sort again since score was updated by LLM
        top_candidates = sorted(top_candidates, key=lambda x: x["score"], reverse=True)
        
        t_total = time.perf_counter() - t_start
        
        # Cache full timings in engine instance
        self.last_timing = {
            "audio_extract": round(t_audio, 2),
            "transcribe": round(t_transcribe, 2),
            "motion": round(t_motion, 2),
            "heuristic_scoring": round(t_heuristic, 2),
            "llm_rerank": round(t_llm, 2),
            "total": round(t_total, 2)
        }
        
        print(f"Timing: audio_extract={t_audio:.1f}s transcribe={t_transcribe:.1f}s motion={t_motion:.1f}s llm_rerank={t_llm:.1f}s total={t_total:.1f}s")
        
        return top_candidates
