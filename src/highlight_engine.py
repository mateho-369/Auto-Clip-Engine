import os
import wave
import json
import re
import urllib.request
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

    def transcribe_with_faster_whisper(self, wav_path, whisper_model="tiny"):
        """Transcribes the entire audio file using local faster-whisper model."""
        print(f"Running local faster-whisper ({whisper_model})...")
        try:
            from faster_whisper import WhisperModel
            # Initialize model on CPU
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
        """Computes average frame-to-frame motion score of the video."""
        print("Analyzing visual motion peaks...")
        motion_scores = []
        try:
            cap = cv2.VideoCapture(self.video_path)
            ret, prev_frame = cap.read()
            if not ret:
                cap.release()
                return []
                
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            # Resize for faster processing
            prev_gray = cv2.resize(prev_gray, (160, 90))
            
            frame_count = 0
            while cap.isOpened() and frame_count < max_frames_to_check:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (160, 90))
                
                # Compute absolute difference between frames
                diff = cv2.absdiff(gray, prev_gray)
                mean_diff = np.mean(diff)
                motion_scores.append(float(mean_diff)) # Cast to float
                
                prev_gray = gray
                frame_count += 1
                
            cap.release()
            return motion_scores
        except Exception as e:
            print(f"Error in visual analysis: {e}")
            return []

    def score_highlights_via_ollama(self, candidates, ollama_host="http://localhost:11434", ollama_model="llama3.2:3b"):
        """
        Sends the top highlight candidates to a local Ollama instance for semantic virality analysis.
        Integrates seamlessly and degrades gracefully if Ollama is offline.
        """
        print(f"Connecting to Ollama model '{ollama_model}' on {ollama_host}...")
        
        # Prepare the candidates to be sent to Ollama
        simplified_candidates = [
            {
                "index": i,
                "start": cand["start"],
                "end": cand["end"],
                "text": cand["text"]
            }
            for i, cand in enumerate(candidates)
        ]
        
        # Craft prompt requesting strict JSON response
        prompt = (
            "You are a viral social media editor. Analyze the transcript segments of this video "
            "and assign a 'semantic_score' between 0 and 100 based on humor, hook potential, insight, "
            "and suitability for TikTok/Shorts. Provide a short 'reason' for each.\n\n"
            f"Candidates JSON:\n{json.dumps(simplified_candidates, indent=2)}\n\n"
            "Respond ONLY with a valid JSON array of objects, each containing exact keys: 'index' (int), "
            "'semantic_score' (int), and 'reason' (string). Do not output any introductory or concluding text."
        )
        
        url = f"{ollama_host}/api/generate"
        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                response_text = resp_data.get("response", "").strip()
                
                # Attempt to extract JSON array out of response text in case model added markdown wrapping
                json_match = re.search(r"\[\s*\{.*\}\s*\]", response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
                    
                ollama_scores = json.loads(response_text)
                
                # Build lookup
                scores_lookup = {item["index"]: item for item in ollama_scores}
                
                # Blend the local heuristic score with the LLM semantic score (40% heuristic, 60% LLM semantic)
                for i, cand in enumerate(candidates):
                    if i in scores_lookup:
                        semantic_info = scores_lookup[i]
                        sem_score = float(semantic_info.get("semantic_score", cand["score"]))
                        reason = semantic_info.get("reason", "Ollama compiled rating")
                        
                        # Blended scoring
                        blended_score = round((cand["score"] * 0.4) + (sem_score * 0.6), 2)
                        cand["score"] = blended_score
                        cand["hook_title"] = f"★ {cand['hook_title']}"
                        cand["explanation"] = reason
                        cand["breakdown"]["llm_semantic_score"] = sem_score
                        print(f"  - Clip #{i+1} Ollama re-score: {cand['score']}% (Reason: {reason})")
                    else:
                        cand["explanation"] = "Ollama ranking skipped (Index mismatch)"
                        cand["breakdown"]["llm_semantic_score"] = cand["score"]
                        
            print("Ollama semantic pass completed successfully!")
        except Exception as e:
            print(f"Ollama integration warning/offline: {e}. Gracefully falling back to heuristic-only scoring.")
            for cand in candidates:
                cand["explanation"] = "Local Ollama offline. Heuristic fallback active."
                cand["breakdown"]["llm_semantic_score"] = 0.0

    def detect_highlights(self, min_clip_duration=15, max_clip_duration=30, top_n=5, 
                          use_whisper=True, whisper_model="tiny", 
                          use_ollama=False, ollama_model="llama3.2:3b", ollama_host="http://localhost:11434"):
        """
        Combines audio energy, motion detection, speech rate, and optional local LLM analysis (Ollama)
        to identify and rank high-virality highlights.
        """
        print("Starting Local-Upgraded Highlight Detection Pipeline...")
        
        # 1. Extract audio
        temp_wav = "temp_extraction.wav"
        audio_extracted = self.extract_audio(temp_wav)
        
        # 2. Get audio energy
        rms_energy = []
        if audio_extracted:
            rms_energy = self.analyze_audio_energy(temp_wav, segment_duration=1.0)
            
        # 3. Get speech transcripts
        transcripts = []
        if audio_extracted:
            if use_whisper:
                transcripts = self.transcribe_with_faster_whisper(temp_wav, whisper_model=whisper_model)
            else:
                transcripts = self.transcribe_audio_segments(temp_wav, segment_duration=15)
            
        # 4. Get motion scores
        motion_scores = self.analyze_visual_motion()
        
        # Clean up temp wave
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except:
                pass

        # 5. Generate candidates and calculate scores
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
                
            # (c) Visual Motion Score (0 to 100)
            if motion_scores:
                motion_sub = motion_scores[start:end] if len(motion_scores) >= end else []
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
        
        # 7. Apply optional local Ollama semantic re-ranking pass on top candidates
        if use_ollama and len(top_candidates) > 0:
            self.score_highlights_via_ollama(top_candidates, ollama_host=ollama_host, ollama_model=ollama_model)
            
        # Sort again since score was updated by Ollama
        top_candidates = sorted(top_candidates, key=lambda x: x["score"], reverse=True)
        
        return top_candidates
