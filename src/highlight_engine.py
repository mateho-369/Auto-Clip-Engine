import os
import wave
import numpy as np
import speech_recognition as sr
from moviepy import VideoFileClip, AudioFileClip
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
                    rms_values.append(rms)
                    
                return rms_values
        except Exception as e:
            print(f"Error analyzing audio energy: {e}")
            return []

    def transcribe_audio_segments(self, wav_path, segment_duration=10):
        """Transcribes the audio in chunks using SpeechRecognition."""
        if not os.path.exists(wav_path):
            return []
            
        recognizer = sr.Recognizer()
        transcripts = []
        
        try:
            with sr.AudioFile(wav_path) as source:
                total_duration = source.duration
                # NOTE: record()'s `offset` skips forward from the *current* stream
                # position, not from the start of the file — passing offset=start on
                # every loop iteration compounds and desyncs each chunk from its
                # labelled start/end. Instead, read sequentially with duration only,
                # since AudioFile keeps its own read cursor between record() calls.
                start = 0
                while start < total_duration:
                    end = min(start + segment_duration, total_duration)
                    try:
                        audio_chunk = recognizer.record(source, duration=segment_duration)
                        text = recognizer.recognize_google(audio_chunk, language="en-US")
                        transcripts.append({
                            "start": start,
                            "end": end,
                            "text": text
                        })
                    except sr.UnknownValueError:
                        # Speech was unintelligible
                        transcripts.append({
                            "start": start,
                            "end": end,
                            "text": ""
                        })
                    except sr.RequestError as e:
                        print(f"Google speech API request failed: {e}")
                        # Network issue or quota, append empty string
                        transcripts.append({
                            "start": start,
                            "end": end,
                            "text": ""
                        })
                    except Exception as e:
                        print(f"Transcription chunk error: {e}")
                        transcripts.append({
                            "start": start,
                            "end": end,
                            "text": ""
                        })
                    start += segment_duration
            return transcripts
        except Exception as e:
            print(f"Error initializing SpeechRecognition: {e}")
            return []

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
                motion_scores.append(float(mean_diff))
                
                prev_gray = gray
                frame_count += 1
                
            cap.release()
            return motion_scores
        except Exception as e:
            print(f"Error in visual analysis: {e}")
            return []

    def detect_highlights(self, min_clip_duration=15, max_clip_duration=30, top_n=5):
        """
        Combines audio, lexical, and motion signals to detect viral highlights.
        Returns a sorted list of dictionaries representing ranked highlight candidates.
        """
        print("Starting Highlight Detection Pipeline...")
        
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
            print("Running Speech Recognition...")
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
        # We search with sliding window of length 20 seconds, step 10 seconds
        candidates = []
        step = 10
        window_size = 20
        
        # Virality keywords for lexical scoring
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
                avg_rms = np.mean(audio_sub)
                peak_rms = np.max(audio_sub)
                # Combined metric of average level and dramatic peaks
                audio_score = min((avg_rms * 40 + peak_rms * 60) * 100, 100)
            else:
                audio_score = 50 # Fallback
                
            # (b) Lexical & Speech Score (0 to 100)
            segment_text = ""
            speech_rate = 0 # Words per second
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
                # Score based on speech density (faster speech is usually more energetic)
                density_score = min((speech_rate / 3.0) * 100, 100) # 3 words/sec is high density
                
                # Check for virality words
                matched_words = [w for w in words if w in virality_words]
                word_score = min(len(matched_words) * 20, 100)
                lexical_score = (density_score * 0.4) + (word_score * 0.6)
            else:
                lexical_score = 40 # Standard conversational fallback score
                
            # (c) Visual Motion Score (0 to 100)
            # Map start/end to frame-based motion list
            if motion_scores:
                motion_sub = motion_scores[start:end] if len(motion_scores) >= end else []
                if motion_sub:
                    motion_score = min(np.mean(motion_sub) * 10, 100)
                else:
                    motion_score = 50
            else:
                motion_score = 50
                
            # (d) Combined Weighted Virality Score
            # Audio (35%), Speech & Lexical (40%), Motion (15%), Hook Quality (10%)
            # Hook Quality is based on whether it starts with high audio energy
            hook_sub = rms_energy[start:start+3] if len(rms_energy) >= start+3 else []
            hook_score = min(np.max(hook_sub) * 150, 100) if hook_sub else 50
            
            final_score = (
                (audio_score * 0.35) + 
                (lexical_score * 0.40) + 
                (motion_score * 0.15) + 
                (hook_score * 0.10)
            )
            
            # Ensure it's bounded [0, 100] and clean
            final_score = round(float(final_score), 2)
            
            # Hook description/summary
            # Select first few words as potential title/hook text
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
                "breakdown": {
                    "audio_energy": round(audio_score, 1),
                    "lexical_virality": round(lexical_score, 1),
                    "visual_motion": round(motion_score, 1),
                    "hook_strength": round(hook_score, 1)
                }
            })
            
        # 6. Deduplicate candidates (remove highly overlapping windows)
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        unique_highlights = []
        
        for cand in candidates:
            # Check if this candidate overlaps too much with already accepted highlights
            overlap = False
            for accepted in unique_highlights:
                # Calculate intersection of intervals [start, end]
                intersect_start = max(cand["start"], accepted["start"])
                intersect_end = min(cand["end"], accepted["end"])
                if intersect_end > intersect_start:
                    # Overlap duration
                    overlap_dur = intersect_end - intersect_start
                    # If overlap is more than 30% of the candidate's duration
                    if overlap_dur / window_size > 0.30:
                        overlap = True
                        break
            if not overlap:
                unique_highlights.append(cand)
                
        # Return top N
        return unique_highlights[:top_n]
