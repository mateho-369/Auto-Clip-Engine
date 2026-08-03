import cv2
import os
import math
import numpy as np
from src.utils import write_video_safely
from moviepy import VideoFileClip

class CaptionGenerator:
    def __init__(self):
        pass
        
    def generate_srt(self, words_with_timing, output_srt_path):
        """Generates a standard clean .srt file from a list of words with timing."""
        try:
            # Group words into 3-4 word phrases for readable SRT subtitles
            phrases = []
            current_phrase_words = []
            words_per_phrase = 3
            
            for i, item in enumerate(words_with_timing):
                current_phrase_words.append(item)
                if len(current_phrase_words) >= words_per_phrase or i == len(words_with_timing) - 1:
                    start_time = current_phrase_words[0]["start"]
                    end_time = current_phrase_words[-1]["end"]
                    text = " ".join([w["word"] for w in current_phrase_words])
                    phrases.append({
                        "start": start_time,
                        "end": end_time,
                        "text": text
                    })
                    current_phrase_words = []
                    
            # Write SRT file
            with open(output_srt_path, "w", encoding="utf-8") as f:
                for idx, phrase in enumerate(phrases, 1):
                    # Format timing: HH:MM:SS,mmm
                    def format_time(seconds):
                        hours = int(seconds // 3600)
                        minutes = int((seconds % 3600) // 60)
                        secs = int(seconds % 60)
                        millis = int((seconds % 1) * 1000)
                        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
                        
                    f.write(f"{idx}\n")
                    f.write(f"{format_time(phrase['start'])} --> {format_time(phrase['end'])}\n")
                    f.write(f"{phrase['text']}\n\n")
            print(f"SRT file saved successfully to {output_srt_path}!")
            return True
        except Exception as e:
            print(f"Error generating SRT: {e}")
            return False

    def estimate_word_timings(self, text, start_clip, end_clip):
        """
        Splits a text paragraph into words and estimates their start/end timestamps
        proportionately over the clip duration based on word length.
        """
        words = text.strip().split()
        if not words:
            return []
            
        total_chars = sum(len(w) for w in words)
        clip_duration = end_clip - start_clip
        
        words_timing = []
        current_time = 0.0 # relative to clip start
        
        for w in words:
            # Word duration is proportional to its characters
            word_weight = len(w) / total_chars if total_chars > 0 else (1.0 / len(words))
            word_dur = clip_duration * word_weight
            
            # Bound word duration to reasonable range (0.15s to 1.5s)
            word_dur = max(0.15, min(word_dur, 1.5))
            
            words_timing.append({
                "word": w,
                "start": current_time,
                "end": current_time + word_dur
            })
            current_time += word_dur
            
        # Rescale timings so they fit exactly within the clip duration
        last_end = words_timing[-1]["end"] if words_timing else 1.0
        scale = clip_duration / last_end if last_end > 0 else 1.0
        
        for item in words_timing:
            item["start"] = round(item["start"] * scale, 3)
            item["end"] = round(item["end"] * scale, 3)
            
        return words_timing

    def burn_captions_opencv(self, input_video_path, output_video_path, words_timing, style_type="Reels"):
        """
        Burns animated karaoke-style word captions onto a video frame-by-frame
        using OpenCV. Creates a highly professional look with black drop-shadows,
        clean bold typography, and bright active-word color highlights.
        """
        print(f"Burning captions on {input_video_path} with style {style_type}...")
        
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print("Error: Could not open cropped video track.")
            return False
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        temp_silent_output = "temp_caption_silent.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_silent_output, fourcc, fps, (width, height))
        
        # We process frame by frame
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time_sec = frame_idx / fps
            
            # Determine which word is currently active
            active_word_idx = -1
            for idx, w_time in enumerate(words_timing):
                if w_time["start"] <= current_time_sec <= w_time["end"]:
                    active_word_idx = idx
                    break
                    
            # If no active word found, find the closest upcoming one
            if active_word_idx == -1 and len(words_timing) > 0:
                for idx, w_time in enumerate(words_timing):
                    if current_time_sec < w_time["start"]:
                        active_word_idx = idx
                        break
                if active_word_idx == -1: # after the last word
                    active_word_idx = len(words_timing) - 1
                    
            # Prepare the phrase to render
            # We display a 3-word sliding window: previous word, active word, and next word
            if active_word_idx != -1 and len(words_timing) > 0:
                start_w = max(0, active_word_idx - 1)
                end_w = min(len(words_timing), active_word_idx + 2)
                
                phrase_words = words_timing[start_w:end_w]
                
                # Render the phrase centered
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.1 if style_type == "Reels" else 1.3
                thickness = 3
                shadow_thickness = 7
                
                # Base Y position (around lower 70% of the screen)
                y_pos = int(height * 0.70)
                
                # We calculate horizontal spacing for each word in the phrase
                # to color-code the active word yellow and others white.
                word_sizes = []
                for w_item in phrase_words:
                    size = cv2.getTextSize(w_item["word"], font, font_scale, thickness)[0]
                    word_sizes.append(size)
                    
                total_phrase_width = sum(size[0] for size in word_sizes) + (len(phrase_words) - 1) * 15 # with padding
                
                # Starting X position to center the entire block
                start_x = int((width - total_phrase_width) / 2)
                
                # Draw the words
                current_x = start_x
                for idx, w_item in enumerate(phrase_words):
                    word_text = w_item["word"].upper() if style_type == "Shorts" else w_item["word"]
                    is_active = (start_w + idx == active_word_idx)
                    
                    # Colors: Active is bright yellow/green, inactive is pure white
                    text_color = (0, 255, 255) if is_active else (255, 255, 255) # BGR (Yellow vs White)
                    if style_type == "Tech":
                        text_color = (0, 200, 0) if is_active else (240, 240, 240) # Green for active tech-style
                        
                    shadow_color = (0, 0, 0)
                    
                    # Render shadow (slightly thicker, drawn underneath)
                    cv2.putText(frame, word_text, (current_x + 2, y_pos + 2), font, font_scale, shadow_color, shadow_thickness, cv2.LINE_AA)
                    # Render foreground text
                    cv2.putText(frame, word_text, (current_x, y_pos), font, font_scale, text_color, thickness, cv2.LINE_AA)
                    
                    current_x += word_sizes[idx][0] + 15
                    
            out.write(frame)
            frame_idx += 1
            
        cap.release()
        out.release()
        
        # Merge audio back
        try:
            print("Merging audio back with captioned video...")
            with VideoFileClip(input_video_path) as orig_cropped_clip:
                audio = orig_cropped_clip.audio
                with VideoFileClip(temp_silent_output) as captioned_silent:
                    final_clip = captioned_silent.with_audio(audio)
                    write_video_safely(final_clip, output_video_path, audio_codec="aac")
            if os.path.exists(temp_silent_output):
                os.remove(temp_silent_output)
            print("Captioned vertical video successfully generated!")
            return True
        except Exception as e:
            print(f"Error merging audio with captions: {e}")
            if os.path.exists(temp_silent_output):
                try:
                    os.rename(temp_silent_output, output_video_path)
                    return True
                except:
                    pass
            return False
