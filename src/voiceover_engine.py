import os
from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
from src.utils import write_video_safely

class VoiceoverEngine:
    def __init__(self, kokoro_model_path="kokoro-v0_19.onnx", kokoro_voices_path="voices.bin"):
        self.kokoro_model_path = kokoro_model_path
        self.kokoro_voices_path = kokoro_voices_path
        self.kokoro_available = False
        
        # Check if Kokoro ONNX model and voices are available locally
        if os.path.exists(kokoro_model_path) and os.path.exists(kokoro_voices_path):
            try:
                from kokoro_onnx import Kokoro
                # Try initializing to verify no syntax or binding errors
                self.kokoro_instance = Kokoro(kokoro_model_path, kokoro_voices_path)
                self.kokoro_available = True
                print("Local Kokoro-82M TTS engine initialized successfully!")
            except Exception as e:
                print(f"Failed to initialize Kokoro-82M engine: {e}. Using gTTS.")
        else:
            print("Kokoro model files not found. Using gTTS as fallback engine.")

    def generate_voiceover_kokoro(self, text, output_path, voice="af_bella"):
        """Generates premium local voiceover track using Kokoro ONNX model."""
        print(f"Generating premium Kokoro voiceover for: '{text}'...")
        try:
            import soundfile as sf
            samples, sample_rate = self.kokoro_instance.create(
                text, voice=voice, speed=1.0, lang="en-us"
            )
            sf.write(output_path, samples, sample_rate)
            print(f"Kokoro voiceover saved to {output_path}!")
            return True
        except Exception as e:
            print(f"Kokoro-82M generation error: {e}. Falling back to gTTS.")
            return False

    def generate_voiceover_mp3(self, text, output_path, lang='en', tld='com', use_kokoro=True, kokoro_voice="af_bella"):
        """Generates voiceover, utilizing Kokoro if requested/available, otherwise falling back to gTTS."""
        if use_kokoro and self.kokoro_available:
            success = self.generate_voiceover_kokoro(text, output_path, voice=kokoro_voice)
            if success:
                return True
                
        # Fallback to gTTS
        print(f"Generating gTTS fallback voiceover...")
        try:
            tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
            tts.save(output_path)
            print(f"Voiceover saved to {output_path}!")
            return True
        except Exception as e:
            print(f"gTTS fallback error: {e}")
            return False

    def overlay_voiceover_on_video(self, video_path, voiceover_text, output_video_path, duck_ratio=0.25, use_kokoro=True, kokoro_voice="af_bella"):
        """
        Generates a narration clip, overlays it at the start of the video,
        and automatically ducks the original background audio while the narrator speaks.
        """
        temp_voice_path = "temp_voiceover.wav" if (use_kokoro and self.kokoro_available) else "temp_voiceover.mp3"
        
        success = self.generate_voiceover_mp3(
            voiceover_text, temp_voice_path, use_kokoro=use_kokoro, kokoro_voice=kokoro_voice
        )
        if not success or not os.path.exists(temp_voice_path):
            print("Failed to generate voiceover. Exporting raw clip instead.")
            try:
                # Fallback: copy video file
                import shutil
                shutil.copy(video_path, output_video_path)
                return True
            except:
                return False
                
        try:
            print("Overlaying voiceover and running audio ducking...")
            with VideoFileClip(video_path) as video:
                # Load voiceover audio
                voiceover_clip = AudioFileClip(temp_voice_path)
                voice_dur = voiceover_clip.duration
                
                orig_audio = video.audio
                if orig_audio is not None:
                    ducked_end = min(voice_dur, video.duration)
                    # Background is ducked while narrator speaks, then back to normal volume after
                    full_bg_ducked = orig_audio.subclipped(0, ducked_end).multiply_volume(duck_ratio)
                    voice_track_positioned = voiceover_clip.with_start(0)

                    if video.duration > voice_dur:
                        full_bg_normal = orig_audio.subclipped(ducked_end, video.duration).with_start(ducked_end)
                        final_audio = CompositeAudioClip([full_bg_ducked, full_bg_normal, voice_track_positioned])
                    else:
                        final_audio = CompositeAudioClip([full_bg_ducked, voice_track_positioned])
                else:
                    final_audio = voiceover_clip
                    
                # Bind the new composite audio back to the video
                final_video = video.with_audio(final_audio)
                write_video_safely(final_video, output_video_path, audio_codec="aac")
                
            # Clean up temp files
            if os.path.exists(temp_voice_path):
                os.remove(temp_voice_path)
            print("Voiceover overlay complete with auto-ducking!")
            return True
        except Exception as e:
            print(f"Error mixing audio track with voiceover: {e}")
            if os.path.exists(temp_voice_path):
                try:
                    os.remove(temp_voice_path)
                except:
                    pass
            return False
