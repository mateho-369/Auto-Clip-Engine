import os
from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

class VoiceoverEngine:
    def __init__(self):
        pass
        
    def generate_voiceover_mp3(self, text, output_path, lang='en', tld='com'):
        """Generates a high-quality narrator voiceover MP3 using gTTS."""
        print(f"Generating voiceover for text: '{text}'...")
        try:
            # We can use different TLDs for accents (e.g. co.uk for British, com for American, etc.)
            tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
            tts.save(output_path)
            print(f"Voiceover saved to {output_path}!")
            return True
        except Exception as e:
            print(f"Error generating voiceover: {e}")
            return False

    def overlay_voiceover_on_video(self, video_path, voiceover_text, output_video_path, duck_ratio=0.3):
        """
        Generates a narration clip, overlays it at the start of the video,
        and automatically ducks the original background audio while the narrator speaks.
        """
        temp_voice_path = "temp_voiceover.mp3"
        success = self.generate_voiceover_mp3(voiceover_text, temp_voice_path)
        if not success or not os.path.exists(temp_voice_path):
            print("Failed to generate voiceover. Exporting raw clip instead.")
            # Fallback to copy
            try:
                os.rename(video_path, output_video_path)
                return True
            except:
                return False
                
        try:
            print("Overlaying voiceover and running audio ducking...")
            with VideoFileClip(video_path) as video:
                # Load voiceover audio
                voiceover_clip = AudioFileClip(temp_voice_path)
                voice_dur = voiceover_clip.duration
                
                # We split the original audio into two parts:
                # 1. During the voiceover (ducked down to 30% volume)
                # 2. After the voiceover (original 100% volume)
                orig_audio = video.audio
                
                if orig_audio is not None:
                    # Apply volume ducking during voiceover duration
                    ducked_audio_part = orig_audio.subclip(0, min(voice_dur, video.duration)).multiply_volume(duck_ratio)
                    
                    if video.duration > voice_dur:
                        normal_audio_part = orig_audio.subclip(voice_dur, video.duration)
                        # Composite ducked background audio with the voiceover track
                        intro_audio = CompositeAudioClip([ducked_audio_part, voiceover_clip])
                        
                        # Concatenate/compose full audio stream
                        # MoviePy composite audio can position tracks relative to each other
                        voiceover_clip_positioned = voiceover_clip.with_start(0)
                        ducked_bg_positioned = orig_audio.multiply_volume(duck_ratio).subclip(0, min(voice_dur, video.duration))
                        
                        mixed_audio_intro = CompositeAudioClip([ducked_bg_positioned, voiceover_clip_positioned])
                        
                        # We can construct the final audio by concatenating or using CompositeAudioClip
                        # A clean way is composite audio clip:
                        # voiceover plays from 0 to voice_dur.
                        # background is ducked from 0 to voice_dur, and then full volume from voice_dur onwards.
                        # In MoviePy, we can schedule when each track starts.
                        full_bg_ducked = orig_audio.multiply_volume(duck_ratio).subclip(0, min(voice_dur, video.duration))
                        full_bg_normal = orig_audio.subclip(min(voice_dur, video.duration), video.duration).with_start(min(voice_dur, video.duration))
                        voice_track_positioned = voiceover_clip.with_start(0)
                        
                        final_audio = CompositeAudioClip([full_bg_ducked, full_bg_normal, voice_track_positioned])
                    else:
                        final_audio = CompositeAudioClip([ducked_audio_part, voiceover_clip])
                else:
                    final_audio = voiceover_clip
                    
                # Bind the new composite audio back to the video
                final_video = video.with_audio(final_audio)
                final_video.write_videofile(
                    output_video_path,
                    codec="libx264",
                    audio_codec="aac",
                    logger=None
                )
                
            # Clean up temp mp3
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
