import os
import cv2
import pytest
import numpy as np
from src.video_cropper import VideoCropper
from moviepy import VideoFileClip

def test_haar_cascade_loaded_successfully():
    """Verify that VideoCropper initializes correctly and its cascade is not empty."""
    cropper = VideoCropper()
    assert cropper.face_cascade is not None
    assert not cropper.face_cascade.empty()

def create_dummy_video(filename, duration_sec=2, fps=24, width=640, height=360):
    """Creates a short 2-second landscape video with actual moving graphics and sinus audio."""
    # Write temporary silent video track
    temp_silent = "temp_test_silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_silent, fourcc, fps, (width, height))
    
    for f in range(duration_sec * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Add grid lines
        for x in range(0, width, 40):
            cv2.line(frame, (x, 0), (x, height), (50, 50, 50), 1)
        # Draw a moving white circle (simulating a face)
        cx = int(width / 2 + np.sin(f / 10) * 100)
        cy = int(height / 2)
        cv2.circle(frame, (cx, cy), 40, (255, 255, 255), -1)
        out.write(frame)
    out.release()
    
    # Generate simple test voiceover using gTTS
    from gtts import gTTS
    temp_audio = "temp_test_audio.mp3"
    tts = gTTS(text="This is a test speech clip.", lang="en")
    tts.save(temp_audio)
    
    # Combine them using MoviePy
    with VideoFileClip(temp_silent) as video_clip:
        from moviepy import AudioFileClip
        with AudioFileClip(temp_audio) as audio_clip:
            trimmed_audio = audio_clip.subclipped(0, duration_sec)
            final_clip = video_clip.with_audio(trimmed_audio)
            final_clip.write_videofile(
                filename,
                codec="libx264",
                audio_codec="aac",
                logger=None
            )
            
    # Cleanup intermediate files
    for temp_f in [temp_silent, temp_audio]:
        if os.path.exists(temp_f):
            os.remove(temp_f)

def test_crop_vertical_integration_and_subclipped_smoke():
    """
    Run an integration test against a programmatically generated source video.
    Asserts the output 9:16 vertical video exists and retains a non-zero audio track.
    Also validates that MoviePy's .subclipped() (MoviePy 2.x standard) is utilized.
    """
    input_video = "test_source_input.mp4"
    output_video = "test_output_vertical.mp4"
    
    try:
        # Create 2-second test video
        create_dummy_video(input_video, duration_sec=2)
        assert os.path.exists(input_video)
        
        # Instantiate and run vertical crop with face-tracking
        cropper = VideoCropper()
        success = cropper.crop_to_vertical(
            input_video_path=input_video,
            output_video_path=output_video,
            start_time=0.0,
            end_time=2.0,
            track_faces=True
        )
        
        assert success
        assert os.path.exists(output_video)
        
        # Verify video specifications
        with VideoFileClip(output_video) as clip:
            assert clip.duration > 0
            assert clip.size[0] < clip.size[1] # Height is larger than width (9:16 vertical!)
            assert clip.audio is not None
            assert clip.audio.duration > 0
            
    finally:
        # Cleanup test clips
        for f in [input_video, output_video]:
            if os.path.exists(f):
                os.remove(f)

def test_no_deprecated_subclip_usage():
    """Verify statically that the code uses MoviePy 2.x's .subclipped() instead of deprecated .subclip()."""
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Ensure we don't have .subclip( other than .subclipped(
                    # Look for `.subclip(` specifically
                    assert ".subclip(" not in content, f"Deprecated .subclip() found in {path}! Use .subclipped() instead."
