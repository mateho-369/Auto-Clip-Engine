import cv2
import os
import numpy as np
from moviepy import VideoFileClip, AudioFileClip

class VideoCropper:
    def __init__(self, face_cascade_path=None):
        if face_cascade_path is None:
            # Use OpenCV's built-in cascade path
            self.face_cascade_path = os.path.join(
                cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'
            )
        else:
            self.face_cascade_path = face_cascade_path
            
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
        if self.face_cascade.empty():
            print("Warning: Could not load Haar cascade face detector. Defaulting to center-crop.")
            
    def crop_to_vertical(self, input_video_path, output_video_path, start_time, end_time, track_faces=True):
        """
        Crops a landscape video (16:9) to a vertical format (9:16) from start_time to end_time,
        with optional smooth face-tracking.
        """
        print(f"Cropping video from {start_time}s to {end_time}s (Face tracking: {track_faces})...")
        
        # Open source video to inspect dimensions and properties
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open source video {input_video_path}")
            return False
            
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate target vertical aspect ratio crop (9:16)
        # Height is same as original height, width is height * 9 / 16
        target_height = orig_height
        target_width = int(target_height * (9 / 16))
        
        # Target width must be even for standard video encoders
        if target_width % 2 != 0:
            target_width += 1
            
        if target_width > orig_width:
            # If source is already vertical or tall, make target_width original width
            target_width = orig_width
            target_height = int(target_width * (16 / 9))
            if target_height % 2 != 0:
                target_height += 1
                
        # Set start frame and end frame
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # VideoWriter configuration for the temporary silent video
        temp_silent_path = "temp_cropped_silent.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # use mp4v for high compatibility
        out = cv2.VideoWriter(temp_silent_path, fourcc, fps, (target_width, target_height))
        
        # Face tracking parameters
        # smooth_center_x starts at the exact middle of the screen
        smooth_center_x = orig_width / 2.0
        alpha = 0.08  # Tracking speed coefficient (smaller = smoother, larger = snappier)
        drift_back_rate = 0.03  # Speed at which camera returns to center when face is lost
        
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_center_x = orig_width / 2.0
            target_x = frame_center_x
            
            if track_faces and not self.face_cascade.empty():
                # Face detection is faster and more robust on smaller, grayscale frames
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Resize gray for detection to speed up the loop
                scale_f = 0.5
                gray_small = cv2.resize(gray, (0, 0), fx=scale_f, fy=scale_f)
                
                faces = self.face_cascade.detectMultiScale(
                    gray_small, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                if len(faces) > 0:
                    # Select the largest face detected (based on bounding box area)
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    fx, fy, fw, fh = largest_face
                    
                    # Convert coordinates back to original size
                    fx = int(fx / scale_f)
                    fw = int(fw / scale_f)
                    
                    # Calculate center of the face
                    target_x = fx + (fw / 2.0)
                    # Exponential Moving Average for buttery smooth movement
                    smooth_center_x = (alpha * target_x) + ((1 - alpha) * smooth_center_x)
                else:
                    # No face detected: drift back to absolute center
                    smooth_center_x = (drift_back_rate * frame_center_x) + ((1 - drift_back_rate) * smooth_center_x)
            else:
                # No face tracking: stay locked at the center
                smooth_center_x = frame_center_x
                
            # Compute cropping window bounds
            crop_x_left = int(smooth_center_x - (target_width / 2.0))
            
            # Clamp crop window to stay within original video frame boundaries
            if crop_x_left < 0:
                crop_x_left = 0
            elif crop_x_left + target_width > orig_width:
                crop_x_left = orig_width - target_width
                
            crop_y_top = 0
            if target_height < orig_height:
                crop_y_top = int((orig_height - target_height) / 2.0)
                
            # Perform crop slice
            cropped_frame = frame[crop_y_top:crop_y_top+target_height, crop_x_left:crop_x_left+target_width]
            
            # Write cropped frame to output
            out.write(cropped_frame)
            current_frame += 1
            
        cap.release()
        out.release()
        
        # 7. Merge the cropped video track back with original audio segment
        try:
            print("Combining video with original audio track...")
            with VideoFileClip(input_video_path) as full_video:
                # Extract the subclip audio
                audio_subclip = full_video.audio.subclip(start_time, end_time)
                
                # Load the cropped silent clip
                with VideoFileClip(temp_silent_path) as cropped_silent_clip:
                    # Stitch audio into the cropped video
                    final_clip = cropped_silent_clip.with_audio(audio_subclip)
                    final_clip.write_videofile(
                        output_video_path,
                        codec="libx264",
                        audio_codec="aac",
                        logger=None
                    )
            
            # Clean up silent temp file
            if os.path.exists(temp_silent_path):
                os.remove(temp_silent_path)
            print("Cropped vertical video rendered successfully!")
            return True
        except Exception as e:
            print(f"Error merging audio tracks: {e}")
            # If fallback fails, copy the silent crop to final path
            if os.path.exists(temp_silent_path):
                try:
                    os.rename(temp_silent_path, output_video_path)
                    return True
                except:
                    pass
            return False
