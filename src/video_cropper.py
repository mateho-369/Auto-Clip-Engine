import cv2
import os
import numpy as np
from moviepy import VideoFileClip

class VideoCropper:
    def __init__(self, face_cascade_path=None, tflite_model_path="blaze_face_short_range.tflite"):
        # Setup Haar Cascade path — bundled directly in the repo. cv2.data.haarcascades
        # points to a real directory, but the actual XML file is missing from the
        # opencv-contrib-python package (confirmed empirically), so relying on it
        # silently breaks the fallback. Ship our own copy instead.
        if face_cascade_path is None:
            bundled_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
            cv2_data_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            self.face_cascade_path = bundled_path if os.path.exists(bundled_path) else cv2_data_path
        else:
            self.face_cascade_path = face_cascade_path
            
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
        
        # Setup MediaPipe Face Detector
        self.tflite_model_path = tflite_model_path
        self.mp_detector = None
        
        if os.path.exists(tflite_model_path):
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                base_options = python.BaseOptions(model_asset_path=tflite_model_path)
                options = vision.FaceDetectorOptions(base_options=base_options)
                self.mp_detector = vision.FaceDetector.create_from_options(options)
                print("MediaPipe local face detector initialized successfully!")
            except Exception as e:
                print(f"Failed to initialize MediaPipe face detector: {e}. Falling back to Haar Cascade.")
        else:
            print("MediaPipe model file blaze_face_short_range.tflite not found. Using Haar Cascade as primary.")

    def detect_face_center_mediapipe(self, frame):
        """Attempts to find the speaker's face center X-coordinate using MediaPipe Tasks API."""
        if self.mp_detector is None:
            return None
            
        try:
            import mediapipe as mp
            # MediaPipe tasks expect an RGB frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.mp_detector.detect(mp_image)
            
            if result.detections:
                # Find largest detection based on absolute width/height
                largest_det = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
                bbox = largest_det.bounding_box
                
                # Check coordinate type (MediaPipe tasks are absolute pixel dimensions)
                width = frame.shape[1]
                
                # Double-check scaling in case they are relative [0, 1] (though Tasks are usually pixels)
                if bbox.width <= 1.0:
                    center_x = (bbox.origin_x + (bbox.width / 2.0)) * width
                else:
                    center_x = bbox.origin_x + (bbox.width / 2.0)
                    
                return float(center_x)
        except Exception as e:
            print(f"MediaPipe face detection runtime error: {e}")
            
        return None

    def detect_face_center_haar(self, frame):
        """Attempts to find the speaker's face center X-coordinate using Haar Cascade."""
        if self.face_cascade.empty():
            return None
            
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scale_f = 0.5
            gray_small = cv2.resize(gray, (0, 0), fx=scale_f, fy=scale_f)
            
            faces = self.face_cascade.detectMultiScale(
                gray_small, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) > 0:
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = largest_face
                fx = int(fx / scale_f)
                fw = int(fw / scale_f)
                center_x = fx + (fw / 2.0)
                return float(center_x)
        except Exception as e:
            print(f"Haar face detection error: {e}")
            
        return None

    def crop_to_vertical(self, input_video_path, output_video_path, start_time, end_time, track_faces=True, update_progress=None):
        """
        Crops a landscape video (16:9) to vertical format (9:16) with smooth face tracking.
        Offers callback progress reporting.
        """
        print(f"Cropping video from {start_time}s to {end_time}s (Face tracking: {track_faces})...")
        
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open source video {input_video_path}")
            return False
            
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 9:16 target size calculations
        target_height = orig_height
        target_width = int(target_height * (9 / 16))
        
        if target_width % 2 != 0:
            target_width += 1
            
        if target_width > orig_width:
            target_width = orig_width
            target_height = int(target_width * (16 / 9))
            if target_height % 2 != 0:
                target_height += 1
                
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        segment_frames = max(1, end_frame - start_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        temp_silent_path = "temp_cropped_silent.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_silent_path, fourcc, fps, (target_width, target_height))
        
        smooth_center_x = orig_width / 2.0
        alpha = 0.08
        drift_back_rate = 0.03
        
        current_frame = start_frame
        frame_counter = 0
        
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_center_x = orig_width / 2.0
            target_x = frame_center_x
            
            if track_faces:
                # 1. Attempt MediaPipe face detection first
                target_x_detected = self.detect_face_center_mediapipe(frame)
                
                # 2. Fall back to Haar Cascade if MediaPipe returned None
                if target_x_detected is None:
                    target_x_detected = self.detect_face_center_haar(frame)
                    
                if target_x_detected is not None:
                    target_x = target_x_detected
                    smooth_center_x = (alpha * target_x) + ((1 - alpha) * smooth_center_x)
                else:
                    smooth_center_x = (drift_back_rate * frame_center_x) + ((1 - drift_back_rate) * smooth_center_x)
            else:
                smooth_center_x = frame_center_x
                
            crop_x_left = int(smooth_center_x - (target_width / 2.0))
            crop_x_left = max(0, min(crop_x_left, orig_width - target_width))
            
            crop_y_top = 0
            if target_height < orig_height:
                crop_y_top = int((orig_height - target_height) / 2.0)
                
            cropped_frame = frame[crop_y_top:crop_y_top+target_height, crop_x_left:crop_x_left+target_width]
            # OpenCV 5.0.0's VideoWriter.write() throws "Unknown C++ exception from
            # OpenCV code" when handed a non-contiguous array — a numpy slice like
            # this is a view, not a copy, and becomes non-contiguous once crop_x_left
            # shifts away from 0. Force a contiguous copy before writing.
            cropped_frame = np.ascontiguousarray(cropped_frame)
            out.write(cropped_frame)
            
            current_frame += 1
            frame_counter += 1
            
            # Update progress dynamically if callback registered
            if update_progress and frame_counter % 15 == 0:
                percent = int((frame_counter / segment_frames) * 100)
                update_progress(percent)
                
        cap.release()
        out.release()
        
        # Merge audio
        try:
            print("Combining vertical video with original subclipped audio track...")
            with VideoFileClip(input_video_path) as full_video:
                audio_subclip = full_video.audio.subclipped(start_time, end_time)
                with VideoFileClip(temp_silent_path) as cropped_silent_clip:
                    final_clip = cropped_silent_clip.with_audio(audio_subclip)
                    final_clip.write_videofile(
                        output_video_path,
                        codec="libx264",
                        audio_codec="aac",
                        logger=None
                    )
            if os.path.exists(temp_silent_path):
                os.remove(temp_silent_path)
            print("Cropped vertical video rendered successfully with audio!")
            return True
        except Exception as e:
            print(f"Error merging audio tracks: {e}")
            if os.path.exists(temp_silent_path):
                try:
                    os.rename(temp_silent_path, output_video_path)
                    return True
                except:
                    pass
            return False
