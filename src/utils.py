import os
import subprocess

_nvenc_available = None

def has_nvidia_gpu():
    """Checks if a physical NVIDIA GPU is available on the host system."""
    # On Linux, check for standard device nodes
    if os.path.exists("/dev/nvidia0") or os.path.exists("/dev/nvidiactl"):
        return True
    # On Windows or generic, check if nvidia-smi command exists in PATH
    try:
        import shutil
        if shutil.which("nvidia-smi") is not None:
            # Let's run a quick non-blocking nvidia-smi check to ensure driver is functional
            res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
            if res.returncode == 0:
                return True
    except:
        pass
    return False

def is_nvenc_available():
    global _nvenc_available
    if _nvenc_available is not None:
        return _nvenc_available
        
    # Skip NVENC check completely if no physical NVIDIA GPU device is present (prevents driver-load hangs)
    if not has_nvidia_gpu():
        _nvenc_available = False
        return False
        
    try:
        cmd = ["ffmpeg", "-encoders"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        if "h264_nvenc" in res.stdout:
            _nvenc_available = True
            print("NVIDIA NVENC hardware acceleration detected and verified!")
            return True
    except Exception as e:
        print(f"Error checking NVENC encoders: {e}")
    _nvenc_available = False
    return False

def write_video_safely(clip, output_path, audio_codec="aac", **kwargs):
    """
    Writes a MoviePy video clip to disk, trying NVENC hardware encoding first
    if available, with automatic, failsafe fallback to standard libx264 CPU encoding.
    """
    if is_nvenc_available():
        try:
            print(f"Attempting GPU acceleration (NVENC) for {output_path}...")
            clip.write_videofile(
                output_path,
                codec="h264_nvenc",
                audio_codec=audio_codec,
                logger=None,
                **kwargs
            )
            print("GPU render complete!")
            return True
        except Exception as e:
            print(f"NVENC hardware encoding failed ({e}). Falling back to libx264...")
            
    # Standard libx264 CPU fallback
    print(f"Rendering {output_path} via CPU (libx264)...")
    try:
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec=audio_codec,
            logger=None,
            **kwargs
        )
        return True
    except Exception as e:
        print(f"CPU encoding failed: {e}")
        return False
