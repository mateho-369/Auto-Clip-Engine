import os
import subprocess
import platform

_selected_codec = None

def has_nvidia_gpu():
    """Checks if a physical NVIDIA GPU is available on the host system."""
    if os.path.exists("/dev/nvidia0") or os.path.exists("/dev/nvidiactl"):
        return True
    try:
        import shutil
        if shutil.which("nvidia-smi") is not None:
            res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
            if res.returncode == 0:
                return True
    except:
        pass
    return False

def _detected_gpu_names():
    """Returns a lowercase string of detected GPU adapter names, best-effort,
    for substring-matching vendor (amd/radeon, intel). Empty string on any
    failure — callers must treat that as 'unknown', not 'no GPU'."""
    sys_platform = platform.system()
    try:
        if sys_platform == "Windows":
            # wmic is deprecated but still present on most Windows installs;
            # fall back to the PowerShell CIM cmdlet if it's missing.
            try:
                res = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3
                )
                if res.returncode == 0 and res.stdout.strip():
                    return res.stdout.lower()
            except Exception:
                pass
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
            )
            if res.returncode == 0:
                return res.stdout.lower()
        elif sys_platform == "Linux":
            res = subprocess.run(
                ["lspci"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3
            )
            if res.returncode == 0:
                return "\n".join(l for l in res.stdout.lower().splitlines() if "vga" in l or "3d" in l)
    except Exception:
        pass
    return ""

def has_amd_gpu():
    """Physical AMD/Radeon GPU check — mirrors has_nvidia_gpu()'s safety principle."""
    names = _detected_gpu_names()
    return ("amd" in names) or ("radeon" in names)

def has_intel_gpu():
    """Physical Intel GPU check — mirrors has_nvidia_gpu()'s safety principle."""
    names = _detected_gpu_names()
    return "intel" in names

def get_best_available_codec():
    """
    Auto-detects the best available hardware-accelerated video codec 
    on the workstation (NVIDIA, AMD, Intel, Apple Silicon, or CPU Fallback).
    """
    global _selected_codec
    if _selected_codec is not None:
        return _selected_codec
        
    # Default standard CPU encoder
    _selected_codec = "libx264"
    
    try:
        # 1. Probe FFmpeg's compiled encoders list
        cmd = ["ffmpeg", "-encoders"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        encoders = res.stdout
    except Exception as e:
        print(f"Error checking FFmpeg encoders: {e}. Defaulting to libx264.")
        return _selected_codec

    sys_platform = platform.system()

    # Priority A: Apple Silicon / macOS VideoToolbox
    if "h264_videotoolbox" in encoders and sys_platform == "Darwin":
        _selected_codec = "h264_videotoolbox"
        print("✔ Apple VideoToolbox hardware acceleration detected!")
        return _selected_codec

    # Priority B: NVIDIA NVENC (Requires physical card verification to prevent driver hangs)
    if "h264_nvenc" in encoders and has_nvidia_gpu():
        _selected_codec = "h264_nvenc"
        print("✔ NVIDIA NVENC hardware acceleration detected!")
        return _selected_codec

    # Priority C: AMD AMF (Native Windows AMD GPU acceleration for cards like Radeon RX / Ryzen iGPU)
    if "h264_amf" in encoders and sys_platform == "Windows" and has_amd_gpu():
        _selected_codec = "h264_amf"
        print("✔ AMD AMF hardware acceleration detected! Optimizing render for AMD Radeon.")
        return _selected_codec

    # Priority D: Intel QuickSync Video (QSV)
    if "h264_qsv" in encoders and has_intel_gpu():
        _selected_codec = "h264_qsv"
        print("✔ Intel QuickSync (QSV) hardware acceleration detected!")
        return _selected_codec

    print("ℹ No compatible GPU hardware encoder verified. Using robust CPU rendering (libx264).")
    return _selected_codec

def write_video_safely(clip, output_path, audio_codec="aac", **kwargs):
    """
    Writes a MoviePy video clip to disk, automatically using the best 
    detected hardware encoder (NVIDIA, AMD, Intel, Apple) with a failsafe 
    and robust fallback to libx264 CPU rendering.
    """
    codec = get_best_available_codec()
    
    # MoviePy only defaults pix_fmt to "yuv420p" for codecs in its own internal
    # table (libx264, mpeg4, ...). It has no entry for the hardware encoders below,
    # so without an explicit pix_fmt, ffmpeg auto-negotiates one against whatever
    # the encoder advertises support for — confirmed empirically that h264_nvenc
    # lands on "gbrp" (GBR planar) here, which standard players/browsers do not
    # handle correctly: video renders with a green cast and audio silently fails.
    # yuv420p is universally supported and must be forced explicitly.
    ffmpeg_params = kwargs.pop("ffmpeg_params", [])
    ffmpeg_params = ["-pix_fmt", "yuv420p"] + list(ffmpeg_params)

    if codec != "libx264":
        try:
            print(f"Attempting GPU acceleration ({codec}) for {output_path}...")
            clip.write_videofile(
                output_path,
                codec=codec,
                audio_codec=audio_codec,
                logger=None,
                ffmpeg_params=ffmpeg_params,
                **kwargs
            )
            print("✔ GPU rendering completed successfully!")
            return True
        except Exception as e:
            print(f"⚠️ GPU acceleration ({codec}) failed: {e}. Falling back to CPU libx264...")

    # CPU fallback
    print(f"Rendering {output_path} via CPU (libx264)...")
    try:
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec=audio_codec,
            logger=None,
            ffmpeg_params=ffmpeg_params,
            **kwargs
        )
        return True
    except Exception as e:
        print(f"❌ CPU rendering failed: {e}")
        return False
