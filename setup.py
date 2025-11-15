from cx_Freeze import setup, Executable

# Include extra files and folders
include_files = [
    "coco.names",
    "yolov3-tiny.cfg",
    "yolov3-tiny.weights",
    ("model", "model")   # speech recognition model folder
]

# Dependencies to include explicitly
build_exe_options = {
    "include_files": include_files,
    "packages": [
        "os", "sys", "cv2", "numpy", "mediapipe", "sounddevice", "vosk", "threading", "queue"
    ],
    "excludes": ["tkinter"],
    "zip_include_packages": ["*"],
    "zip_exclude_packages": []
}

# Define two executables
executables = [
    Executable("main.py", target_name="handposegame.exe", base=None),
    Executable("detection.py", target_name="objectdetection.exe", base=None)
]

# Setup
setup(
    name="HandGameAndObjectDetection",
    version="1.0",
    description="Hand pose game + YOLO object detection + speech recognition",
    options={"build_exe": build_exe_options},
    executables=executables
)
