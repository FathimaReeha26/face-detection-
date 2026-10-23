# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

block_cipher = None

mediapipe_datas, mediapipe_binaries, mediapipe_hiddenimports = collect_all("mediapipe")

a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=mediapipe_binaries,
    datas=[
        ("src", "FaceDetectionApp/src"),
        ("models/face_detection_yunet_2023mar.onnx", "FaceDetectionApp/models"),
        ("models/face_recognition_sface_2021dec.onnx", "FaceDetectionApp/models"),
        ("models/scrfd_10g.onnx", "FaceDetectionApp/models"),
        ("models/face_landmarker.task", "FaceDetectionApp/models"),
        ("models/deploy.prototxt", "FaceDetectionApp/models"),
        ("models/res10_300x300_ssd_iter_140000.caffemodel", "FaceDetectionApp/models"),
    ] + mediapipe_datas,
    hiddenimports=[
        "cv2",
        "numpy",
        "onnxruntime",
        "face_detector",
        "face_recognizer",
        "detection",
        "detection.backends",
        "detection.tracking",
        "liveness",
        "dnn_detector",
    ] + mediapipe_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "tkinter.test"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="FaceDetectionLauncher",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)