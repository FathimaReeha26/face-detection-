# Anti-spoofing model weights

This folder is intentionally empty in the repository. You must download the
pretrained MiniFASNet weights manually before running the liveness check
with anti-spoofing enabled.

Model: Silent-Face-Anti-Spoofing (MiniFASNet), from
https://github.com/minivision-ai/Silent-Face-Anti-Spoofing

Weights are published under that project's license. Check the license before
using the model in your own project or product.

This project runs the model through ONNX Runtime. Place an ONNX export of the
weights in this folder, for example:

    models/antispoof/model.onnx

You can obtain ONNX weights from the yakhyo/face-anti-spoofing release or
convert the PyTorch weights with that project's onnx_export.py. PyTorch
(.pth) files are not loaded directly by this project.
