# transformers-mtl
Goal: Use MTL NPU as transformers backend device.

Approach: Hacking model forward function to call OpenVINO to run on NPU.
