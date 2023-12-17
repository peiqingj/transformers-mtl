from typing import TYPE_CHECKING
import numpy as np
import openvino as ov
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

try:
    from einops import rearrange
except ImportError:
    rearrange = None
from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator


qwen_compiled_models = {}
DEVICE = "GPU"


def create_ov_mlp_model(np_hidden_states, np_w1_weight, np_w2_weight, np_c_proj_weight):
    import openvino.runtime as ovrt

    hidden_states = ovrt.opset1.parameter(
        np_hidden_states.shape, np_hidden_states.dtype
    )
    w1_weight = ovrt.opset1.parameter(np_w1_weight.shape, np_w1_weight.dtype)
    w2_weight = ovrt.opset1.parameter(np_w2_weight.shape, np_w2_weight.dtype)
    c_proj_weight = ovrt.opset1.parameter(
        np_c_proj_weight.shape, np_c_proj_weight.dtype
    )
    w1 = ovrt.opset1.matmul(hidden_states, w1_weight, False, True)
    w2 = ovrt.opset1.matmul(hidden_states, w2_weight, False, True)
    silu = ovrt.opset4.swish(w2)
    mul = ovrt.opset1.multiply(w1, silu)
    c_proj = ovrt.opset1.matmul(mul, c_proj_weight, False, True)
    out = ovrt.opset1.result(c_proj)
    model = ovrt.Model(
        [out], [hidden_states, w1_weight, w2_weight, c_proj_weight], "QWenMLP"
    )
    return model


def qwenmlp_forward(self, hidden_states):
    TensorClass = hidden_states.__class__
    np_hidden_states = hidden_states.numpy()
    np_w1 = self.w1.weight.numpy()
    np_w2 = self.w2.weight.numpy()
    np_c_proj = self.c_proj.weight.numpy()

    model_spec = f"{np_hidden_states.shape}_mlp"
    if model_spec not in qwen_compiled_models:
        print(f"compiling {model_spec}")
        model = create_ov_mlp_model(np_hidden_states, np_w1, np_w2, np_c_proj)
        qwen_compiled_models[model_spec] = ov.compile_model(model, DEVICE)
    output = qwen_compiled_models[model_spec].infer(
        [np_hidden_states, np_w1, np_w2, np_c_proj]
    )[0]
    return TensorClass(output)


def get_modeling_qwen():
    import inspect
    import transformers

    # get modeling_qwen module and hacking QWenAttention by QWenAttentionNPU
    QWenLMHeadModel = transformers.dynamic_module_utils.get_class_from_dynamic_module(
        "modeling_qwen.QWenLMHeadModel", "Qwen/Qwen-1_8B-Chat"
    )
    modeling_qwen = inspect.getmodule(QWenLMHeadModel)
    return modeling_qwen


def hack_mlp():
    modeling_qwen = get_modeling_qwen()
    qwenmlp_forward.hacking_target = modeling_qwen.QWenMLP.forward
    modeling_qwen.QWenMLP.forward = qwenmlp_forward
