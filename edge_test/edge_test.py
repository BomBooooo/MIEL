import sys
import os

import torch
import numpy as np
import torchvision.models as models
from onnxruntime.quantization import quantize_dynamic, QuantType
import argparse
from types import SimpleNamespace
import onnxruntime as ort
import time
import psutil
import numpy as np


def main():
    # for pl in [96, 192, 336, 720, 960, 1080, 1440, 1680, 2880]:
    model_path = os.path.join(
        "checkpoints_onnx", "{}_pl{}.onnx".format(sys.argv[3], int(sys.argv[1]))
    )
    # model_path = os.path.join("checkpoints_onnx", "traffic_pl{}_quant.onnx".format(pl))

    input_shape = (1, 3, 96, 862) if sys.argv[3] == "traffic" else (1, 3, 96, 7)
    options = ort.SessionOptions()
    # options.enable_cuda_graph = True  # 启用 CUDA 图优化

    providers = ["CUDAExecutionProvider"]  # 使用 GPU
    if (int(sys.argv[4]) == 0):
        session = ort.InferenceSession(model_path, options)  # , providers=providers)
    else:
        session = ort.InferenceSession(model_path, options, providers=providers)
    input_name = session.get_inputs()[0].name
    # print("当前使用的执行提供程序:", session.get_providers())

    # 模拟输入数据
    # print(model_path)
    # if 'quant' in model_path:
    #     dummy_input = np.random.randn(*input_shape).astype(np.float32)
    #     dummy_input = (dummy_input / 0.05).astype(np.uint8)  # scale 是量化比例因子
    # else:
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    # print(dummy_input.shape)

    # 将 NumPy 数组转换为 PyTorch 张量，并移动到 GPU
    dummy_input_tensor = torch.from_numpy(dummy_input).to("cuda")
    # 将 PyTorch 张量转换回 NumPy 数组（仍在 GPU 上）
    dummy_input_gpu = dummy_input_tensor.cpu().numpy()
    # 注意：这里需要将数据移回 CPU，因为 ONNX Runtime 需要 NumPy 数组

    # 预热
    test_times = 10
    start_time = time.time()
    for _ in range(test_times):
        output = session.run(None, {input_name: dummy_input_gpu})
    avg_latency = (time.time() - start_time) / test_times * 1000  # 转换为毫秒
    print(f"预热 - 平均延迟: {avg_latency:.2f}ms")
    # print('test...')
    # 检查输出是否为 GPU 上的张量
    # print("输出类型:", type(output[0]))
    # print("输出设备:", "GPU" if "cuda" in str(type(output[0])) else "CPU")
    # 测试延迟
    if (int(sys.argv[2]) == 1):
      test_times = 5000
      # start_time = time.time()
      for _ in range(test_times):
          output = session.run(None, {input_name: dummy_input_gpu})
    #   avg_latency = (time.time() - start_time) / test_times * 1000  # 转换为毫秒
    #   print(f"FP32模型 - 平均延迟: {avg_latency:.2f}ms")

    # 测量内存占用（峰值）
    # process = psutil.Process(os.getpid())
    # mem_usage = process.memory_info().rss / 1024**2  # 转换为MB


if __name__ == "__main__":
    main()
