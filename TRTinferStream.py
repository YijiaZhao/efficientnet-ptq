import logging
import argparse
import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms.functional import convert_image_dtype
#from image_batcher import ImageBatcher
#from visualize import visualize_detections
from torch.utils import data

logging.basicConfig(level=logging.ERROR)  # INFO, WARNING, ERROR
logging.getLogger("EngineInference").setLevel(logging.ERROR)
log = logging.getLogger("EngineInference")
from tqdm import tqdm
        
class TensorRTInferStream:
    def __init__(self, engine_path, streams=1):
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.contexts=[]
        self.streams=streams
        for i in range(self.streams):
            context = self.engine.create_execution_context()
            context.active_optimization_profile = 0
            self.contexts.append(context)

    def set_shape(self, shape):
        assert self.engine
        for i in range(self.streams):
            self.contexts[i].set_binding_shape(0, (shape[0], shape[1], shape[2], shape[3]))
            assert self.contexts[i]
            
        # Setup I/O bindings
        self.allocations = [[] for _ in range(self.streams)]

        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            dtype = self.engine.get_binding_dtype(i)
            shape = self.contexts[0].get_binding_shape(i)
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s

            for stream_i in range(self.streams):
                allocation = cuda.mem_alloc(size)
                self.allocations[stream_i].append(allocation)
