from ctypes import sizeof
import tensorrt as trt
import pycuda.driver as cuda
import os
import torch
import pycuda.autoinit
import numpy as np

class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_folder_path, cache_file, data_loader, cali_max_batch=300, batch_size=1):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        # Prepare data for the the following
        self.calib_folder_path = calib_folder_path
        file_list = os.listdir(self.calib_folder_path)
        self.image_path_generator = iter(file_list)
        self.cache_file = cache_file
        self.batch_size = batch_size
        # Allocate enough memory for a whole batch.
        # self.device_input = cuda.mem_alloc(self.data.nbytes * self.batch_size)
        self.device_input = cuda.mem_alloc(288*288*3 * self.batch_size * 4)
        self.data_loader = data_loader
        self.batches = self.load_batches()
        self.cali_max_batch = cali_max_batch
    
    def get_batch_size(self):
        return self.batch_size

    def load_batches(self):
        # Populates a persistent self.batch buffer with images.
        for batch_idx, (input, target) in enumerate(self.data_loader):
            if batch_idx < self.cali_max_batch:
                print("calib batch: {} / {}".format(batch_idx, self.cali_max_batch))
                input = input.contiguous(memory_format=torch.channels_last)
                yield input
            else:
                break

    def get_batch(self, names):
        try:
            batch = next(self.batches)
            image_array = np.ascontiguousarray(batch.cpu().numpy().astype(np.float32))
            # import pdb;pdb.set_trace()
            cuda.memcpy_htod(self.device_input, image_array)
            # print("11111")
            return [int(self.device_input)]

        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None
        

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)