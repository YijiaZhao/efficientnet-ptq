import logging
import tensorrt as trt

def GiB(val):
    return val * 1 << 30

def build_int8_engine(onnx_file_path, calibrator, batch_size, calibration_cache):
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    TRT_LOGGER = trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = batch_size

        config.max_workspace_size = GiB(2)
        config.flags |= 1 << int(trt.BuilderFlag.INT8)
        config.flags |= 1 << int(trt.BuilderFlag.FP16)

        # config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        config.int8_calibrator = calibrator

        # Parse Onnx model
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # For the fixed batch, please use the following code
        network.get_input(0).shape = [batch_size, 3, 288, 288]
        
        # For dynamic batch, please use the following code
        # profile = builder.create_optimization_profile()
        # profile.set_shape("input_array", (24, 3, 288, 288))
        # config.add_optimization_profile(profile)
        # config.set_calibration_profile(profile)




        # Start to build engine and do int8 calibration.
        print('--- Starting to build engine! ---')
        engine = builder.build_engine(network, config)
        print('--- Building engine is finished! ---')

        return engine