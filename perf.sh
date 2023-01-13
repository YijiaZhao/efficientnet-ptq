run_benchmark() {
ONNX="${1}"
INPUT_NAME="${2}"
BATCH_SIZES="${3}"
INPUT_SHAPE="${4}"
PRECISION="${5}"
WORKSPACE="${6}"
LOGPATH="${7}"
SPARSITY="${8}"

mkdir ${LOGPATH}
SHAPE="${INPUT_NAME}:${BATCH_SIZES}x${INPUT_SHAPE}"
BUILDER_ARGS=""

if [ "${PRECISION}" == "fp32" ]; then
    if [ "${SPARSITY}" == "force" ]; then
        BUILDER_ARGS="${BUILDER_ARGS} --onnx=${ONNX} --shapes=${SHAPE} --workspace=${WORKSPACE} --sparsity=${SPARSITY} --saveEngine=${ONNX}_b${BATCH_SIZES}.trt"
    else
        BUILDER_ARGS="${BUILDER_ARGS} --onnx=${ONNX} --shapes=${SHAPE} --workspace=${WORKSPACE} --saveEngine=${ONNX}_b${BATCH_SIZES}.trt"
    fi;
else [ "${PRECISION}" == "fp16" ] || [ "${PRECISION}" == "int8" ];
    if [ "${SPARSITY}" == "force" ]; then
        BUILDER_ARGS="${BUILDER_ARGS} --onnx=${ONNX} --shapes=${SHAPE} --${PRECISION} --workspace=${WORKSPACE} --sparsity=${SPARSITY} --saveEngine=${ONNX}_b${BATCH_SIZES}.trt"
    else
        BUILDER_ARGS="${BUILDER_ARGS} --onnx=${ONNX} --shapes=${SHAPE} --${PRECISION} --workspace=${WORKSPACE} --saveEngine=${ONNX}_b${BATCH_SIZES}.trt"
    fi;
fi;

# echo "===============${BUILDER_ARGS} ==============="
    trtexec ${BUILDER_ARGS}
    # CUDA_VISIBLE_DEVICES=0 trtexec ${BUILDER_ARGS}
    INFERENCE_ARGS="--loadEngine=${ONNX}_b${BATCH_SIZES}.trt --warmUp=100 --iterations=300"
    trtexec ${INFERENCE_ARGS} 2>&1 | tee ${LOGPATH}${PRECISION}_b${BATCH_SIZES}.log
    # CUDA_VISIBLE_DEVICES=0 trtexec ${INFERENCE_ARGS} 2>&1 | tee ${LOGPATH}${PRECISION}_b${BATCH_SIZES}_gpu0.log& \
    # CUDA_VISIBLE_DEVICES=1 trtexec ${INFERENCE_ARGS} 2>&1 | tee ${LOGPATH}${PRECISION}_b${BATCH_SIZES}_gpu1.log& \
    # CUDA_VISIBLE_DEVICES=2 trtexec ${INFERENCE_ARGS} 2>&1 | tee ${LOGPATH}${PRECISION}_b${BATCH_SIZES}_gpu2.log& \
    # CUDA_VISIBLE_DEVICES=3 trtexec ${INFERENCE_ARGS} 2>&1 | tee ${LOGPATH}${PRECISION}_b${BATCH_SIZES}_gpu3.log& \
    # CUDA_VISIBLE_DEVICES=4 trtexec ${INFERENCE_ARGS} 2>&1 | tee ${LOGPATH}${PRECISION}_b${BATCH_SIZES}_gpu4.log& \
    # CUDA_VISIBLE_DEVICES=5 trtexec ${INFERENCE_ARGS} 2>&1 | tee ${LOGPATH}${PRECISION}_b${BATCH_SIZES}_gpu5.log& \
    # CUDA_VISIBLE_DEVICES=6 trtexec ${INFERENCE_ARGS} 2>&1 | tee ${LOGPATH}${PRECISION}_b${BATCH_SIZES}_gpu6.log& \
    # CUDA_VISIBLE_DEVICES=7 trtexec ${INFERENCE_ARGS} 2>&1 | tee ${LOGPATH}${PRECISION}_b${BATCH_SIZES}_gpu7.log


echo
}

EFFICIENT_ONNX="efficientnetv2_rw_t.onnx"
SSD300VGG=""

###### vgg16 ######
echo
echo
echo "=============================================================================="
echo "================================== Efficient =================================="
echo "=============================================================================="
# run_benchmark ${EFFICIENT_ONNX} "input_array" "1" "3x288x288" "fp32" "1024"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "8" "3x288x288" "fp32" "1024"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "16" "3x288x288" "fp32" "1024"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "32" "3x288x288" "fp32" "1024"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "64" "3x288x288" "fp32" "1024"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "128" "3x288x288" "fp32" "1024"

# run_benchmark ${EFFICIENT_ONNX} "input_array" "1" "3x288x288" "fp32" "1024" "force"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "8" "3x288x288" "fp32" "1024" "force"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "16" "3x288x288" "fp32" "1024" "force"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "32" "3x288x288" "fp32" "1024" "force"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "64" "3x288x288" "fp32" "1024" "force"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "128" "3x288x288" "fp32" "1024" "force"

# run_benchmark ${EFFICIENT_ONNX} "input_array" "1" "3x288x288" "fp16" "1024"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "8" "3x288x288" "fp16" "1024"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "16" "3x288x288" "fp16" "1024"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "32" "3x288x288" "fp16" "1024"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "64" "3x288x288" "fp16" "1024"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "128" "3x288x288" "fp16" "1024"

# run_benchmark ${EFFICIENT_ONNX} "input_array" "1" "3x288x288" "fp16" "1024" "force"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "8" "3x288x288" "fp16" "1024" "force"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "16" "3x288x288" "fp16" "1024" "force"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "32" "3x288x288" "fp16" "1024" "force"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "64" "3x288x288" "fp16" "1024" "force"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "128" "3x288x288" "fp16" "1024" "force"

# run_benchmark ${EFFICIENT_ONNX} "input_array" "1" "3x288x288" "int8" "1024" "force"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "8" "3x288x288" "int8" "1024" "force"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "16" "3x288x288" "int8" "1024" "force"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "32" "3x288x288" "int8" "1024" "force"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "64" "3x288x288" "int8" "1024" "force"
# run_benchmark ${EFFICIENT_ONNX} "input_array" "128" "3x288x288" "int8" "1024" "force"

run_benchmark ${EFFICIENT_ONNX} "input_array" "24" "3x288x288" "int8" "1024"
#run_benchmark ${EFFICIENT_ONNX} "input_array" "1" "3x288x288" "int8" "1024" "./log_dir/"
#run_benchmark ${EFFICIENT_ONNX} "input_array" "8" "3x288x288" "int8" "1024" "./log_dir/"
#run_benchmark ${EFFICIENT_ONNX} "input_array" "16" "3x288x288" "int8" "1024" "./log_dir/"
#run_benchmark ${EFFICIENT_ONNX} "input_array" "32" "3x288x288" "int8" "1024" "./log_dir/"
#run_benchmark ${EFFICIENT_ONNX} "input_array" "64" "3x288x288" "int8" "1024" "./log_dir/"
#run_benchmark ${EFFICIENT_ONNX} "input_array" "128" "3x288x288" "int8" "1024" "./log_dir/"
