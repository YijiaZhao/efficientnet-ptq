rm -rf ./log/
export CUDA_DEVICE_ORDER=PCI_BUS_ID
gpu_all=$[`nvidia-smi -L |wc -l` - 1]
mkdir log
for gpu in `seq 0 ${gpu_all}` ;
do
    export CUDA_VISIBLE_DEVICES=${gpu}
    export save_dir="gpu_${gpu}"
    python validate_trt.py /val_data/ --model efficientnetv2_rw_t --pretrained --batch-size 24 --engine_name efficientnetv2_rw_t.trt --streams $1 --save_dir $save_dir 2>&1 | tee ./log/${gpu}_b24.log &
done
