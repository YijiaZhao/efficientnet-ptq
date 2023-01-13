python onnx2trt.py /click/val_data/ --model efficientnetv2_rw_t --pretrained --batch-size 24 --onnx_name efficientnetv2_rw_t --engine_name efficientnetv2_rw_t
#验证精度：
python validate_trt_ppl.py /click/val_data/ --model efficientnetv2_rw_t --pretrained --batch-size 24 --engine_name efficientnetv2_rw_t.trt --acc
#异步拷贝qps：
python validate_trt_ppl.py /click/val_data/ --model efficientnetv2_rw_t --pretrained --batch-size 24 --engine_name efficientnetv2_rw_t.trt --tested_batch_times 50
#qps测算：
bash perf.sh