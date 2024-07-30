python -m hover_net.run_infer \
    --gpu='0' \
    --nr_types=6 \
    --type_info_path=type_info.json \
    --batch_size=1 \
    --model_mode=fast \
    --model_path=pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar \
    --nr_inference_workers=8 \
    --nr_post_proc_workers=8 \
    tile \
    --input_dir=dataset/tile/input \
    --output_dir=dataset/tile/output \
    --mem_usage=0.1 \
    --draw_dot \
    --save_qupath