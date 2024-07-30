python -m hover_net.run_infer \
    --gpu='0' \
    --nr_types=6 \
    --type_info_path=type_info.json \
    --batch_size=1 \
    --model_mode=fast \
    --model_path=pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar \
    --nr_inference_workers=16 \
    --nr_post_proc_workers=16 \
    wsi \
    --input_dir=dataset/wsi/input \
    --output_dir=dataset/wsi/output \
    --input_mask_dir=dataset/sample_wsis/msk/ \
    --save_thumb \
    --save_mask
