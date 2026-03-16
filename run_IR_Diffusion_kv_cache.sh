python ./src/DreamStory/pipe/pipe_test.py test --prompts_path="./results/examples/example.json" --output_root="./results/example_IR-Diffusion_kv_cache/" \
            --sam_step 50 --is_spatial_self_attn True --is_isolate_sa True --attn_processor_name "DreamStoryScaledAttnProcessor" --use_kv_cache True
