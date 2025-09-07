

DS_PATH=/path/to/ScreenSpot-Pro

# Download the ScreenSpot-Pro dataset from Hugging Face
hf download likaixin/ScreenSpot-Pro --repo-type dataset --local-dir $DS_PATH


# change the --planner argument to "None", "gpt-4o" or "o4-mini" to test different settings

python test_screenspot_pro.py  \
    --model_name_or_path microsoft/Phi-Ground  \
    --screenspot_imgs "${DS_PATH}/images"  \
    --screenspot_test "./new_annotations/screenspot_pro"  \
    --task "all" \
    --language "en" \
    --gt_type "positive" \
    --log_path "./eval_results/sspro.json" \
    --inst_style "instruction" \
    --planner "None" --num_crops 7 --input_format "IF" --output_format "point"