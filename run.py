from datetime import datetime
import subprocess
import sys

PROMPT          = 'masterpiece,best quality,ultra high res,((dark skinned)) African girl,(fractal art:1.3),deep shadow,dark theme,fully clothed,necklace,forlorn,cowboy shot,'
NEGATIVE_PROMPT = 'easynegative,(badhandv4),(bad quality:1.3),(worst quality:1.3),watermark,(blurry),5-funny-looking-fingers,'

MODEL_DIR          = 'models/stable_diffusion'
EMBEDDING_DIR      = 'embeddings'
OUTPUT_DIR         = 'outputs'

WIDTH          = 512
HEIGHT         = 904
STEPS          = 40
GUIDANCE_SCALE = 7
SEED           = 4113150656
CLIP_SKIP      = 2

DENOISING_STRENGTH = 0.3
UPSCALE_FACTOR     = 2

BASE_IMG_NAME     = f'{WIDTH}x{HEIGHT}_{datetime.now().strftime('%Y%m%d_%H%M%S')}'
UPSCALED_IMG_NAME = f'{WIDTH * UPSCALE_FACTOR}x{HEIGHT * UPSCALE_FACTOR}_{datetime.now().strftime('%Y%m%d_%H%M%S')}'

# ------------------------------------------------------------------------------

print(f'Generating image with the following parameters:\n')
print(f'  Model Directory: {MODEL_DIR}')
print(f'  Embedding Directory: {EMBEDDING_DIR}')
print(f'  Output Directory: {OUTPUT_DIR}')
print(f'  Image Name: {BASE_IMG_NAME}')
print(f'  Width: {WIDTH}')
print(f'  Height: {HEIGHT}')
print(f'  Steps: {STEPS}')
print(f'  Guidance Scale: {GUIDANCE_SCALE}')
print(f'  Seed: {SEED}')
print(f'  Clip Skip: {CLIP_SKIP}')
print(f'  Denoising Strength: {DENOISING_STRENGTH}')
print(f'  Upscale Factor: {UPSCALE_FACTOR}')
print(f'  Prompt: {PROMPT}')
print(f'  Negative Prompt: {NEGATIVE_PROMPT}')

print(f'[INFO] Generating...')
subprocess.run([
    sys.executable,
    'img_generate.py',
    '--model_dir',          f'{MODEL_DIR}',
    '--embedding_dir',      f'{EMBEDDING_DIR}',
    '--output_dir',         f'{OUTPUT_DIR}/base',
    '--image_name',         f'{BASE_IMG_NAME}',

    '--width',              f'{WIDTH}',
    '--height',             f'{HEIGHT}',
    '--steps',              f'{STEPS}',
    '--guidance_scale',     f'{GUIDANCE_SCALE}',
    '--seed',               f'{SEED}',
    '--clip_skip',          f'{CLIP_SKIP}',

    '--prompt',             f'{PROMPT}',
    '--negative_prompt',    f'{NEGATIVE_PROMPT}'
], check=True)

print(f'[INFO] Upscaling...')
subprocess.run([
    sys.executable,
    'img_upscale.py',
    '--model_dir',          f'{MODEL_DIR}',
    '--embedding_dir',      f'{EMBEDDING_DIR}',
    '--output_dir',         f'{OUTPUT_DIR}/upscale',
    '--input_dir',          f'{OUTPUT_DIR}/base',
    '--base_image_name',    f'{BASE_IMG_NAME}',
    '--image_name',         f'{UPSCALED_IMG_NAME}',

    '--width',              f'{WIDTH}',
    '--height',             f'{HEIGHT}',
    '--steps',              f'{STEPS}',
    '--guidance_scale',     f'{GUIDANCE_SCALE}',
    '--seed',               f'{SEED}',
    '--clip_skip',          f'{CLIP_SKIP}',
    '--denoising_strength', f'{DENOISING_STRENGTH}',
    '--upscale_factor',     f'{UPSCALE_FACTOR}',

    '--prompt',             f'{PROMPT}',
    '--negative_prompt',    f'{NEGATIVE_PROMPT}'
], check=True)
