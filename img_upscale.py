import argparse
import torch

from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler

parser = argparse.ArgumentParser()
parser.add_argument('--base_image_path',    default=None,                       type=str,   help='Path to the image to be upscaled.'           )

parser.add_argument('--model_dir',          default='models/stable_diffusion',  type=str,   help='Path to the SD models directory.'            )
parser.add_argument('--embedding_dir',      default='embeddings',               type=str,   help='Path to the Negative Embeddings directory.'  )
parser.add_argument('--output_dir',         default='outputs/upscale',          type=str,   help='Path to the Output directory.'               )
parser.add_argument('--input_dir',          default='outputs/base',             type=str,   help='Path to the Input directory.'                )
parser.add_argument('--base_image_name',    default='tmp_image.png',            type=str,   help='Name of the input image.'                    )
parser.add_argument('--image_name',         default='tmp_image.png',            type=str,   help='Name of the output image.'                   )

parser.add_argument('--width',              default=512,                        type=int,   help='Width of the image.'                         )
parser.add_argument('--height',             default=512,                        type=int,   help='Height of the image.'                        )
parser.add_argument('--steps',              default=20,                         type=int,   help='Number of steps for the image generation.'   )
parser.add_argument('--guidance_scale',     default=10.0,                       type=float, help='Guidance scale for the image generation.'    )
parser.add_argument('--seed',               default=-1,                         type=int,   help='Seed for the image generation.'              )
parser.add_argument('--clip_skip',          default=2,                          type=int,   help='Clip skip.'                                  )
parser.add_argument('--denoising_strength', default=0.7,                        type=float, help='Denoising strength for the image generation.')
parser.add_argument('--upscale_factor',     default=2,                          type=int,   help='Upscale factor for the image generation.'    )

parser.add_argument('--prompt',             default='spiderman',                type=str,   help='Prompt for the image generation.'            )
parser.add_argument('--negative_prompt',    default='bad quality',              type=str,   help='Negative prompt for the image generation.'   )
args = parser.parse_args()

MODEL_DIR          = f'{args.model_dir}/realcartoon3d_v11'
EMBEDDING_DIR      = args.embedding_dir
OUTPUT_DIR         = args.output_dir
BASE_IMAGE_NAME    = f'{args.input_dir}/{args.base_image_name}.png'
IMAGE_NAME         = f'{args.image_name}'

WIDTH, HEIGHT      = args.width, args.height
STEPS              = args.steps
GUIDANCE_SCALE     = args.guidance_scale
SEED               = args.seed
CLIP_SKIP          = args.clip_skip

DENOISING_STRENGTH = args.denoising_strength
UPSCALE_FACTOR     = args.upscale_factor

PROMPT             = args.prompt
NEGATIVE_PROMPT    = args.negative_prompt

#==================================================================================================================================================================

base_image = Image.open(BASE_IMAGE_NAME).convert('RGB')

# LOAD MODEL -------------------------------------------------------------------
generator = torch.Generator(device='cuda').manual_seed(SEED)
img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    safety_checker=None,
    local_files_only=True
).to('cuda')
# ------------------------------------------------------------------------------

# LOAD SCHEDULER ---------------------------------------------------------------
img2img.scheduler = EulerAncestralDiscreteScheduler.from_config(
    img2img.scheduler.config
)
# ------------------------------------------------------------------------------

# CLIP SKIP --------------------------------------------------------------------
total_layers = img2img.text_encoder.config.num_hidden_layers
img2img.text_encoder.num_hidden_layers = total_layers - CLIP_SKIP
# ------------------------------------------------------------------------------

# LOAD NEGATIVE EMBEDDINGS -----------------------------------------------------
img2img.load_textual_inversion(f'{EMBEDDING_DIR}/badhandv4.pt', weight_name='badhandv4')
# ------------------------------------------------------------------------------


# UPSCALE ----------------------------------------------------------------------
base_resized = base_image.resize(( WIDTH * UPSCALE_FACTOR, HEIGHT * UPSCALE_FACTOR), resample=Image.LANCZOS)
img2img(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    image=base_resized,
    strength=DENOISING_STRENGTH,
    guidance_scale=GUIDANCE_SCALE,
    num_inference_steps=STEPS,
    generator=generator
).images[0].save(f'{OUTPUT_DIR}/{IMAGE_NAME}.png')
# ------------------------------------------------------------------------------