import os, io, base64, time, logging, asyncio, json, sys

# Monkey-patch for basicsr/torchvision compatibility
try:
    import torchvision.transforms.functional as _F
    sys.modules["torchvision.transforms.functional_tensor"] = _F
except ImportError:
    pass

import torch
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from PIL import Image, ImageFilter
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image-gen-cuda")

# Global state
pipelines: Dict[str, Any] = {}
inpaint_pipeline = None
upscaler = None
executor = ThreadPoolExecutor(max_workers=1)

# Device config
DEVICE = os.environ.get("TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
USE_HALF = os.environ.get("HALF_PRECISION", "true").lower() == "true" and DEVICE == "cuda"
DTYPE = torch.float16 if USE_HALF else torch.float32

# Model registry — uses HuggingFace model IDs or local paths
MODELS = {
    "sd15": {
        "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "local_path": "/data/models/sd-v1-5",
        "type": "sd15",
        "description": "Stable Diffusion 1.5 (CUDA)",
        "default_steps": 20,
        "default_guidance": 7.5,
    },
    "dreamshaper": {
        "model_id": "Lykon/dreamshaper-8",
        "local_path": "/data/models/dreamshaper-8",
        "type": "sd15",
        "description": "DreamShaper 8 (high quality, SD 1.5 based)",
        "default_steps": 25,
        "default_guidance": 7.0,
    },
    "sdxl-turbo": {
        "model_id": "stabilityai/sdxl-turbo",
        "local_path": "/data/models/sdxl-turbo",
        "type": "sdxl",
        "description": "SDXL Turbo (fast, 4 steps)",
        "default_steps": 4,
        "default_guidance": 0.0,
    },
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "local_path": "/data/models/sdxl-base",
        "type": "sdxl",
        "description": "SDXL 1.0 Base (high quality)",
        "default_steps": 30,
        "default_guidance": 7.5,
    },
    "flux-schnell": {
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "local_path": "/data/models/flux-schnell",
        "type": "flux",
        "description": "FLUX.1 Schnell (fast, high quality)",
        "default_steps": 4,
        "default_guidance": 0.0,
    },
}

INPAINT_MODEL_ID = "runwayml/stable-diffusion-inpainting"
INPAINT_LOCAL_PATH = "/data/models/sd15-inpainting"

# Style presets
STYLE_PRESETS = {
    "photorealistic": {
        "prompt_prefix": "photorealistic, ultra detailed, professional photography, 8k, sharp focus, ",
        "negative": "cartoon, anime, illustration, painting, drawing, unrealistic, blurry, low quality, deformed",
    },
    "anime": {
        "prompt_prefix": "anime style, vibrant colors, clean linework, detailed, ",
        "negative": "photorealistic, photo, realistic, blurry, low quality, deformed hands",
    },
    "landscape": {
        "prompt_prefix": "breathtaking landscape photography, golden hour, dramatic lighting, 8k, ",
        "negative": "people, person, human, text, watermark, low quality, blurry",
    },
    "scifi": {
        "prompt_prefix": "sci-fi concept art, cyberpunk, futuristic, neon lights, detailed, ",
        "negative": "medieval, fantasy, low quality, blurry, deformed",
    },
    "cute-dog": {
        "prompt_prefix": "adorable cute puppy, happy, playful, professional pet photography, soft lighting, bokeh, ",
        "negative": "aggressive, scary, low quality, blurry, deformed, ugly",
    },
}

# Aspect ratio presets (width, height)
ASPECT_RATIOS = {
    "square": (512, 512),
    "wide": (768, 432),
    "ultrawide": (768, 320),
    "portrait": (432, 768),
}

ASPECT_RATIOS_SDXL = {
    "square": (1024, 1024),
    "wide": (1024, 576),
    "ultrawide": (1024, 448),
    "portrait": (576, 1024),
}

DEFAULT_NEGATIVE = "ugly, blurry, low quality, deformed, disfigured, bad anatomy, bad hands, missing fingers, extra fingers, watermark, text"


def _resolve_model_path(config: dict) -> str:
    """Return local path if exists, otherwise HuggingFace model ID for download."""
    if os.path.exists(config["local_path"]):
        return config["local_path"]
    return config["model_id"]


def load_pipeline(model_key: str):
    """Load a pipeline by model key."""
    if model_key in pipelines:
        return pipelines[model_key]

    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

    config = MODELS[model_key]
    model_path = _resolve_model_path(config)

    logger.info(f"Loading model '{model_key}' from {model_path} (device={DEVICE}, dtype={DTYPE})...")

    if config["type"] == "flux":
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=DTYPE)
    elif config["type"] == "sdxl":
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=DTYPE)
    else:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=DTYPE)

    pipe = pipe.to(DEVICE)

    # Enable memory optimisations
    try:
        pipe.enable_xformers_memory_efficient_attention()
        logger.info(f"xformers enabled for {model_key}")
    except Exception:
        logger.info(f"xformers not available for {model_key}, using default attention")

    # Save locally if downloaded from HuggingFace
    if not os.path.exists(config["local_path"]):
        try:
            os.makedirs(config["local_path"], exist_ok=True)
            pipe.save_pretrained(config["local_path"])
            logger.info(f"Saved model to {config['local_path']}")
        except Exception as e:
            logger.warning(f"Could not save model locally: {e}")

    pipelines[model_key] = pipe
    logger.info(f"Model '{model_key}' loaded on {DEVICE}")

    # Log VRAM usage
    if DEVICE == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"VRAM: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    return pipe


def load_inpaint_pipeline_fn():
    """Load the SD 1.5 inpainting pipeline."""
    global inpaint_pipeline
    if inpaint_pipeline is not None:
        return inpaint_pipeline

    from diffusers import StableDiffusionInpaintPipeline

    model_path = INPAINT_LOCAL_PATH if os.path.exists(INPAINT_LOCAL_PATH) else INPAINT_MODEL_ID
    logger.info(f"Loading inpainting model from {model_path}...")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path, torch_dtype=DTYPE)
    pipe = pipe.to(DEVICE)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # Save locally if downloaded
    if not os.path.exists(INPAINT_LOCAL_PATH):
        try:
            os.makedirs(INPAINT_LOCAL_PATH, exist_ok=True)
            pipe.save_pretrained(INPAINT_LOCAL_PATH)
        except Exception as e:
            logger.warning(f"Could not save inpaint model: {e}")

    inpaint_pipeline = pipe
    logger.info("Inpainting pipeline loaded")
    return pipe


def load_upscaler():
    """Load Real-ESRGAN upscaler."""
    global upscaler
    if upscaler is not None:
        return upscaler

    model_path = "/data/models/upscaler/RealESRGAN_x4plus.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Real-ESRGAN model not found at /data/models/upscaler/RealESRGAN_x4plus.pth")

    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        rrdb_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upscaler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=rrdb_model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=USE_HALF,
            device=DEVICE,
        )
        logger.info(f"Real-ESRGAN upscaler loaded on {DEVICE}")
        return upscaler
    except ImportError:
        logger.warning("Real-ESRGAN not installed, upscaling unavailable")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Log GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({vram_total:.1f}GB VRAM)")
    else:
        logger.warning("No CUDA GPU detected, running on CPU")

    # Load default model on startup
    default_model = os.environ.get("DEFAULT_MODEL", "sd15")
    try:
        load_pipeline(default_model)
    except Exception as e:
        logger.error(f"Failed to load default model '{default_model}': {e}")
        if default_model != "sd15":
            try:
                load_pipeline("sd15")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")

    yield

    pipelines.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Image Generation API (CUDA)", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    model: Optional[str] = None
    style_preset: Optional[str] = None
    aspect_ratio: Optional[str] = None
    upscale: Optional[bool] = False
    upscale_target_width: Optional[int] = None
    upscale_target_height: Optional[int] = None


class GenerateResponse(BaseModel):
    image_base64: str
    seed: int
    elapsed_seconds: float
    model: str
    width: int
    height: int
    style_preset: Optional[str] = None


class OutpaintRequest(BaseModel):
    image_base64: str
    target_width: int = 1920
    target_height: int = 1080
    prompt: str = ""
    negative_prompt: Optional[str] = None
    placement: str = "center"
    model: Optional[str] = "sd15"
    steps: Optional[int] = 25
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None
    blend_feather: int = 40


class OutpaintResponse(BaseModel):
    image_base64: str
    seed: int
    elapsed_seconds: float
    width: int
    height: int
    method: str


def _resolve_params(req: GenerateRequest):
    """Resolve all generation parameters with defaults."""
    model_key = req.model or os.environ.get("DEFAULT_MODEL", "sd15")
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

    config = MODELS[model_key]
    is_sdxl = config["type"] in ("sdxl", "flux")
    ar_map = ASPECT_RATIOS_SDXL if is_sdxl else ASPECT_RATIOS

    if req.aspect_ratio and req.aspect_ratio in ar_map:
        width, height = ar_map[req.aspect_ratio]
    elif req.width and req.height:
        width, height = req.width, req.height
    else:
        width = req.width or (1024 if is_sdxl else 512)
        height = req.height or (1024 if is_sdxl else 512)

    steps = req.steps or config["default_steps"]
    guidance = req.guidance_scale if req.guidance_scale is not None else config["default_guidance"]

    prompt = req.prompt
    negative = req.negative_prompt or DEFAULT_NEGATIVE

    if req.style_preset and req.style_preset in STYLE_PRESETS:
        preset = STYLE_PRESETS[req.style_preset]
        prompt = preset["prompt_prefix"] + prompt
        if not req.negative_prompt:
            negative = preset["negative"] + ", " + DEFAULT_NEGATIVE

    return model_key, prompt, negative, width, height, steps, guidance


def _upscale_image(img: Image.Image, target_width: int = None, target_height: int = None) -> Image.Image:
    """Upscale image using Real-ESRGAN."""
    up = load_upscaler()
    if up is None:
        raise RuntimeError("Upscaler not available")

    import cv2
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    output, _ = up.enhance(img_bgr, outscale=4)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(output_rgb)

    if target_width and target_height:
        result = result.resize((target_width, target_height), Image.LANCZOS)
    elif target_width:
        ratio = target_width / result.width
        result = result.resize((target_width, int(result.height * ratio)), Image.LANCZOS)

    return result


def _run_pipeline(model_key, prompt, negative, width, height, steps, guidance, seed,
                  upscale=False, upscale_tw=None, upscale_th=None):
    """Run the generation pipeline."""
    pipe = load_pipeline(model_key)
    config = MODELS[model_key]

    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    t0 = time.time()

    kwargs = dict(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )

    # FLUX doesn't use negative prompts
    if config["type"] != "flux":
        kwargs["negative_prompt"] = negative

    with torch.inference_mode():
        result = pipe(**kwargs)

    img = result.images[0]
    gen_elapsed = time.time() - t0
    logger.info(f"Generated {width}x{height} in {gen_elapsed:.1f}s with {model_key}: {prompt[:80]}")

    if upscale:
        t1 = time.time()
        try:
            img = _upscale_image(img, upscale_tw, upscale_th)
            up_elapsed = time.time() - t1
            logger.info(f"Upscaled to {img.width}x{img.height} in {up_elapsed:.1f}s")
        except Exception as e:
            logger.warning(f"Upscaling failed: {e}")

    elapsed = time.time() - t0
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), seed, elapsed, img.width, img.height


def _calc_placement(src_w, src_h, target_w, target_h, placement):
    """Calculate where to place the source in the target canvas."""
    scale = min(target_w / src_w, target_h / src_h)
    new_src_w = int(src_w * scale)
    new_src_h = int(src_h * scale)

    if placement == "center":
        x, y = (target_w - new_src_w) // 2, (target_h - new_src_h) // 2
    elif placement == "left":
        x, y = 0, (target_h - new_src_h) // 2
    elif placement == "right":
        x, y = target_w - new_src_w, (target_h - new_src_h) // 2
    elif placement == "top":
        x, y = (target_w - new_src_w) // 2, 0
    elif placement == "bottom":
        x, y = (target_w - new_src_w) // 2, target_h - new_src_h
    else:
        x, y = (target_w - new_src_w) // 2, (target_h - new_src_h) // 2

    return x, y, new_src_w, new_src_h


def _create_outpaint_canvas(source_img, target_w, target_h, placement, feather=40):
    """Create padded canvas and mask for outpainting."""
    src_w, src_h = source_img.size
    src_x, src_y, new_src_w, new_src_h = _calc_placement(src_w, src_h, target_w, target_h, placement)

    scaled_source = source_img.resize((new_src_w, new_src_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (128, 128, 128))

    # Edge-extend fill for context
    if src_x > 0:
        left_strip = scaled_source.crop((0, 0, min(src_x, new_src_w), new_src_h))
        left_strip = left_strip.transpose(Image.FLIP_LEFT_RIGHT)
        canvas.paste(left_strip.resize((src_x, new_src_h), Image.LANCZOS), (0, src_y))

    right_start = src_x + new_src_w
    if right_start < target_w:
        right_w = target_w - right_start
        right_strip = scaled_source.crop((max(0, new_src_w - right_w), 0, new_src_w, new_src_h))
        right_strip = right_strip.transpose(Image.FLIP_LEFT_RIGHT)
        canvas.paste(right_strip.resize((right_w, new_src_h), Image.LANCZOS), (right_start, src_y))

    if src_y > 0:
        top_strip = scaled_source.crop((0, 0, new_src_w, min(src_y, new_src_h)))
        top_strip = top_strip.transpose(Image.FLIP_TOP_BOTTOM)
        canvas.paste(top_strip.resize((new_src_w, src_y), Image.LANCZOS), (src_x, 0))

    bottom_start = src_y + new_src_h
    if bottom_start < target_h:
        bottom_h = target_h - bottom_start
        bottom_strip = scaled_source.crop((0, max(0, new_src_h - bottom_h), new_src_w, new_src_h))
        bottom_strip = bottom_strip.transpose(Image.FLIP_TOP_BOTTOM)
        canvas.paste(bottom_strip.resize((new_src_w, bottom_h), Image.LANCZOS), (src_x, bottom_start))

    blurred_canvas = canvas.filter(ImageFilter.GaussianBlur(radius=20))
    canvas.paste(blurred_canvas, (0, 0))
    canvas.paste(scaled_source, (src_x, src_y))

    mask = Image.new("L", (target_w, target_h), 255)
    source_mask_region = Image.new("L", (new_src_w, new_src_h), 0)
    mask.paste(source_mask_region, (src_x, src_y))

    if feather > 0:
        mask_inv = Image.fromarray(255 - np.array(mask))
        mask_inv_blurred = mask_inv.filter(ImageFilter.GaussianBlur(radius=feather))
        mask = Image.fromarray(255 - np.array(mask_inv_blurred))

    return canvas, mask, src_x, src_y, new_src_w, new_src_h


def _clamp_to_pipeline_size(w, h, max_dim=768):
    """Scale dimensions to fit within max_dim, keeping aspect ratio and aligning to 64."""
    scale = min(max_dim / w, max_dim / h, 1.0)
    new_w = max(64, int(w * scale / 64) * 64)
    new_h = max(64, int(h * scale / 64) * 64)
    return new_w, new_h


def _run_outpaint(req_data: dict) -> tuple:
    """Core outpainting logic."""
    t0 = time.time()

    img_b64 = req_data["image_base64"]
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]
    src_bytes = base64.b64decode(img_b64)
    source_img = Image.open(io.BytesIO(src_bytes)).convert("RGB")

    target_w = req_data["target_width"]
    target_h = req_data["target_height"]
    prompt = req_data.get("prompt", "")
    negative = req_data.get("negative_prompt") or DEFAULT_NEGATIVE
    placement = req_data.get("placement", "center")
    steps = req_data.get("steps", 25)
    guidance = req_data.get("guidance_scale", 7.5)
    seed = req_data.get("seed") or int(time.time()) % (2**32)
    feather = req_data.get("blend_feather", 40)

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    canvas, mask, src_x, src_y, new_src_w, new_src_h = _create_outpaint_canvas(
        source_img, target_w, target_h, placement, feather
    )

    proc_w, proc_h = _clamp_to_pipeline_size(target_w, target_h, max_dim=768)
    method = "img2img_blend"

    try:
        pipe = load_inpaint_pipeline_fn()
        proc_canvas = canvas.resize((proc_w, proc_h), Image.LANCZOS)
        proc_mask = mask.resize((proc_w, proc_h), Image.NEAREST)

        logger.info(f"Running inpainting at {proc_w}x{proc_h}, steps={steps}")

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=proc_canvas,
                mask_image=proc_mask,
                width=proc_w,
                height=proc_h,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
        gen_img = result.images[0]
        gen_img = gen_img.resize((target_w, target_h), Image.LANCZOS)

        scaled_source = source_img.resize((new_src_w, new_src_h), Image.LANCZOS)
        src_region_mask = Image.new("L", (target_w, target_h), 0)
        src_region_inner = Image.new("L", (new_src_w, new_src_h), 255)
        src_region_mask.paste(src_region_inner, (src_x, src_y))
        if feather > 0:
            src_region_mask = src_region_mask.filter(ImageFilter.GaussianBlur(radius=feather // 2))

        output = Image.composite(canvas, gen_img, src_region_mask)
        method = "inpaint"

    except Exception as e:
        logger.warning(f"Inpainting failed: {e}, falling back to img2img")
        # Fallback: simple canvas with source composited
        output = canvas

    elapsed = time.time() - t0
    buf = io.BytesIO()
    output.save(buf, format="JPEG", quality=95)
    return buf.getvalue(), seed, elapsed, target_w, target_h, method


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    try:
        model_key, prompt, negative, width, height, steps, guidance = _resolve_params(req)
    except ValueError as e:
        raise HTTPException(400, str(e))

    seed = req.seed if req.seed is not None else int(time.time()) % (2**32)

    try:
        loop = asyncio.get_event_loop()
        img_bytes, seed, elapsed, out_w, out_h = await loop.run_in_executor(
            executor, _run_pipeline, model_key, prompt, negative, width, height,
            steps, guidance, seed, req.upscale or False,
            req.upscale_target_width, req.upscale_target_height
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Generation failed: {e}")

    b64 = base64.b64encode(img_bytes).decode()
    return GenerateResponse(
        image_base64=b64, seed=seed, elapsed_seconds=elapsed,
        model=model_key, width=out_w, height=out_h,
        style_preset=req.style_preset
    )


@app.post("/generate/image")
async def generate_image(req: GenerateRequest):
    """Returns raw PNG image bytes."""
    try:
        model_key, prompt, negative, width, height, steps, guidance = _resolve_params(req)
    except ValueError as e:
        raise HTTPException(400, str(e))

    seed = req.seed if req.seed is not None else int(time.time()) % (2**32)

    try:
        loop = asyncio.get_event_loop()
        img_bytes, seed, elapsed, out_w, out_h = await loop.run_in_executor(
            executor, _run_pipeline, model_key, prompt, negative, width, height,
            steps, guidance, seed, req.upscale or False,
            req.upscale_target_width, req.upscale_target_height
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

    return Response(content=img_bytes, media_type="image/png",
                    headers={"X-Seed": str(seed), "X-Elapsed": f"{elapsed:.1f}",
                             "X-Model": model_key, "X-Size": f"{out_w}x{out_h}"})


@app.post("/outpaint", response_model=OutpaintResponse)
async def outpaint(req: OutpaintRequest):
    """Extend an image beyond its borders."""
    if not req.image_base64:
        raise HTTPException(400, "image_base64 is required")
    if req.target_width < 64 or req.target_height < 64:
        raise HTTPException(400, "target dimensions must be at least 64px")

    req_data = req.model_dump()

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _run_outpaint, req_data)
        img_bytes, seed, elapsed, out_w, out_h, method = result
    except Exception as e:
        logger.error(f"Outpaint failed: {e}", exc_info=True)
        raise HTTPException(500, f"Outpainting failed: {e}")

    b64 = base64.b64encode(img_bytes).decode()
    return OutpaintResponse(
        image_base64=b64, seed=seed, elapsed_seconds=elapsed,
        width=out_w, height=out_h, method=method
    )


from fastapi import Query

@app.post("/upscale")
async def upscale_endpoint(
    image: UploadFile = File(...),
    target_width: Optional[int] = Query(None),
    target_height: Optional[int] = Query(None),
):
    """Upscale an uploaded image using Real-ESRGAN."""
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(executor, _upscale_image, img, target_width, target_height)
    except Exception as e:
        raise HTTPException(500, f"Upscaling failed: {e}")

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png",
                    headers={"X-Size": f"{result.width}x{result.height}"})


@app.get("/models")
async def list_models():
    """List available models and their status."""
    result = {}
    for key, config in MODELS.items():
        local_exists = os.path.exists(config["local_path"])
        loaded = key in pipelines
        result[key] = {
            "description": config["description"],
            "available": True,  # Can always download from HuggingFace
            "local": local_exists,
            "loaded": loaded,
            "default_steps": config["default_steps"],
            "default_guidance": config["default_guidance"],
        }
    return result


@app.get("/styles")
async def list_styles():
    """List available style presets."""
    return {k: {"prompt_prefix": v["prompt_prefix"], "negative": v["negative"]}
            for k, v in STYLE_PRESETS.items()}


@app.get("/health")
async def health():
    loaded = list(pipelines.keys())
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_available else 0
    vram_used = torch.cuda.memory_allocated() / 1024**3 if gpu_available else 0
    upscaler_ready = os.path.exists("/data/models/upscaler/RealESRGAN_x4plus.pth")
    inpaint_ready = os.path.exists(INPAINT_LOCAL_PATH) or inpaint_pipeline is not None

    return {
        "status": "ok" if loaded else "no_models_loaded",
        "device": DEVICE,
        "dtype": str(DTYPE),
        "gpu": gpu_name,
        "vram_total_gb": round(vram_total, 1),
        "vram_used_gb": round(vram_used, 1),
        "loaded_models": loaded,
        "upscaler": upscaler_ready,
        "inpaint_model": inpaint_ready,
    }
