import os, io, base64, time, logging, asyncio, json, sys

# Monkey-patch for basicsr/torchvision compatibility
try:
    import torchvision.transforms.functional as _F
    sys.modules["torchvision.transforms.functional_tensor"] = _F
except ImportError:
    pass
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from PIL import Image, ImageFilter
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image-gen")

# Global state
pipelines: Dict[str, Any] = {}
inpaint_pipeline = None
upscaler = None
executor = ThreadPoolExecutor(max_workers=1)

# Model registry
MODELS = {
    "sd15": {
        "path": "/data/models/sd-v1-5-openvino",
        "type": "sd15",
        "description": "Stable Diffusion 1.5 (OpenVINO)",
        "default_steps": 20,
        "default_guidance": 7.5,
    },
    "dreamshaper": {
        "path": "/data/models/lcm-dreamshaper-v7-int8-ov",
        "type": "lcm",
        "description": "LCM DreamShaper v7 INT8 (fast, 4 steps)",
        "default_steps": 4,
        "default_guidance": 1.0,
    },
    "sdxl-turbo": {
        "path": "/data/models/sdxl-turbo-openvino-8bit",
        "type": "sdxl",
        "description": "SDXL Turbo INT8 (high quality, 4 steps)",
        "default_steps": 4,
        "default_guidance": 0.0,
    },
}

# Inpainting model (for outpainting)
INPAINT_MODEL_PATH = "/data/models/sd15-inpainting-ov"

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
    "wide": (768, 432),       # ~16:9 for SD 1.5
    "ultrawide": (768, 320),  # ~21:9 for SD 1.5
    "portrait": (432, 768),
}

ASPECT_RATIOS_SDXL = {
    "square": (1024, 1024),
    "wide": (1024, 576),      # ~16:9
    "ultrawide": (1024, 448), # ~21:9 (must be multiple of 64)
    "portrait": (576, 1024),
}

# Default negative prompt with embeddings
DEFAULT_NEGATIVE = "ugly, blurry, low quality, deformed, disfigured, bad anatomy, bad hands, missing fingers, extra fingers, watermark, text"


def load_pipeline(model_key: str):
    """Load a pipeline by model key."""
    if model_key in pipelines:
        return pipelines[model_key]

    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    config = MODELS[model_key]
    model_path = config["path"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info(f"Loading model '{model_key}' from {model_path}...")

    if config["type"] == "sdxl":
        from optimum.intel import OVStableDiffusionXLPipeline
        pipe = OVStableDiffusionXLPipeline.from_pretrained(model_path, compile=False)
    elif config["type"] == "lcm":
        from optimum.intel import OVLatentConsistencyModelPipeline
        pipe = OVLatentConsistencyModelPipeline.from_pretrained(model_path, compile=False)
    else:
        from optimum.intel import OVStableDiffusionPipeline
        pipe = OVStableDiffusionPipeline.from_pretrained(model_path, compile=False)

    device = os.environ.get("OV_DEVICE", "CPU")
    pipe.to(device)
    pipe.compile()
    pipelines[model_key] = pipe
    logger.info(f"Model '{model_key}' loaded successfully on {device}")
    return pipe


def load_inpaint_pipeline_fn():
    """Load the SD 1.5 inpainting pipeline."""
    global inpaint_pipeline
    if inpaint_pipeline is not None:
        return inpaint_pipeline

    from optimum.intel import OVStableDiffusionInpaintPipeline

    if os.path.exists(INPAINT_MODEL_PATH):
        logger.info(f"Loading inpainting model from {INPAINT_MODEL_PATH}...")
        pipe = OVStableDiffusionInpaintPipeline.from_pretrained(INPAINT_MODEL_PATH, compile=False)
    else:
        # Download and convert on-demand (slow, ~10+ min first time)
        logger.info("Inpainting model not found locally, downloading and converting...")
        pipe = OVStableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", export=True, compile=False
        )
        logger.info(f"Saving inpainting model to {INPAINT_MODEL_PATH}...")
        os.makedirs(INPAINT_MODEL_PATH, exist_ok=True)
        pipe.save_pretrained(INPAINT_MODEL_PATH)

    device = os.environ.get("OV_DEVICE", "CPU")
    pipe.to(device)
    pipe.compile()
    inpaint_pipeline = pipe
    logger.info("Inpainting pipeline loaded successfully")
    return pipe


def load_upscaler():
    """Load Real-ESRGAN upscaler."""
    global upscaler
    if upscaler is not None:
        return upscaler

    model_path = "/data/models/upscaler/RealESRGAN_x4plus.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Real-ESRGAN model not found")

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
            half=False,  # CPU mode
        )
        logger.info("Real-ESRGAN upscaler loaded")
        return upscaler
    except ImportError:
        logger.warning("Real-ESRGAN not installed, upscaling unavailable")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load default model on startup
    default_model = os.environ.get("DEFAULT_MODEL", "sd15")
    try:
        load_pipeline(default_model)
    except Exception as e:
        logger.error(f"Failed to load default model '{default_model}': {e}")
        # Try sd15 as fallback
        if default_model != "sd15":
            try:
                load_pipeline("sd15")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")

    # Pre-load inpainting model if available
    if os.path.exists(INPAINT_MODEL_PATH):
        try:
            load_inpaint_pipeline_fn()
        except Exception as e:
            logger.warning(f"Could not pre-load inpainting model: {e}")

    yield
    pipelines.clear()
    inpaint_pipeline = None


app = FastAPI(title="Image Generation API", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    model: Optional[str] = None  # sd15, dreamshaper, sdxl-turbo
    style_preset: Optional[str] = None  # photorealistic, anime, landscape, scifi, cute-dog
    aspect_ratio: Optional[str] = None  # square, wide, ultrawide, portrait
    upscale: Optional[bool] = False
    upscale_target_width: Optional[int] = None  # target width for upscaling
    upscale_target_height: Optional[int] = None  # target height for upscaling


class GenerateResponse(BaseModel):
    image_base64: str
    seed: int
    elapsed_seconds: float
    model: str
    width: int
    height: int
    style_preset: Optional[str] = None


class OutpaintRequest(BaseModel):
    image_base64: str  # base64-encoded source image
    target_width: int = 1920
    target_height: int = 1080
    prompt: str = ""
    negative_prompt: Optional[str] = None
    placement: str = "center"  # center, left, right, top, bottom
    model: Optional[str] = "sd15"
    steps: Optional[int] = 25
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None
    # Blending options
    blend_feather: int = 40  # feather width in pixels for seamless blending


class OutpaintResponse(BaseModel):
    image_base64: str
    seed: int
    elapsed_seconds: float
    width: int
    height: int
    method: str  # "inpaint" or "img2img_blend"


def _resolve_params(req: GenerateRequest):
    """Resolve all generation parameters with defaults."""
    model_key = req.model or os.environ.get("DEFAULT_MODEL", "sd15")
    if model_key not in MODELS:
        # Check if model exists on disk as a path
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

    config = MODELS[model_key]
    is_sdxl = config["type"] == "sdxl"
    ar_map = ASPECT_RATIOS_SDXL if is_sdxl else ASPECT_RATIOS

    # Resolve dimensions
    if req.aspect_ratio and req.aspect_ratio in ar_map:
        width, height = ar_map[req.aspect_ratio]
    elif req.width and req.height:
        width, height = req.width, req.height
    else:
        width = req.width or (1024 if is_sdxl else 512)
        height = req.height or (1024 if is_sdxl else 512)

    # Resolve steps and guidance
    steps = req.steps or config["default_steps"]
    guidance = req.guidance_scale if req.guidance_scale is not None else config["default_guidance"]

    # Build prompt with style preset
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
    # Convert PIL to cv2
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    output, _ = up.enhance(img_bgr, outscale=4)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(output_rgb)

    # Resize to target if specified
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

    np.random.seed(seed)
    t0 = time.time()

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
    )

    result = pipe(**kwargs)
    img = result.images[0]
    gen_elapsed = time.time() - t0
    logger.info(f"Generated {width}x{height} in {gen_elapsed:.1f}s with {model_key}: {prompt[:80]}")

    # Upscale if requested
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
    """
    Calculate where to place the source in the target canvas.
    Returns (x, y) offset of source top-left corner.
    Also returns the scaled (src_w, src_h) to fit within target.
    """
    # Scale source to fit within target (maintaining aspect ratio)
    scale = min(target_w / src_w, target_h / src_h)
    new_src_w = int(src_w * scale)
    new_src_h = int(src_h * scale)

    # Calculate position based on placement
    if placement == "center":
        x = (target_w - new_src_w) // 2
        y = (target_h - new_src_h) // 2
    elif placement == "left":
        x = 0
        y = (target_h - new_src_h) // 2
    elif placement == "right":
        x = target_w - new_src_w
        y = (target_h - new_src_h) // 2
    elif placement == "top":
        x = (target_w - new_src_w) // 2
        y = 0
    elif placement == "bottom":
        x = (target_w - new_src_w) // 2
        y = target_h - new_src_h
    else:
        x = (target_w - new_src_w) // 2
        y = (target_h - new_src_h) // 2

    return x, y, new_src_w, new_src_h


def _create_outpaint_canvas(source_img, target_w, target_h, placement, feather=40):
    """
    Create padded canvas and mask for outpainting.
    
    Returns:
        canvas: RGB image at target dimensions with source placed + new areas filled
        mask: L-mode image (255=generate/white, 0=preserve/black)
        src_x, src_y: position of source in canvas
        new_src_w, new_src_h: scaled source dimensions
    """
    src_w, src_h = source_img.size

    # Calculate placement
    src_x, src_y, new_src_w, new_src_h = _calc_placement(
        src_w, src_h, target_w, target_h, placement
    )

    # Resize source to scaled dimensions
    scaled_source = source_img.resize((new_src_w, new_src_h), Image.LANCZOS)

    # --- Create canvas filled with edge-reflected content ---
    # Start with a neutral gray canvas
    canvas = Image.new("RGB", (target_w, target_h), (128, 128, 128))

    # Fill new areas with edge-extended/mirrored content for better context
    # Left extension
    if src_x > 0:
        left_strip = scaled_source.crop((0, 0, min(src_x, new_src_w), new_src_h))
        left_strip = left_strip.transpose(Image.FLIP_LEFT_RIGHT)
        # Tile/stretch to fill
        left_fill = left_strip.resize((src_x, new_src_h), Image.LANCZOS)
        canvas.paste(left_fill, (0, src_y))

    # Right extension
    right_start = src_x + new_src_w
    if right_start < target_w:
        right_w = target_w - right_start
        right_strip = scaled_source.crop((max(0, new_src_w - right_w), 0, new_src_w, new_src_h))
        right_strip = right_strip.transpose(Image.FLIP_LEFT_RIGHT)
        right_fill = right_strip.resize((right_w, new_src_h), Image.LANCZOS)
        canvas.paste(right_fill, (right_start, src_y))

    # Top extension
    if src_y > 0:
        top_strip = scaled_source.crop((0, 0, new_src_w, min(src_y, new_src_h)))
        top_strip = top_strip.transpose(Image.FLIP_TOP_BOTTOM)
        top_fill = top_strip.resize((new_src_w, src_y), Image.LANCZOS)
        canvas.paste(top_fill, (src_x, 0))

    # Bottom extension
    bottom_start = src_y + new_src_h
    if bottom_start < target_h:
        bottom_h = target_h - bottom_start
        bottom_strip = scaled_source.crop((0, max(0, new_src_h - bottom_h), new_src_w, new_src_h))
        bottom_strip = bottom_strip.transpose(Image.FLIP_TOP_BOTTOM)
        bottom_fill = bottom_strip.resize((new_src_w, bottom_h), Image.LANCZOS)
        canvas.paste(bottom_fill, (src_x, bottom_start))

    # Blur the extended areas for smoother blending
    blurred_canvas = canvas.filter(ImageFilter.GaussianBlur(radius=20))
    # Keep original areas from before-blur
    canvas.paste(blurred_canvas, (0, 0))
    # Re-paste source (sharp)
    canvas.paste(scaled_source, (src_x, src_y))

    # --- Create mask ---
    # White (255) = areas to generate, Black (0) = areas to preserve
    mask = Image.new("L", (target_w, target_h), 255)  # All white initially
    # Black out the source area (preserve it)
    source_mask_region = Image.new("L", (new_src_w, new_src_h), 0)
    mask.paste(source_mask_region, (src_x, src_y))

    # Feather the mask edges for smoother transition
    if feather > 0:
        # Invert, blur, invert back — creates soft transition at edges
        mask_inv = Image.fromarray(255 - np.array(mask))
        mask_inv_blurred = mask_inv.filter(ImageFilter.GaussianBlur(radius=feather))
        mask = Image.fromarray(255 - np.array(mask_inv_blurred))

    return canvas, mask, src_x, src_y, new_src_w, new_src_h


def _run_outpaint(req_data: dict) -> tuple:
    """
    Core outpainting logic (runs in executor thread).
    Returns: (img_bytes, seed, elapsed, width, height, method)
    """
    t0 = time.time()

    # Decode source image
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

    np.random.seed(seed)

    # Create padded canvas and mask
    canvas, mask, src_x, src_y, new_src_w, new_src_h = _create_outpaint_canvas(
        source_img, target_w, target_h, placement, feather
    )

    # Determine processing size — SD 1.5 works best at 512 or 768
    # Keep aspect ratio, target roughly 768 on longest side
    proc_w, proc_h = _clamp_to_pipeline_size(target_w, target_h, max_dim=768)

    method = "img2img_blend"

    if os.path.exists(INPAINT_MODEL_PATH) or inpaint_pipeline is not None:
        # --- Proper inpainting with dedicated model ---
        try:
            pipe = load_inpaint_pipeline_fn()

            # Resize canvas and mask to processing size
            proc_canvas = canvas.resize((proc_w, proc_h), Image.LANCZOS)
            proc_mask = mask.resize((proc_w, proc_h), Image.NEAREST)

            logger.info(f"Running inpainting at {proc_w}x{proc_h}, steps={steps}")

            result = pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=proc_canvas,
                mask_image=proc_mask,
                width=proc_w,
                height=proc_h,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
            gen_img = result.images[0]

            # Scale generated result back to target size
            gen_img = gen_img.resize((target_w, target_h), Image.LANCZOS)

            # Composite original source back (pixel-perfect preservation)
            # Use scaled source as authoritative for the preserved region
            scaled_source = source_img.resize((new_src_w, new_src_h), Image.LANCZOS)

            # Create feathered composite mask for source area
            src_region_mask = Image.new("L", (target_w, target_h), 0)
            src_region_inner = Image.new("L", (new_src_w, new_src_h), 255)
            src_region_mask.paste(src_region_inner, (src_x, src_y))
            if feather > 0:
                src_region_mask = src_region_mask.filter(ImageFilter.GaussianBlur(radius=feather // 2))

            # Composite: original source over generated background
            output = Image.composite(canvas, gen_img, src_region_mask)
            method = "inpaint"

            logger.info(f"Inpainting complete in {time.time()-t0:.1f}s")

        except Exception as e:
            logger.warning(f"Inpainting pipeline failed: {e}, falling back to img2img blend")
            output = _fallback_img2img_outpaint(canvas, mask, source_img, src_x, src_y,
                                                 new_src_w, new_src_h, target_w, target_h,
                                                 proc_w, proc_h, prompt, negative, steps,
                                                 guidance, seed, feather)
    else:
        # --- Fallback: img2img with blending ---
        logger.info("Inpainting model not available, using img2img blend fallback")
        output = _fallback_img2img_outpaint(canvas, mask, source_img, src_x, src_y,
                                             new_src_w, new_src_h, target_w, target_h,
                                             proc_w, proc_h, prompt, negative, steps,
                                             guidance, seed, feather)

    elapsed = time.time() - t0
    buf = io.BytesIO()
    output.save(buf, format="JPEG", quality=95)
    return buf.getvalue(), seed, elapsed, target_w, target_h, method


def _clamp_to_pipeline_size(w, h, max_dim=768):
    """Scale dimensions to fit within max_dim, keeping aspect ratio and aligning to 64."""
    scale = min(max_dim / w, max_dim / h, 1.0)
    new_w = max(64, int(w * scale / 64) * 64)
    new_h = max(64, int(h * scale / 64) * 64)
    return new_w, new_h


def _fallback_img2img_outpaint(canvas, mask, source_img, src_x, src_y,
                                new_src_w, new_src_h, target_w, target_h,
                                proc_w, proc_h, prompt, negative, steps,
                                guidance, seed, feather):
    """
    Outpaint fallback using img2img pipeline.
    
    Approach: Run img2img on the padded canvas at high strength (0.85) so
    the AI generates new content in the empty areas, then composite the
    original source back with feathered edges.
    """
    from optimum.intel import OVStableDiffusionImg2ImgPipeline

    pipe_key = "sd15-img2img"
    if pipe_key not in pipelines:
        model_path = MODELS["sd15"]["path"]
        logger.info(f"Loading img2img pipeline from {model_path}...")
        pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained(model_path, compile=False)
        device = os.environ.get("OV_DEVICE", "CPU")
        pipe.to(device)
        pipe.compile()
        pipelines[pipe_key] = pipe
    else:
        pipe = pipelines[pipe_key]

    # Resize canvas to processing dimensions
    proc_canvas = canvas.resize((proc_w, proc_h), Image.LANCZOS)

    logger.info(f"Running img2img outpaint at {proc_w}x{proc_h}, steps={steps}, strength=0.85")

    result = pipe(
        prompt=prompt if prompt else "high quality, detailed",
        negative_prompt=negative,
        image=proc_canvas,
        strength=0.85,
        num_inference_steps=max(steps, 20),
        guidance_scale=guidance if guidance > 0 else 7.5,
    )
    gen_img = result.images[0]

    # Scale generated image to target size
    gen_img = gen_img.resize((target_w, target_h), Image.LANCZOS)

    # Composite original source back with feathered mask
    scaled_source = source_img.resize((new_src_w, new_src_h), Image.LANCZOS)

    # Create hard source region mask, then feather it
    src_region_mask = Image.new("L", (target_w, target_h), 0)
    src_region_inner = Image.new("L", (new_src_w, new_src_h), 255)
    src_region_mask.paste(src_region_inner, (src_x, src_y))
    if feather > 0:
        src_region_mask = src_region_mask.filter(ImageFilter.GaussianBlur(radius=feather))

    # Create full-canvas version of source for compositing
    canvas_with_src = canvas.copy()
    canvas_with_src.paste(scaled_source, (src_x, src_y))

    # Composite: source area over generated image
    output = Image.composite(canvas_with_src, gen_img, src_region_mask)
    return output


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
    """
    Extend an image beyond its borders (outpainting).
    
    Takes a source image and expands it to the target dimensions by generating
    new content in the surrounding areas while preserving the original.
    
    Uses SD 1.5 inpainting model if available at /data/models/sd15-inpainting-ov,
    otherwise falls back to img2img with feathered blending.
    """
    if not req.image_base64:
        raise HTTPException(400, "image_base64 is required")

    if req.target_width < 64 or req.target_height < 64:
        raise HTTPException(400, "target dimensions must be at least 64px")

    if req.placement not in ("center", "left", "right", "top", "bottom"):
        raise HTTPException(400, "placement must be one of: center, left, right, top, bottom")

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
        result = await loop.run_in_executor(
            executor, _upscale_image, img, target_width, target_height
        )
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
        exists = os.path.exists(config["path"])
        loaded = key in pipelines
        result[key] = {
            "description": config["description"],
            "available": exists,
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
    available = [k for k, v in MODELS.items() if os.path.exists(v["path"])]
    upscaler_ready = os.path.exists("/data/models/upscaler/RealESRGAN_x4plus.pth")
    inpaint_ready = os.path.exists(INPAINT_MODEL_PATH)
    return {
        "status": "ok" if loaded else "no_models_loaded",
        "loaded_models": loaded,
        "available_models": available,
        "upscaler": upscaler_ready,
        "inpaint_model": inpaint_ready,
        "outpaint": "ready" if inpaint_ready else "fallback_mode (img2img blend)",
    }
