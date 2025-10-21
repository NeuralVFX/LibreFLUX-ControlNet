#!/usr/bin/env python
import os, json, argparse
from typing import Optional, Tuple
import torch
from PIL import Image
import random
from src.pipelines.pipeline_flux_controlnet_st import FluxControlNetPipeline
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
from src.models.transformer import FluxTransformer2DModel
from src.models.controlnet_flux import FluxControlNetModel as ControlNetFlux
from optimum.quanto import freeze, quantize, qint8, qint4
from src.util_pil import make_alpha_all_ones

try:
    from controlnet_aux import LineartDetector
    HAS_LINEART = True
except Exception:
    HAS_LINEART = False

def _round_to_multiple(x: int, k: int) -> int:
    return max(k, int(round(x / k) * k))

# NEW: you were calling this but hadn’t defined it
def _suggest_hw_like_input(img_w: int, img_h: int, multiple: int = 16, max_dim: Optional[int] = None):
    w, h = img_w, img_h
    if max_dim is not None and max(w, h) > max_dim:
        s = max_dim / float(max(w, h))
        w = int(w * s); h = int(h * s)
    # snap to multiples of 16
    return _round_to_multiple(h, multiple), _round_to_multiple(w, multiple)

def concat_on_longest_edge(img_a: Image.Image, img_b: Image.Image) -> Image.Image:
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.BILINEAR)
    w, h = img_a.size
    if w >= h:
        canvas = Image.new("RGB", (w + w, h))
        canvas.paste(img_a, (0, 0))
        canvas.paste(img_b, (w, 0))
    else:
        canvas = Image.new("RGB", (w, h + h))
        canvas.paste(img_a, (0, 0))
        canvas.paste(img_b, (0, h))
    return canvas

def _str_to_torch_dtype(s):
    if not isinstance(s, str): return s
    s = s.lower()
    m = {
        "fp16": torch.float16, "float16": torch.float16,
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp32": torch.float32, "float32": torch.float32,
    }
    if s not in m:
        raise ValueError(f"Unsupported --dtype '{s}'. Use one of: {list(m.keys())}.")
    return m[s]

def _ensure_supported_dtype(d):
    if d is torch.bfloat16:
        ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if not ok:
            print("[warn] BF16 not supported; falling back to FP16.")
            return torch.float16
    return d

def load_pipeline(
    pretrained_model_name_or_path: str,
    controlnet_path: str,
    revision: str = None,
    variant: str = None,
    dtype: str = "fp16",
    device='cuda'
):
    torch_dtype = _ensure_supported_dtype(_str_to_torch_dtype(dtype))

    # --- load components exactly like your working script ---
    tok1 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path,
                                         subfolder="tokenizer", revision=revision)
    tok2 = T5TokenizerFast.from_pretrained(pretrained_model_name_or_path,
                                           subfolder="tokenizer_2", revision=revision)

    te1 = CLIPTextModel.from_pretrained(pretrained_model_name_or_path,
                                        subfolder="text_encoder", revision=revision,
                                        variant=variant, torch_dtype=torch_dtype)
    te2 = T5EncoderModel.from_pretrained(pretrained_model_name_or_path,
                                         subfolder="text_encoder_2", revision=revision,
                                         variant=variant, torch_dtype=torch_dtype)

    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path,
                                        subfolder="vae", revision=revision,
                                        variant=variant, torch_dtype=torch_dtype)
    tr  = FluxTransformer2DModel.from_pretrained(pretrained_model_name_or_path,
                                                 subfolder="transformer", revision=revision,
                                                 variant=variant, torch_dtype=torch_dtype)

    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path,
                                                          subfolder="scheduler")
    cn  = ControlNetFlux.from_pretrained(controlnet_path, torch_dtype=torch_dtype)

    pipe = FluxControlNetPipeline(
        scheduler=sch,
        vae=vae,
        text_encoder=te1,
        tokenizer=tok1,
        text_encoder_2=te2,
        tokenizer_2=tok2,
        transformer=tr,
        controlnet=cn,
    )

    # device placement (kept as in your working version)
    dev = torch.device(device)
    tr.to(dev, dtype=torch_dtype)
    cn.to(dev, dtype=torch_dtype)
    vae.to(dev, dtype=torch_dtype)
    te1.to(dev, dtype=torch_dtype)
    te2.to(dev, dtype=torch_dtype)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    return pipe

def apply_quantization(pipe: FluxControlNetPipeline,
                       device: torch.device,
                       quantize_all: bool,
                       quant_dtype: str,
                       selective_exclude_norms: bool):
    dtype_map = {"int8": qint8, "int4": qint4}
    qdtype = dtype_map.get(quant_dtype, qint8)

    transformer = pipe.transformer
    vae = pipe.vae
    te1 = pipe.text_encoder
    te2 = pipe.text_encoder_2

    transformer.to("cpu"); vae.to("cpu")
    if quantize_all:
        te1.to("cpu"); te2.to("cpu")

    if quantize_all:
        quantize(transformer, weights=qdtype)
        quantize(vae,         weights=qdtype)
        quantize(te1,         weights=qdtype)
        quantize(te2,         weights=qdtype)
        freeze(transformer); freeze(vae); freeze(te1); freeze(te2)
    else:
        exclude_list = [
            "*.norm", "*.norm1", "*.norm2", "*.norm2_context",
            "proj_out", "x_embedder", "norm_out", "context_embedder",
        ] if selective_exclude_norms else None
        quantize(transformer, weights=qdtype, exclude=exclude_list)
        quantize(vae,         weights=qdtype)
        freeze(transformer); freeze(vae)

    transformer.to(device); vae.to(device)
    if quantize_all:
        te1.to(device); te2.to(device)

def _prepare_control_image(path: str, *, device: str, auto_hw: bool, multiple: int, max_dim: Optional[int],
                           sketchify: bool, lineart):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing control image: {path}")
    raw = Image.open(path).convert("RGBA")
    raw = make_alpha_all_ones(raw)
    control_rgb = raw.convert("RGB")
    if auto_hw:
        h, w = _suggest_hw_like_input(control_rgb.width, control_rgb.height, multiple=multiple, max_dim=max_dim)
    else:
        h = w = _round_to_multiple(max(control_rgb.width, control_rgb.height), multiple)
    if sketchify and lineart is not None:
        cond = lineart(control_rgb).resize((w, h))
    else:
        cond = control_rgb.resize((w, h), Image.BILINEAR)
    ctrl_resized = control_rgb.resize((w, h), Image.BILINEAR)
    return cond, ctrl_resized, (h, w)

def run_single(
    pipe: FluxControlNetPipeline,
    image_path: str,
    prompt: str,
    output_dir: str,
    *,
    steps: int = 28,
    guidance_scale: float = 4.0,
    controlnet_conditioning_scale: float = 1.0,
    control_mode: int = None,
    auto_hw: bool = True,
    multiple: int = 16,
    max_dim: int = None,
    num_images_per_prompt: int = 1,
    negative_prompt: str = "blurry, bokeh, jpg",
    device: str = "cuda",
    seed: Optional[int] = None,
    sketchify_single: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    lineart = None
    if HAS_LINEART:
        try:
            lineart = LineartDetector.from_pretrained("lllyasviel/Annotators").to(device)
        except Exception:
            lineart = None

    gen_device = torch.device(device)
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    g = torch.Generator(device=gen_device).manual_seed(seed)

    cond, ctrl_resized, (h, w) = _prepare_control_image(
        image_path, device=device, auto_hw=auto_hw, multiple=multiple, max_dim=max_dim,
        sketchify=sketchify_single, lineart=lineart
    )

    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_image=cond,
            control_mode=control_mode,
            height=h, width=w,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=g,
            return_dict=True,
        )
    images = out.images

    base = os.path.splitext(os.path.basename(image_path))[0]
    for i, im in enumerate(images):
        if w >= h:
            canvas = Image.new("RGB", (w + w, h))
            canvas.paste(ctrl_resized, (0, 0))
            canvas.paste(im, (w, 0))
        else:
            canvas = Image.new("RGB", (w, h + h))
            canvas.paste(ctrl_resized, (0, 0))
            canvas.paste(im, (0, h))

        suffix = f"_{seed}" + (f"_{i}" if num_images_per_prompt > 1 else "")
        out_path = os.path.join(output_dir, f"{base}{suffix}.png")
        canvas.save(out_path)
        print(f"[saved] {out_path}")

def run_from_metadata(
    pipe: FluxControlNetPipeline,
    metadata_dir: str,
    output_dir: str,
    *,
    steps: int = 28,
    guidance_scale: float = 4.0,
    controlnet_conditioning_scale: float = 1.0,
    control_mode: int = None,
    auto_hw: bool = True,
    multiple: int = 16,
    max_dim: int = None,
    num_images_per_prompt: int = 1,
    negative_prompt: str = "blurry, bokeh, jpg",
    device: str = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)

    lineart = None
    if HAS_LINEART:
        try:
            lineart = LineartDetector.from_pretrained("lllyasviel/Annotators").to(device)
        except Exception:
            lineart = None

    meta = os.path.join(metadata_dir, "metadata.jsonl")
    if not os.path.exists(meta):
        raise FileNotFoundError(f"Missing {meta}")

    gen_device = torch.device(device)
    seed = random.randint(0, 2**31 - 1)
    g = torch.Generator(device=gen_device).manual_seed(seed)

    with open(meta, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            rec = json.loads(ln)
            fname      = rec["file"]
            prompt     = rec.get("prompt", "")
            sketchify  = bool(rec.get("sketchify", False))

            ctrl_path = os.path.join(metadata_dir, fname)
            if not os.path.exists(ctrl_path):
                print(f"[skip] missing control image: {ctrl_path}")
                continue

            raw = Image.open(ctrl_path).convert("RGBA")
            raw = make_alpha_all_ones(raw)
            control_rgb = raw.convert("RGB")

            if auto_hw:
                h, w = _suggest_hw_like_input(control_rgb.width, control_rgb.height, multiple=multiple, max_dim=max_dim)
            else:
                h = w = _round_to_multiple(max(control_rgb.width, control_rgb.height), multiple)

            if sketchify and lineart is not None:
                cond = lineart(control_rgb).resize((w, h))
            else:
                cond = control_rgb.resize((w, h), Image.BILINEAR)

            with torch.inference_mode():
                out = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    control_image=cond,
                    control_mode=control_mode,
                    height=h, width=w,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=g,
                    return_dict=True,
                )
            images = out.images

            base = os.path.splitext(os.path.basename(fname))[0]
            ctrl_resized = control_rgb.resize((w, h), Image.BILINEAR)
            for i, im in enumerate(images):
                if w >= h:
                    canvas = Image.new("RGB", (w + w, h))
                    canvas.paste(ctrl_resized, (0, 0))
                    canvas.paste(im, (w, 0))
                else:
                    canvas = Image.new("RGB", (w, h + h))
                    canvas.paste(ctrl_resized, (0, 0))
                    canvas.paste(im, (0, h))

                suffix = f"_{seed}" + (f"_{i}" if num_images_per_prompt > 1 else "")
                out_path = os.path.join(output_dir, f"{base}{suffix}.png")
                canvas.save(out_path)
                print(f"[saved] {out_path}")

def main():
    ap = argparse.ArgumentParser("Batch FLUX-ControlNet inference from metadata.jsonl (with optional quantization) + single-image mode")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--controlnet_path", required=True)

    # Metadata/batch mode
    ap.add_argument("--metadata_dir", required=False)
    ap.add_argument("--output_dir", required=True)

    # NEW: simple single-image mode
    ap.add_argument("--image", type=str, default=None, help="Path to a single control image (enables simple mode when combined with --prompt)")
    ap.add_argument("--prompt", type=str, default=None, help="Prompt text for simple mode")
    ap.add_argument("--seed", type=int, default=None, help="Optional seed for simple mode")
    ap.add_argument("--sketchify_single", action="store_true", help="Apply lineart detector to the single input image if available")

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance_scale", type=float, default=4.0)
    ap.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    ap.add_argument("--negative_prompt", type=str, default="blurry, bokeh, jpeg artifacts")
    ap.add_argument("--control_mode", type=int, default=None)
    ap.add_argument("--auto_hw", action="store_true")
    ap.add_argument("--multiple", type=int, default=16)
    ap.add_argument("--max_dim", type=int, default=None)
    ap.add_argument("--num_images_per_prompt", type=int, default=1)

    # quantization flags
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--legacy_quant", action="store_true")
    ap.add_argument("--quantize_dtype", choices=["int8", "int4"], default="int8")
    ap.add_argument("--exclude_norms", action="store_true")

    args = ap.parse_args()

    pipe = load_pipeline(
        pretrained_model_name_or_path=args.base_model,
        controlnet_path=args.controlnet_path,
        revision=None,
        variant=None,
        device=args.device,
        dtype=args.dtype,
    )

    if args.quantize:
        apply_quantization(
            pipe,
            torch.device(args.device),
            quantize_all=args.legacy_quant,
            quant_dtype=args.quantize_dtype,
            selective_exclude_norms=args.exclude_norms,
        )

    # NEW: single-image path if both provided
    if args.image is not None and args.prompt is not None:
        run_single(
            pipe,
            image_path=args.image,
            prompt=args.prompt,
            output_dir=args.output_dir,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            control_mode=args.control_mode,
            auto_hw=args.auto_hw,
            multiple=args.multiple,
            max_dim=args.max_dim,
            num_images_per_prompt=args.num_images_per_prompt,
            negative_prompt=args.negative_prompt,
            device=args.device,
            seed=args.seed,
            sketchify_single=args.sketchify_single,
        )
    else:
        if not args.metadata_dir:
            raise ValueError("Either provide --image and --prompt for simple mode, or provide --metadata_dir for batch mode.")
        run_from_metadata(
            pipe,
            metadata_dir=args.metadata_dir,
            output_dir=args.output_dir,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            control_mode=args.control_mode,
            auto_hw=args.auto_hw,
            multiple=args.multiple,
            max_dim=args.max_dim,
            num_images_per_prompt=args.num_images_per_prompt,
            negative_prompt=args.negative_prompt,
            device=args.device,
        )

if __name__ == "__main__":
    main()
