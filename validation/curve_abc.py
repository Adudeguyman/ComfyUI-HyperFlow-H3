"""Fixed-input A/B/C generation using local ComfyUI models, without a server."""
import argparse
import copy
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1])]

import torch
import comfy.cli_args

comfy.cli_args.args.disable_fast_disk = True
comfy.cli_args.args.disable_comfy_compiler = True
comfy.cli_args.args.disable_dynamic_vram = True
comfy.cli_args.args.reserve_vram = 8.0

import comfy.model_management
import comfy.samplers
import comfy.sd
import comfy.utils
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
from hyperflow_h3.nodes import _apply

PROMPT = ("A red ceramic kettle on a wooden kitchen counter gently releases steam. "
          "A static camera, warm daylight from a window, realistic details. "
          "Quiet room ambience with a soft kettle hiss. No music, no speech.")


def prepare(directory):
    clip = comfy.sd.load_clip([str(ROOT / "models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")],
                              clip_type=comfy.sd.CLIPType.MINIMAX, disable_dynamic=True)
    positive, latent = MiniMaxH3ImageToVideo.execute(clip, None, PROMPT, 768, 768, 124)
    noise = Noise_RandomNoise(42).generate_noise(latent)
    torch.save({"positive": positive, "latent": latent["samples"].unbind(),
                "noise": noise.unbind(), "seed": 42, "prompt": PROMPT}, directory / "inputs.pt")
    print("Saved shared conditioning, initial latents and noise", flush=True)


def generate(directory, family, variant):
    inputs = torch.load(directory / "inputs.pt", weights_only=False)
    filename = f"minimax_h3_{family}{'_pruned' if variant != 'a' else ''}_int8_convrot.safetensors"
    model = comfy.sd.load_diffusion_model(str(ROOT / "models/diffusion_models" / filename), disable_dynamic=True)
    adapter = ROOT / "models/hyperflow" / ("custom_node_hyperflow_8step_v1.0_comfyui"
                                          + ("_pruned" if variant != "a" else "") + ".safetensors")
    model, sigmas = _apply(model, str(adapter), 1., "bypass", True,
                           experimental_curve_refit=variant == "c")
    if variant == "c" and "diffusion_model.time_embedder" not in model.object_patches:
        raise RuntimeError("C must actually install the refit, not silently compare B twice")
    class RejectFallback(logging.Handler):
        def emit(self, record):
            if "curve refit disabled" in record.getMessage():
                raise RuntimeError("C fell back to backbone-only: " + record.getMessage())
    if variant == "c":
        logging.getLogger("comfy.hyperflow").addHandler(RejectFallback())
    guider = Guider_Basic(model)
    guider.set_conds(copy.deepcopy(inputs["positive"]))
    latent, noise = NestedTensor(inputs["latent"]), NestedTensor(inputs["noise"])
    started = time.monotonic()

    def progress(step, denoised, x, total):
        print(f"{family}/{variant} step {step + 1}/{total} elapsed {time.monotonic() - started:.1f}s", flush=True)

    output = guider.sample(noise, latent, comfy.samplers.sampler_object("euler"), sigmas,
                           callback=progress, seed=inputs["seed"])
    torch.save({"video": output.unbind()[0].cpu(), "audio": output.unbind()[1].cpu(),
                "elapsed": time.monotonic() - started, "checkpoint": filename,
                "variant": variant, "sigmas": sigmas}, directory / f"{family}_{variant}.pt")
    print("Saved sampled latents", flush=True)


def decode(directory, family, variant):
    import av
    import numpy as np
    from PIL import Image
    latent = torch.load(directory / f"{family}_{variant}.pt", weights_only=True)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(ROOT / "models/vae/minimax_h3_video_vae_fp16.safetensors")))
    frames = vae.decode(latent["video"])
    print("Decoded", frames.shape, flush=True)
    frames = frames.reshape(-1, *frames.shape[-3:])
    rgb = (frames.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    with av.open(str(directory / f"{family}_{variant}.mp4"), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height = rgb.shape[2], rgb.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for frame in rgb:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    for i in (0, len(rgb) // 4, len(rgb) // 2, 3 * len(rgb) // 4, len(rgb) - 1):
        Image.fromarray(rgb[i]).save(directory / f"{family}_{variant}_{i:03}.png")


def compare(directory, family):
    samples = {v: torch.load(directory / f"{family}_{v}.pt", weights_only=True) for v in "abc"}
    result = {}
    for stream in ("video", "audio"):
        ref = samples["a"][stream].float().flatten()
        result[stream] = {}
        for v in "bc":
            actual = samples[v][stream].float().flatten()
            result[stream][v] = {
                "rmse": (actual-ref).square().mean().sqrt().item(),
                "relative_l2": ((actual-ref).norm()/ref.norm()).item(),
                "cosine": torch.nn.functional.cosine_similarity(actual[None], ref[None]).item()}
    (directory / f"{family}_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "generate", "decode", "compare"))
    parser.add_argument("--directory", default="validation/abc")
    parser.add_argument("--family", choices=("fl2va", "ref2va"), default="fl2va")
    parser.add_argument("--variant", choices=list("abc"), default="a")
    args = parser.parse_args()
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    with torch.inference_mode():
        if args.action == "prepare": prepare(directory)
        elif args.action == "generate": generate(directory, args.family, args.variant)
        elif args.action == "decode": decode(directory, args.family, args.variant)
        else: compare(directory, args.family)
