"""RunPod handler for LongCat-Video-Avatar-1.5 (audio-driven avatar video).

Adapted from meituan-longcat/LongCat-Video's run_demo_avatar_single_audio_to_video.py.
That script is a torchrun multi-process CLI tool; this collapses it to a
single-process, single-GPU worker that loads everything once at import time
and reuses it across jobs, the same shape as every other handler here.

Weights are not baked into the image or on a network volume -- they're
attached to the endpoint via RunPod's HF model cache (`--model-reference`),
which is host-distributed rather than pinned to one data center. This file
only has to bridge the gap between where that cache puts the files and the
sibling-directory layout the upstream code expects (see _link_weights).
"""

import base64
import glob
import io
import json
import math
import os
import tempfile
import uuid

import librosa
import numpy as np
import PIL.Image
import requests
import runpod
import torch
import torch.distributed as dist
from audio_separator.separator import Separator
from diffusers.utils import load_image
from transformers import AutoTokenizer, UMT5EncoderModel

from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
from longcat_video.audio_process.torch_utils import save_video_ffmpeg
from longcat_video.context_parallel import context_parallel_util
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video.modules.quantization import load_quantized_dit
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline

MODEL_TYPE = "avatar-v1.5"
HF_CACHE_HUB = "/runpod-volume/huggingface-cache/hub"
WEIGHTS_ROOT = "/tmp/weights"
NUM_FRAMES = 93
NUM_COND_FRAMES = 13
SAVE_FPS = 25
AUDIO_STRIDE = 1
NEGATIVE_PROMPT = (
    "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly "
    "drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)


def _snapshot_dir(repo_id: str) -> str:
    """Where --model-reference lands a cached HF repo, resolved without
    hardcoding the revision hash (it's pinned by the endpoint's :ref, not by
    this file, so whatever RunPod resolved is whatever is on disk)."""
    pattern = os.path.join(HF_CACHE_HUB, f"models--{repo_id.replace('/', '--')}", "snapshots", "*")
    hits = glob.glob(pattern)
    if not hits:
        raise RuntimeError(f"{repo_id} not found under {HF_CACHE_HUB} -- "
                            f"is the endpoint's --model-reference set for it?")
    return hits[0]


def _link_weights():
    """Bridge RunPod's HF-cache layout to the sibling-directory layout
    run_demo_avatar_single_audio_to_video.py expects: the base LongCat-Video
    repo (tokenizer/text_encoder/vae) as a directory literally named
    "LongCat-Video" next to the avatar checkpoint dir."""
    os.makedirs(WEIGHTS_ROOT, exist_ok=True)
    base_link = os.path.join(WEIGHTS_ROOT, "LongCat-Video")
    avatar_link = os.path.join(WEIGHTS_ROOT, "LongCat-Video-Avatar-1.5")
    if not os.path.islink(base_link):
        os.symlink(_snapshot_dir("meituan-longcat/LongCat-Video"), base_link)
    if not os.path.islink(avatar_link):
        os.symlink(_snapshot_dir("meituan-longcat/LongCat-Video-Avatar-1.5"), avatar_link)
    return avatar_link


def _init_single_process_dist():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl")
    context_parallel_util.init_context_parallel(context_parallel_size=1, global_rank=0, world_size=1)


def _load_pipeline():
    checkpoint_dir = _link_weights()
    base_dir = os.path.join(checkpoint_dir, "..", "LongCat-Video")

    tokenizer = AutoTokenizer.from_pretrained(base_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(base_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(base_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)

    print("[INFO] Loading INT8 quantized DiT model...")
    dit = load_quantized_dit(checkpoint_dir, subfolder="base_model_int8", cp_split_hw=context_parallel_util.get_optimal_split(1))
    distill_checkpoint_path = os.path.join(checkpoint_dir, "lora", "dmd_lora.safetensors")
    if os.path.exists(distill_checkpoint_path):
        dit.load_lora(distill_checkpoint_path, "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
        dit.enable_loras(["dmd"])

    audio_model_checkpoint_path = os.path.join(checkpoint_dir, "whisper-large-v3")
    audio_encoder = get_audio_encoder(audio_model_checkpoint_path, MODEL_TYPE).to(0)
    audio_feature_extractor = get_audio_feature_extractor(audio_model_checkpoint_path, MODEL_TYPE)

    vocal_separator_path = os.path.join(checkpoint_dir, "vocal_separator", "Kim_Vocal_2.onnx")
    audio_tmp_dir = "/tmp/audio_temp"
    os.makedirs(audio_tmp_dir, exist_ok=True)
    vocal_separator = Separator(
        output_dir=os.path.join(audio_tmp_dir, "vocals"),
        output_single_stem="vocals",
        model_file_dir=os.path.dirname(vocal_separator_path),
    )
    vocal_separator.load_model(os.path.basename(vocal_separator_path))

    pipe = LongCatVideoAvatarPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler,
        dit=dit, audio_encoder=audio_encoder, audio_feature_extractor=audio_feature_extractor,
        model_type=MODEL_TYPE,
    )
    pipe.to(0)
    return pipe, vocal_separator, audio_tmp_dir


_init_single_process_dist()
PIPE, VOCAL_SEPARATOR, AUDIO_TMP_DIR = _load_pipeline()
print("[INFO] LongCat-Video-Avatar-1.5 pipeline ready.")


def _fetch_to_file(ref: str, suffix: str) -> str:
    """A URL, a data URI, or bare base64 -- same shapes every adapter in
    this fleet already accepts -- written to a temp file on disk, since the
    upstream code reads audio/images from paths, not bytes."""
    path = f"/tmp/{uuid.uuid4().hex}{suffix}"
    if ref.startswith("http://") or ref.startswith("https://"):
        resp = requests.get(ref, timeout=120)
        resp.raise_for_status()
        data = resp.content
    else:
        data = base64.b64decode(ref.split(",", 1)[-1] if ref.startswith("data:") else ref)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _extract_vocal(raw_speech_path: str) -> str:
    outputs = VOCAL_SEPARATOR.separate(raw_speech_path)
    if not outputs:
        return raw_speech_path
    vocal_path = os.path.join(AUDIO_TMP_DIR, "vocals", outputs[0])
    target = f"/tmp/{uuid.uuid4().hex}_vocal.wav"
    os.replace(vocal_path, target)
    return target


def handler(job):
    inp = job["input"]
    prompt = inp.get("prompt")
    audio_ref = inp.get("audio")
    if not prompt or not audio_ref:
        return {"error": "'prompt' and 'audio' are required"}

    image_ref = inp.get("image")
    resolution = inp.get("resolution", "480p")
    if resolution not in ("480p", "720p"):
        return {"error": "resolution must be '480p' or '720p'"}
    num_segments = max(1, int(inp.get("num_segments", 1)))
    use_distill = bool(inp.get("use_distill", True))
    num_inference_steps = 8 if use_distill else int(inp.get("num_inference_steps", 50))
    guidance = 1.0 if use_distill else float(inp.get("guidance_scale", 4.0))
    stage_1 = "ai2v" if image_ref else "at2v"
    height, width = (480, 832) if resolution == "480p" else (768, 1280)

    raw_speech_path = _fetch_to_file(audio_ref, ".wav")
    generator = torch.Generator(device=0).manual_seed(int(inp.get("seed", 42)))

    try:
        vocal_path = _extract_vocal(raw_speech_path)
        generate_duration = NUM_FRAMES / SAVE_FPS + (num_segments - 1) * (NUM_FRAMES - NUM_COND_FRAMES) / SAVE_FPS
        speech_array, sr = librosa.load(vocal_path, sr=16000)
        pad = math.ceil((generate_duration - len(speech_array) / sr) * sr)
        if pad > 0:
            speech_array = np.append(speech_array, [0.0] * pad)

        full_audio_emb = PIPE.get_audio_embedding(speech_array, fps=SAVE_FPS * AUDIO_STRIDE, device=0,
                                                    sample_rate=sr, model_type=MODEL_TYPE)
        if torch.isnan(full_audio_emb).any():
            return {"error": "audio embedding produced NaNs -- check the input audio"}

        indices = torch.arange(5) - 2
        audio_start_idx = 0
        audio_end_idx = AUDIO_STRIDE * NUM_FRAMES
        centers = torch.arange(audio_start_idx, audio_end_idx, AUDIO_STRIDE).unsqueeze(1) + indices.unsqueeze(0)
        centers = torch.clamp(centers, min=0, max=full_audio_emb.shape[0] - 1)
        audio_emb = full_audio_emb[centers][None, ...].to(0)

        if stage_1 == "at2v":
            output, latent = PIPE.generate_at2v(
                prompt=prompt, negative_prompt=NEGATIVE_PROMPT, height=height, width=width,
                num_frames=NUM_FRAMES, num_inference_steps=num_inference_steps,
                text_guidance_scale=guidance, audio_guidance_scale=guidance,
                generator=generator, output_type="both", audio_emb=audio_emb, use_distill=use_distill,
            )
        else:
            image = load_image(_fetch_to_file(image_ref, ".png"))
            output, latent = PIPE.generate_ai2v(
                image=image, prompt=prompt, negative_prompt=NEGATIVE_PROMPT, resolution=resolution,
                num_frames=NUM_FRAMES, num_inference_steps=num_inference_steps,
                text_guidance_scale=guidance, audio_guidance_scale=guidance,
                generator=generator, output_type="both", audio_emb=audio_emb, use_distill=use_distill,
            )
        frame = output[0]
        video = [PIL.Image.fromarray((frame[i] * 255).astype(np.uint8)) for i in range(frame.shape[0])]
        width_px, height_px = video[0].size
        current_video, ref_latent, all_frames = video, latent[:, :, :1].clone(), video

        for seg in range(1, num_segments):
            audio_start_idx += AUDIO_STRIDE * (NUM_FRAMES - NUM_COND_FRAMES)
            audio_end_idx = audio_start_idx + AUDIO_STRIDE * NUM_FRAMES
            centers = torch.arange(audio_start_idx, audio_end_idx, AUDIO_STRIDE).unsqueeze(1) + indices.unsqueeze(0)
            centers = torch.clamp(centers, min=0, max=full_audio_emb.shape[0] - 1)
            audio_emb = full_audio_emb[centers][None, ...].to(0)

            output, latent = PIPE.generate_avc(
                video=current_video, video_latent=latent, prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
                height=height_px, width=width_px, num_frames=NUM_FRAMES, num_cond_frames=NUM_COND_FRAMES,
                num_inference_steps=num_inference_steps, text_guidance_scale=guidance, audio_guidance_scale=guidance,
                generator=generator, output_type="both", use_kv_cache=True, offload_kv_cache=False,
                enhance_hf=not use_distill, audio_emb=audio_emb, ref_latent=ref_latent,
                ref_img_index=int(inp.get("ref_img_index", 10)), mask_frame_range=int(inp.get("mask_frame_range", 3)),
                use_distill=use_distill,
            )
            frame = output[0]
            new_video = [PIL.Image.fromarray((frame[i] * 255).astype(np.uint8)) for i in range(frame.shape[0])]
            all_frames = all_frames + new_video[NUM_COND_FRAMES:]
            current_video = new_video

        out_dir = f"/tmp/out_{uuid.uuid4().hex}"
        os.makedirs(out_dir, exist_ok=True)
        save_video_ffmpeg(torch.from_numpy(np.array(all_frames)), os.path.join(out_dir, "result"),
                           raw_speech_path, fps=SAVE_FPS, quality=5)
        video_path = os.path.join(out_dir, "result.mp4")
        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
        return {"video": video_b64}
    finally:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


runpod.serverless.start({"handler": handler})
