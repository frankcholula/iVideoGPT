"""Dump action-conditioned rollouts + tokenizer-recon ceiling on held-out nwm_scene episodes for the shared scorer."""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from ivideogpt.transformer import HeadModelWithAction
from ivideogpt.vq_model import CompressiveVQModel

EVAL_DIR = os.environ.get("NWM_EVAL_DIR", "/mnt/Data/nwm-baselines/ogb_scene_ep_eval")
VIEW_IDX = int(os.environ.get("NWM_VIEW_IDX", "0"))  # view-axis index within the export
TOKENIZER = os.environ.get("NWM_TOKENIZER") or os.path.join(
    REPO, "log_vqgan/2026-08-05-12:51:49-nwm_scene_tokenizer_ft/checkpoint-50005/unwrapped_model")
CTX, SEG, TPD = 1, 16, 16


def newest_ckpt():
    cands = glob.glob(os.path.join(REPO, "log_trm/*nwm_scene_llama_ft*/checkpoints/checkpoint_*"))
    assert cands, "no transformer checkpoint found under log_trm"
    return max(cands, key=lambda p: int(p.rsplit("_", 1)[1]))


def down64(frames):
    f = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    f = F.interpolate(f, size=(64, 64), mode="bilinear", align_corners=False, antialias=True)
    return f.round().clamp(0, 255) / 255.0


def to_u8(x):
    return x[0].float().mul(255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()


def load_model(ckpt, vq, device):
    config = AutoConfig.from_pretrained(os.path.join(REPO, "configs/llama/config.json"))
    config.vocab_size = vq.num_vq_embeddings + vq.num_dyn_embeddings + 2
    llm = AutoModelForCausalLM.from_config(config)
    model = HeadModelWithAction(llm, action_dim=5, prelude_tokens_num=(256 + 1) * CTX - 1,
                                tokens_num_per_dyna=TPD, context=CTX, segment_length=SEG, model_type="llama")
    model.load_state_dict(load_file(os.path.join(ckpt, "model.safetensors")), strict=True)
    return model.to(device).eval()


@torch.no_grad()
def rollout(model, vq, ctx, actions, n_steps, device):
    # slide 1-ctx train-layout segments (<=15 new frames each); last decoded frame re-tokenized as next context
    frames = []
    for c0 in range(0, n_steps, SEG - CTX):
        n = min(SEG - CTX, n_steps - c0)
        model.segment_length = CTX + n
        tokens, _ = vq.tokenize(torch.cat([ctx, torch.zeros_like(ctx)], dim=1), CTX)
        act = torch.from_numpy(actions[c0:c0 + n]).float().unsqueeze(0).to(device)
        generated = model.generate(tokens[:, :CTX * (256 + 1)], do_sample=True, temperature=1.0,
                                   top_k=100, max_new_tokens=(1 + TPD) * n - 1, action=act)
        recon = vq.detokenize(generated, CTX).clamp(0, 1)
        frames.append(recon[:, 1:])
        ctx = recon[:, -1:]
    model.segment_length = SEG
    return torch.cat(frames, dim=1)


@torch.no_grad()
def tok_recon(vq, ctx, targets):
    tokens, _ = vq.tokenize(torch.cat([ctx, targets], dim=1), CTX)
    return vq.detokenize(tokens, CTX).clamp(0, 1)[:, 1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/mnt/Data/nwm-baselines/preds/ivideogpt.npz")
    ap.add_argument("--recon-out", default="/mnt/Data/nwm-baselines/preds/ivideogpt_tokrecon.npz")
    ap.add_argument("--ckpt", default=None, help="accelerate save_state dir; default newest under log_trm")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    device = "cuda"
    eps = sorted(glob.glob(os.path.join(EVAL_DIR, "ep*.npz")))[: a.limit]
    assert eps, f"no episodes in {EVAL_DIR}"
    ckpt = a.ckpt or os.environ.get("NWM_CKPT") or newest_ckpt()
    print(f"episodes={len(eps)} ckpt={ckpt}", flush=True)

    vq = CompressiveVQModel.from_pretrained(
        TOKENIZER, subfolder=None, revision=None, variant=None, use_safetensor=True,
        low_cpu_mem_usage=False, device_map=None).to(device).eval()
    assert vq.context_length == CTX
    model = load_model(ckpt, vq, device)

    preds, recons = [], []
    for i, ep in enumerate(eps):
        start = time.time()
        t0 = (i * 37) % (201 - 68)
        torch.manual_seed(i)
        d = np.load(ep)
        frames = down64(d["frames"][t0:t0 + 68, VIEW_IDX]).to(device)
        ctx, gt = frames[3:4].unsqueeze(0), frames[4:].unsqueeze(0)
        acts = d["actions"][t0 + 3:t0 + 67]
        with torch.autocast("cuda", torch.bfloat16):
            pred = rollout(model, vq, ctx, acts, 64, device)
        recon = tok_recon(vq, ctx, gt)
        preds.append(to_u8(pred))
        recons.append(to_u8(recon))
        print(f"{i}: {os.path.basename(ep)} t0={t0} {time.time() - start:.1f}s", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, pred=np.stack(preds), model="ivideogpt", ckpt=ckpt)
    np.savez_compressed(a.recon_out, pred=np.stack(recons), model="ivideogpt_tokrecon", ckpt=TOKENIZER)
    print(f"wrote {a.out} and {a.recon_out}", flush=True)


if __name__ == "__main__":
    main()
