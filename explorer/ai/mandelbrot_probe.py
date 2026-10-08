"""Inference-only test: can the saved PBE model synthesize a finite Mandelbrot mask?

Uses eight IO examples per prompt, as trained. Does not claim to give the model
English instructions, fine-tune it, or synthesize a renderer. Writes only the
chosen output directory. Original training modules are not imported for side effects.
"""
import argparse
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Existing pbe.py containing the trained architecture/interpreter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--prompts", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 1 or args.prompts < 1:
        parser.error("Positive samples/prompts required")
    torch.set_num_threads(2)
    torch.manual_seed(7102026)
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    parsed = ast.parse(args.source.read_text())
    selected = [n for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in {"GPT", "run", "run_many"}]
    if len(selected) != 3: raise RuntimeError("Expected GPT and the two known interpreter functions")
    scope = {"torch": torch, "nn": nn, "F": F, "V": 259, "T": 31, "L": 5}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(args.source), "exec"), scope)
    model = scope["GPT"](d=512, layers=12, heads=8)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    run, run_many = scope["run"], scope["run_many"]
    # Array index is x*256+y. 255 means no escape within 64 complex iterations;
    # 0 means escape. This is a finite numerical target, not exact set membership.
    xs = np.repeat(np.arange(256), 256)
    ys = np.tile(np.arange(256), 256)
    c = (-2.0 + 3.0 * xs / 255) + 1j * (-1.5 + 3.0 * ys / 255)
    z = np.zeros(65536, dtype=np.complex128)
    active = np.ones(65536, dtype=bool)
    for _ in range(64):
        z[active] = z[active] * z[active] + c[active]
        active &= np.abs(z) <= 2
    target = np.where(active, 255, 0)
    all_x = torch.tensor(xs, device=device, dtype=torch.long)
    all_y = torch.tensor(ys, device=device, dtype=torch.long)
    all_target = torch.tensor(target, device=device, dtype=torch.long)
    rng = np.random.default_rng(7102026)
    inside, outside = np.flatnonzero(active), np.flatnonzero(~active)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "mandelbrot_target.npy", target.reshape(256, 256))
    started = time.monotonic()

    @torch.no_grad()
    def generate(indices, answers, count):
        examples = torch.stack([all_x[indices], all_y[indices], answers], 1).flatten()
        prefix = torch.cat([torch.tensor([258], device=device), examples, torch.tensor([257], device=device)])
        batches = []
        for offset in range(0, count, 32):
            tokens = prefix[None].repeat(min(32, count - offset), 1)
            for _ in range(5):
                with torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext():
                    logits = model(tokens)[:, -1, :256].float()
                tokens = torch.cat([tokens, torch.multinomial(F.softmax(logits, -1), 1)], 1)
            batches.append(tokens[:, -5:])
        return torch.cat(batches)

    generated, prompts = [], []
    with torch.inference_mode():
        for i in range(args.prompts):
            indices_np = np.concatenate([rng.choice(inside, 4, replace=False), rng.choice(outside, 4, replace=False)])
            rng.shuffle(indices_np)
            indices = torch.tensor(indices_np, device=device)
            candidates = generate(indices, all_target[indices], args.samples)
            outputs = run_many(candidates, all_x[indices][None].expand(len(candidates), 8), all_y[indices][None].expand(len(candidates), 8))
            fit = (outputs == all_target[indices]).sum(1)
            prompts.append({"examples": [[int(xs[j]), int(ys[j]), int(target[j])] for j in indices_np],
                            "generated": len(candidates), "matches_all_eight": int((fit == 8).sum()),
                            "best_examples_matched": int(fit.max())})
            generated.append(candidates)
            print(f"Mandelbrot prompt {i+1}: {int((fit == 8).sum())}/{args.samples} match all eight examples", flush=True)
        candidates = torch.unique(torch.cat(generated), dim=0)
        # Separate, balanced diagnostic inputs select candidates for full-grid checking.
        diag_np = np.concatenate([rng.choice(inside, 32, replace=False), rng.choice(outside, 32, replace=False)])
        diag = torch.tensor(diag_np, device=device)
        answers = run_many(candidates, all_x[diag][None].expand(len(candidates), 64), all_y[diag][None].expand(len(candidates), 64))
        ranking = (answers == all_target[diag]).sum(1)
        finalist_indices = torch.unique(torch.cat([torch.argsort(ranking, descending=True)[:min(8, len(candidates))],
                                                  torch.nonzero(ranking == 64).flatten()]))
        finalists = candidates[finalist_indices]
        diagnostic_perfect = int((ranking == 64).sum())
        evaluations = []
        best_outputs = None
        best_key = (-1, -1)
        for candidate in finalists:
            outputs = run(candidate[None].expand(65536, 5), all_x, all_y)
            hit = outputs == all_target
            inside_recall = float(hit[all_target == 255].float().mean())
            outside_recall = float(hit[all_target == 0].float().mean())
            score = {"program": bytes(candidate.tolist()).hex().upper(), "pixels_correct": int(hit.sum()),
                     "pixels": 65536, "exact": bool(hit.all()), "inside_recall": inside_recall,
                     "outside_recall": outside_recall, "balanced_accuracy": (inside_recall + outside_recall) / 2,
                     "no_output_pixels": int((outputs < 0).sum())}
            key = (score["balanced_accuracy"], score["pixels_correct"])
            if key > best_key:
                best_key = key
                best_outputs = outputs.cpu().numpy().reshape(256, 256)
            evaluations.append(score)
        # Known-easy held-out function sanity check; no training or checkpoint changes.
        control_hits = set()
        for _ in range(2):
            indices = torch.tensor(rng.choice(65536, 8, replace=False), device=device)
            candidates = torch.unique(generate(indices, 255 - all_y[indices], args.samples), dim=0)
            outputs = run_many(candidates, all_x[indices][None].expand(len(candidates), 8), all_y[indices][None].expand(len(candidates), 8))
            for candidate in candidates[(outputs == (255-all_y[indices])).all(1)]:
                if bool((run(candidate[None].expand(65536, 5), all_x, all_y) == 255-all_y).all()):
                    control_hits.add(bytes(candidate.tolist()).hex().upper())
        print(f"Control NOT b: {len(control_hits)} distinct fully verified programs", flush=True)
    evaluations.sort(key=lambda x: (x["balanced_accuracy"], x["pixels_correct"]), reverse=True)
    if best_outputs is not None: np.save(args.output / "best_generated_outputs.npy", best_outputs)
    report = {"checkpoint": str(args.checkpoint), "device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
              "target": "256x256 grid: Re(c)=-2+3*x/255, Im(c)=-1.5+3*y/255; z0=0; 64 iterations z=z*z+c; 255 if no |z|>2, else 0",
              "instructions_to_model": "Only eight (x,y,output) examples per prompt, not natural language or formula tokens",
              "nano_contract": "five code bytes; 16-byte memory; x=M5,y=M6; first OUT within 64 NANO instructions",
              "inside_pixels": len(inside), "outside_pixels": len(outside),
              "constant_zero_pixel_accuracy": len(outside)/65536, "constant_zero_balanced_accuracy": 0.5,
              "sampled_programs": args.samples*args.prompts, "unique_sampled_programs": len(torch.unique(torch.cat(generated),dim=0)),
              "prompts": prompts, "diagnostic_perfect_candidates": diagnostic_perfect,
              "exact_solutions_found": sum(r["exact"] for r in evaluations),
              "full_grid_candidates_checked": len(evaluations), "full_grid_results": evaluations,
              "control_not_b_attempts": args.samples*2, "control_not_b_verified_programs": sorted(control_hits),
              "seconds": time.monotonic()-started}
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"sampled":report["sampled_programs"],"best":evaluations[0],"seconds":report["seconds"]}),flush=True)


if __name__ == "__main__":
    main()
