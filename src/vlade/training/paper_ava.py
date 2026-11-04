"""Paper-informed AVA sequence LoRA training; independent of the FFT baseline.

Reference: AVA-VLA, arXiv:2511.18960v4, sections 3.4/A/D.1.
Public AVA inference primitives remain in the unchanged pinned submodule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]


class LoRALinear(nn.Module):
    """Frozen linear plus rank-r update, alpha/r scaling, Gaussian A and zero B.

    Native modules preserve custom predict_action methods on the enclosing head.
    The algebra/initialization follows LoRA and the pinned upstream Gaussian recipe.
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.normal_(self.lora_a, std=1/rank)
        self.scale = alpha/rank
        self.in_features, self.out_features = base.in_features, base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        original = self.base(value)
        update = nn.functional.linear(nn.functional.linear(value.float(), self.lora_a), self.lora_b)
        return original + (update*self.scale).to(original.dtype)


def add_lora(module: nn.Module, rank: int, alpha: float) -> int:
    """Adapt all linear layers in one component; keep all its other weights frozen."""
    if type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid LoRA rank/alpha")
    module.requires_grad_(False)
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            raise ValueError("Refusing to apply LoRA twice")
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
            count += 1
        else:
            count += add_lora(child, rank, alpha)
    return count


def validate_recipe(cfg: dict) -> None:
    """Reject silent substitutions of the selected highest-report training recipe."""
    expected = dict(max_steps=100000, effective_batch_size=256, sequence_length=4, detach_every=2,
                    lora_rank=32, learning_rate=5e-4, warmup_steps=10000, action_chunk_size=8,
                    action_dimension=7, attention_lambda=1., attention_target_mean=.6, lr_schedule="cosine",
                    initial_repository="openvla/openvla-7b")
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Paper best recipe mismatch: {key}")
    if cfg.get("attention_gamma") != [1.9,.1] or cfg.get("image_aug") is not False:
        raise ValueError("Unsupported attention/image contract")
    if (type(cfg.get("micro_batch_size")) is not int or cfg["micro_batch_size"] != 1 or
            cfg["micro_batch_size"]*cfg.get("accumulation_steps",0) != cfg["effective_batch_size"]):
        raise ValueError("Effective batch must be 256 sequences (microbatch 1 x accumulation 256)")
    if (not isinstance(cfg.get("initial_revision"), str) or len(cfg["initial_revision"]) != 40 or
            cfg.get("query_interval_steps") != 8 or cfg.get("lora_dropout") != 0):
        raise ValueError("Invalid pinned source/data contract")
    if min(cfg["save_steps"],cfg["session_hours"],cfg["save_reserve_seconds"]) <= 0:
        raise ValueError("Invalid storage/session policy")


def sequence_count(data: Any, split: str = "train") -> int:
    return sum(max(0, item["queries"]-3) for item in data.entries if item["split"] == split)


def sequence_stream(data: Any, *, seed: int, skip: int = 0) -> Iterator[list[dict]]:
    """Deterministic sliding four-query windows; epoch shuffle; no episode crossing."""
    total = sequence_count(data)
    if total < 1 or skip < 0:
        raise ValueError("No four-query train sequences or invalid cursor")
    epoch, cursor = divmod(skip,total)
    while True:
        rng = random.Random(seed+epoch*1000003)
        order = [e["episode_id"] for e in data.entries if e["split"] == "train" and e["queries"] >= 4]
        rng.shuffle(order)
        position = 0
        for entry, samples in data.episodes("train", order=order):
            starts = list(range(len(samples)-3))
            rng.shuffle(starts)
            for start in starts:
                if position >= cursor:
                    window = samples[start:start+4]
                    if (len({q["episode_id"] for q in window}) != 1 or
                            [q["timestep"] for q in window] != list(range(window[0]["timestep"],window[0]["timestep"]+32,8))):
                        raise ValueError("Sequence is not a contiguous same-episode query window")
                    yield window
                position += 1
        epoch, cursor = epoch+1, 0


def attention_penalty(weights: torch.Tensor, target: float = .6) -> torch.Tensor:
    """Equation 13: scalar L2 norm of mean *actual omega*, separately per sample."""
    if weights.ndim != 3 or not torch.isfinite(weights).all():
        raise ValueError("Invalid attention weights")
    return (weights.float().mean(dim=(1,2))-target).abs().mean()


def backward_sequence(frames: list[Any], frame_fn: Callable, *, accumulation: int) -> dict:
    """Unroll 4, backward in two-frame segments, detach only between frames 2/3."""
    if len(frames) != 4 or accumulation < 1:
        raise ValueError("K=4 and positive accumulation are required")
    context, segment = None, None
    totals = {"mae":0., "attention_penalty":0.}
    for index, frame in enumerate(frames):
        loss, context, metrics = frame_fn(frame, context)
        if loss.ndim or not torch.isfinite(loss).item():
            raise FloatingPointError("Invalid sequence loss")
        segment = loss if segment is None else segment+loss
        for key in totals:
            totals[key] += float(metrics[key])/4
        if (index+1)%2 == 0:
            (segment/(4*accumulation)).backward()
            context = context.detach()
            segment = None
    return totals


def _model(cfg: dict, checkpoint: Path, device: torch.device) -> tuple:
    from vlade.training.fft import _activate_upstream
    _activate_upstream("ava_vla")
    from transformers import AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from prismatic.models.attn_weight_generator import AttentionWeightGenerator
    from prismatic.vla.constants import ACTION_DIM,NUM_ACTIONS_CHUNK,PROPRIO_DIM
    if (ACTION_DIM,NUM_ACTIONS_CHUNK,PROPRIO_DIM) != (7,8,8):
        raise ValueError("Pinned source is not configured for the LIBERO action/proprio contract")
    config = OpenVLAConfig.from_pretrained(checkpoint,local_files_only=True)
    config.extra_attn_weights = dict(force_eager_attn=True,extra_layer_ids=list(range(config.text_config.num_hidden_layers)))
    config.text_config.use_cache = False
    model = OpenVLAForActionPrediction.from_pretrained(checkpoint,config=config,torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,attn_implementation="eager",local_files_only=True).to(device)
    model.vision_backbone.set_num_images_in_input(2)
    processor = AutoProcessor.from_pretrained(checkpoint,trust_remote_code=True,local_files_only=True)
    dim = model.llm_dim
    head = L1RegressionActionHead(input_dim=dim,hidden_dim=dim,action_dim=7).to(device=device,dtype=torch.bfloat16)
    proprio = ProprioProjector(llm_dim=dim,proprio_dim=8).to(device=device,dtype=torch.bfloat16)
    patches = model.vision_backbone.get_num_patches()
    edge = math.isqrt(patches)
    if edge*edge != patches:
        raise ValueError("AVA requires square image patches")
    generator = AttentionWeightGenerator(embed_dim=dim//4,image_shape=(edge,edge,dim),
        action_shape=(8,7,dim),text_dim=dim,num_images=2,max_steps=cfg["max_steps"],
        head_type="softmaxscore",score_config=(1.9,.1,0.),sink_ids=[68,75,180,187,324,331,436,443],
        sink_weight=1.).to(device=device,dtype=torch.bfloat16)
    counts = [add_lora(m,cfg["lora_rank"],cfg["lora_alpha"]) for m in (model,head,proprio)]
    if not all(counts):
        raise RuntimeError("LoRA did not attach to every required component")
    generator.requires_grad_(True)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.config.use_cache = False
    return model,processor,head,proprio,generator


def _frame_function(cfg: dict, components: tuple, data: Any, device: torch.device) -> Callable:
    import numpy as np
    from prismatic.vla.datasets import RLDSBatchTransform
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.training.train_utils import get_current_action_mask,get_next_actions_mask
    model,processor,head,proprio,generator = components
    transform = RLDSBatchTransform(ActionTokenizer(processor.tokenizer),processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=True,use_proprio=True)
    collate = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,padding_side="right")
    num_patches = model.vision_backbone.get_num_patches()*2+1

    def frame(sample: dict, context: torch.Tensor | None, step: int = 0) -> tuple:
        item = transform(dict(dataset_name=b"libero_plus_mixdata",action=sample["gt_action"],
            observation=dict(image_primary=np.asarray(data.image(sample,"primary"))[None],
                             image_wrist=np.asarray(data.image(sample,"wrist"))[None],proprio=sample["proprio"][None]),
            task=dict(language_instruction=sample["instruction"].encode())))
        batch = collate([item])
        labels = batch["labels"].to(device)
        mask = get_current_action_mask(labels[:,1:]) | get_next_actions_mask(labels[:,1:])
        captured = []
        hook = generator.register_forward_hook(lambda module,args,result: captured.append(result[1][0]))
        try:
            with torch.autocast(device.type,dtype=torch.bfloat16):
                output = model(input_ids=batch["input_ids"].to(device),attention_mask=batch["attention_mask"].to(device),
                    pixel_values=batch["pixel_values"].to(device=device,dtype=torch.bfloat16),labels=labels,
                    proprio=batch["proprio"].to(device=device,dtype=torch.bfloat16).reshape(1,8),
                    proprio_projector=proprio,temporal_context=context,vision_attn_weight_generator=generator,
                    batch_idx=step,output_hidden_states=True,use_cache=False,return_dict=True)
                def action_features(layer):
                    hidden = output.hidden_states[layer][:,num_patches:-1]
                    if hidden.shape[:2] != mask.shape or mask.sum().item() != 56:
                        raise ValueError("Action hidden/token alignment failed")
                    return hidden[mask].reshape(1,56,-1)
                prediction = head.predict_action(action_features(-1))
                target = torch.as_tensor(sample["gt_action"],device=device,dtype=torch.float32)[None]
                valid = torch.as_tensor(sample["action_valid_mask"],device=device)[None]
                if prediction.shape != (1,8,7) or valid.dtype != torch.bool or not valid.any():
                    raise ValueError("Invalid action/mask shape")
                mae = (prediction.float()[valid]-target[valid]).abs().mean()
                if len(captured) != 1:
                    raise RuntimeError("Expected the actual AVA attention omega once per frame")
                penalty = attention_penalty(captured[0],cfg["attention_target_mean"])
                context = action_features(cfg["temporal_feature_layer"])
                loss = mae+cfg["attention_lambda"]*penalty
            return loss,context,dict(mae=mae.detach().item(),attention_penalty=penalty.detach().item())
        finally:
            hook.remove()
    return frame


def _resolve(path: str) -> Path:
    if str(path).startswith("hfasset:"):
        from vlade.storage.assets import resolve_path
        return resolve_path(path)
    value = Path(path)
    return value if value.is_absolute() else ROOT/value


def preflight(cfg: dict) -> dict:
    from vlade.data.episode_data import EpisodeData
    validate_recipe(cfg)
    checkpoint = _resolve(cfg["initial_checkpoint"])
    manifest = json.loads((checkpoint/"download_manifest.json").read_text())
    if manifest.get("repository") != cfg["initial_repository"] or manifest.get("revision") != cfg["initial_revision"]:
        raise ValueError("This recipe requires the pinned pretrained OpenVLA, not AVA/OFT weights")
    for item in manifest["files"]:
        path = checkpoint/item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("Incomplete pretrained checkpoint")
    data = EpisodeData(_resolve(cfg["data"]))
    count = sequence_count(data)
    if not count:
        raise ValueError("No full K=4 train windows")
    return dict(train_sequences=count,steps=cfg["max_steps"],effective_batch=cfg["effective_batch_size"],
        query_predictions_per_optimizer_step=1024,dataset="LIBERO+ adaptation; paper reports LIBERO",
        full_run_query_predictions=cfg["max_steps"]*1024)


def _identity(cfg: dict) -> dict:
    from vlade.data.colab_assets import sha256
    data = _resolve(cfg["data"])
    code = [Path(__file__),*(Path(__file__).with_name(name) for name in ("loop.py","run_control.py","compute_budget.py")),
            Path(__file__).parents[1]/"data/episode_data.py"]
    manifest = ROOT/"experiments/manifests/colab_source.json"
    source = json.loads(manifest.read_text()) if manifest.is_file() else {}
    return dict(configuration=cfg,code_sha256={p.name:sha256(p) for p in code},
        data_sha256={name:sha256(data/name) for name in ("manifest.json","splits.json","normalization.json")},
        initial_manifest_sha256=sha256(_resolve(cfg["initial_checkpoint"])/"download_manifest.json"),
        source_revisions=source.get("source_revisions",{}),upstream_sha256={p:h for p,h in source.get("files_sha256",{}).items()
            if p.startswith("third_party/ava_vla/") or p.endswith("checkpoint-extra-attn.patch")})


def _save(output: Path, identity: dict, components: tuple, optimizer: Any, scheduler: Any, state: Any,
          *, hub: Any = None, metrics: dict | None = None) -> Path:
    from vlade.training.loop import training_state
    from vlade.training.run_control import seal_checkpoint
    from vlade.utils.filesystem import publish_directory
    destination = output/f"step-{state.step:07d}"
    if destination.exists():
        raise FileExistsError("Refusing to overwrite paper checkpoint")
    stage = output/f".step-{state.step:07d}-{uuid.uuid4().hex}.incomplete"
    stage.mkdir()
    model,processor,head,proprio,generator = components
    def byte_count(value):
        if isinstance(value,torch.Tensor):
            return value.numel()*value.element_size()
        if isinstance(value,dict):
            return sum(byte_count(x) for x in value.values())
        if isinstance(value,(list,tuple)):
            return sum(byte_count(x) for x in value)
        return 0
    required = sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    required += sum(byte_count(m.state_dict()) for m in (head,proprio,generator))
    required += byte_count(optimizer.state_dict())
    if shutil.disk_usage(output).free < required*1.2+1024**3:
        stage.rmdir()
        raise OSError("Insufficient space for LoRA/components/optimizer save plus margin")
    torch.save(dict(model_adapters={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
        action_head=head.state_dict(),proprio=proprio.state_dict(),generator=generator.state_dict(),
        trainer=training_state(state,optimizer,scheduler)),stage/"state.pt")
    processor.save_pretrained(stage)
    (stage/"paper_ava.json").write_text(json.dumps(dict(identity=identity,step=state.step,
        format="native_lora_components_not_merged_fft",sequence_cursor=state.microsteps),indent=2))
    if hub is not None:
        hub.prepare(stage, state.step, metrics or {})
    seal_checkpoint(stage,state.step,state.microsteps)
    (stage/"COMPLETE").write_text("ok\n")
    publish_directory(stage,destination)
    return destination


def _restore(path: Path, identity: dict, components: tuple, optimizer: Any, scheduler: Any) -> Any:
    from vlade.training.run_control import verify_checkpoint
    from vlade.training.loop import restore_training_state
    verify_checkpoint(path)
    metadata = json.loads((path/"paper_ava.json").read_text())
    if metadata["identity"] != identity:
        raise ValueError("Paper resume configuration/code/data/base checkpoint mismatch")
    payload = torch.load(path/"state.pt",map_location="cpu",weights_only=True)
    model,_,head,proprio,generator = components
    current = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(current) != set(payload["model_adapters"]):
        raise ValueError("LoRA module set mismatch")
    with torch.no_grad():
        for key,param in current.items():
            param.copy_(payload["model_adapters"][key])
    for module,key in ((head,"action_head"),(proprio,"proprio"),(generator,"generator")):
        module.load_state_dict(payload[key],strict=True)
    state = restore_training_state(payload["trainer"],optimizer,scheduler)
    if state.step != metadata["step"] or state.microsteps != metadata["sequence_cursor"]:
        raise ValueError("Paper checkpoint step/cursor mismatch")
    return state


def require_pilot(cfg: dict) -> dict:
    """Bind the real GPU pilot report and sealed checkpoint to current inputs/code."""
    from vlade.training.run_control import verify_checkpoint
    expected = _identity(cfg)
    expected["pilot"] = True
    folder = _resolve(cfg["output"]+'-pilot')
    report = json.loads((folder/"pilot_report.json").read_text())
    checkpoint = Path(report["checkpoint"])
    if checkpoint.parent.resolve() != folder.resolve():
        raise ValueError("Pilot checkpoint is outside this run")
    verify_checkpoint(checkpoint)
    saved = json.loads((checkpoint/"paper_ava.json").read_text())
    if report.get("status") != "passed" or report.get("identity") != expected or saved["identity"] != expected:
        raise ValueError("Stale pilot report/configuration/code/data/base checkpoint")
    return report


def train(cfg: dict, *, resume: Path | None = None, pilot: bool = False) -> None:
    from vlade.training.hub import HubCheckpoints
    if resume is not None:
        resume = _resolve(str(resume))
    from vlade.data.episode_data import EpisodeData
    from vlade.data.colab_assets import sha256
    from vlade.training.fft import set_seed
    from vlade.training.loop import LoopState,make_scheduler
    from vlade.training.run_control import configure_tensorflow_cpu,gradient_diagnostics
    from vlade.training.compute_budget import should_stop
    preflight(cfg)
    if not pilot and resume is None:
        require_pilot(cfg)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Paper AVA requires a BF16 CUDA GPU")
    started = time.monotonic()
    deadline = started+cfg["session_hours"]*3600
    external_deadline = float(os.environ.get("VLADE_DEADLINE_UNIX","0"))
    if external_deadline:
        deadline = min(deadline,started+external_deadline-time.time())
    if deadline-started <= cfg["save_reserve_seconds"]:
        raise RuntimeError("Insufficient session time for loading/training/saving")
    configure_tensorflow_cpu(cfg["seed"])
    set_seed(cfg["seed"])
    base = _resolve(cfg["initial_checkpoint"])
    manifest = json.loads((base/"download_manifest.json").read_text())
    for item in manifest["files"]:
        if sha256(base/item["path"]) != item["sha256"]:
            raise ValueError("Pretrained checkpoint SHA mismatch")
    device = torch.device("cuda:0")
    components = _model(cfg,base,device)
    model,_,head,proprio,generator = components
    modules = [model,head,proprio,generator]
    for module in modules:
        module.train()
    params = [p for m in modules for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    scheduler = make_scheduler(optimizer,cfg["max_steps"],cfg["warmup_steps"],"cosine")
    data = EpisodeData(_resolve(cfg["data"]))
    identity = _identity(cfg)
    identity["pilot"] = pilot
    output = _resolve(cfg["output"]+('-pilot' if pilot else ''))
    hub = HubCheckpoints(output, resume=resume, metric="mae")
    output.mkdir(parents=True,exist_ok=True)
    if not resume and any(output.glob("step-*")):
        raise FileExistsError("Use --resume or a new output path")
    state = _restore(resume,identity,components,optimizer,scheduler) if resume else LoopState()
    if not 0 <= state.step <= (1 if pilot else cfg["max_steps"]):
        raise ValueError("Invalid restored paper optimizer step")
    stream = sequence_stream(data,seed=cfg["seed"],skip=state.microsteps)
    frame_fn = _frame_function(cfg,components,data,device)
    run_id = uuid.uuid4().hex
    report = dict(status="running",identity=identity,pilot=pilot,step=state.step,
        gpu=torch.cuda.get_device_name(0),vram_bytes=torch.cuda.get_device_properties(0).total_memory,
        trainable_parameters=sum(p.numel() for p in params))
    from vlade.training.fft import _software_versions
    report["software"] = {**_software_versions(),"cuda":torch.version.cuda}
    def write_report():
        from vlade.data.episode_data import write_json
        write_json(output/f"run-{run_id}.json",report)
    write_report()
    # Pilot checks one complete K=4 backward/update/save/reload without changing the 100K schedule.
    target = 1 if pilot else cfg["max_steps"]
    save_reserve = cfg["save_reserve_seconds"]
    try:
        with (output/f"metrics-{run_id}.jsonl").open("x") as log:
            while state.step < target:
                tick = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                totals = dict(mae=0.,attention_penalty=0.)
                accumulation = 1 if pilot else cfg["accumulation_steps"]
                for _ in range(accumulation):
                    metrics = backward_sequence(next(stream),lambda sample,ctx: frame_fn(sample,ctx,state.step),accumulation=accumulation)
                    for key in totals:
                        totals[key] += metrics[key]/accumulation
                if "gradient_diagnostics" not in report:
                    report["gradient_diagnostics"] = gradient_diagnostics(dict(vision=model.vision_backbone,
                        language=model.language_model,action=head,proprio=proprio,attention=generator))
                norm = torch.nn.utils.clip_grad_norm_(params,cfg["max_grad_norm"],error_if_nonfinite=True)
                lr = optimizer.param_groups[0]["lr"]
                optimizer.step(); scheduler.step()
                state.step += 1
                state.microsteps += accumulation
                torch.cuda.synchronize(device)
                elapsed = time.monotonic()-tick
                stop = should_stop(deadline-time.monotonic(),save_reserve,elapsed)
                row = dict(step=state.step,sequence_cursor=state.microsteps,lr=lr,grad_norm=float(norm),seconds=elapsed,**totals)
                log.write(json.dumps(row)+"\n"); log.flush()
                print(json.dumps(row),flush=True)
                if stop or state.step == 1 or state.step%cfg["save_steps"] == 0 or state.step == target:
                    save_started = time.monotonic()
                    saved = _save(output,identity,components,optimizer,scheduler,state,hub=hub,metrics=row)
                    hub.publish(saved, final=state.step == target)
                    save_reserve = max(save_reserve,(time.monotonic()-save_started)*1.3)
                    stop = stop or should_stop(deadline-time.monotonic(),save_reserve,elapsed)
                    if pilot:
                        restored = _restore(saved,identity,components,optimizer,scheduler)
                        if restored.step != 1:
                            raise RuntimeError("Pilot reload failed")
                        from vlade.data.episode_data import write_json
                        write_json(output/"pilot_report.json",dict(status="passed",identity=identity,
                            scope="K4_backward_LoRA_generator_save_reload_not_full_batch_or_rollout",checkpoint=str(saved),
                            seconds_per_sequence=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(device)))
                report["step"] = state.step
                write_report()
                if stop and state.step < target:
                    report["status"] = "budget_exhausted"
                    break
            else:
                report["status"] = "pilot_completed" if pilot else "completed"
    except BaseException as error:
        report.update(status="failed",error_type=type(error).__name__,error=str(error),step=state.step)
        raise
    finally:
        hub.flush()
        report["elapsed_seconds"] = time.monotonic()-started
        write_report()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--resume",type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.preflight:
        print(json.dumps(preflight(cfg),indent=2))
    else:
        train(cfg,resume=args.resume,pilot=args.pilot)


if __name__ == "__main__":
    main()
