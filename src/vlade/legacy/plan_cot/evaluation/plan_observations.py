"""Wire format for live LIBERO+ frames sent to the isolated T1 process."""
from __future__ import annotations

import base64
from io import BytesIO
import json
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from vlade.legacy.plan_cot.distillation.teacher_plan import InvalidTeacherPlan


def encode_query(query) -> dict:
    cameras = {}
    for name in ("primary", "wrist"):
        buffer = BytesIO()
        getattr(query, name).save(buffer, format="PNG")
        cameras[name] = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"task_id": query.task_id, "timestep": query.timestep,
            "instruction": query.instruction, "proprio": query.proprio.tolist(), **cameras}


def live_response(policy, request: dict, plan_tokens: int) -> dict:
    if request["live"] == "reset":
        policy.reset()
        return {"live": "reset"}
    if request["live"] != "query" or (request["plan"] and not plan_tokens):
        raise ValueError("Unsupported T1 live query")
    proprio = np.asarray(request["proprio"], dtype=np.float32)
    if proprio.shape != (8,) or not np.isfinite(proprio).all():
        raise ValueError("Expected eight finite normalized proprio values")
    sample = {"episode_id": f"libero-plus-{request['task_id']}", "timestep": request["timestep"],
              "instruction": request["instruction"], "proprio": proprio}
    for name in ("primary", "wrist"):
        with Image.open(BytesIO(base64.b64decode(request[name], validate=True))) as image:
            if image.size != (256, 256):
                raise ValueError("Expected a 256 by 256 camera frame")
            sample[name] = image.convert("RGB")
    data = SimpleNamespace(image=lambda row, camera: row[camera])
    result = {key: request[key] for key in ("task_id", "timestep")}
    if request["action"]:
        action, _ = policy.predict(sample, data)
        result["action"] = action.detach().float().cpu().tolist()
    if request["plan"]:
        try:
            result["plan"] = policy.plan(sample, data, max_new_tokens=plan_tokens)
        except InvalidTeacherPlan as error:
            result["plan"] = None
            print(json.dumps({"event": "invalid_teacher_plan", "task_id": request["task_id"],
                              "timestep": request["timestep"], "error": str(error)}, ensure_ascii=False),
                  file=sys.stderr, flush=True)
    return result
