"""Reconnect exact existing Colab environments without moving caches or run outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vlade.runtime.environment import interpreter, locate, prepare, specification


def reconnect(code: Path, workspace: Path, *, search_root: Path,
              allow_student_setup: bool = False) -> dict[str, str]:
    """Reuse exact runtime specs; optional setup is limited to a missing Student.

    Search only immediate workspaces' environments directories. Never scan Drive,
    stamp arbitrary Python installations as compatible, or replace existing paths.
    """
    workspace, search_root = workspace.resolve(), search_root.resolve()
    if not workspace.is_relative_to(search_root) or workspace.is_relative_to(search_root / "drive"):
        raise ValueError("Runtime recovery workspace must be local to the search root, outside Drive")
    roots = {workspace}
    roots.update(path for path in search_root.iterdir()
                 if path.name != "drive" and path.is_dir() and not path.is_symlink())
    selected: dict[str, Path | None] = {}
    targets = {}
    for name in ("student", "ava"):
        spec = specification(code, name)
        identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
        target = workspace / "environments" / (name + "-" + identity)
        if not target.parent.resolve().is_relative_to(workspace):
            raise ValueError("Environment aliases must stay inside the selected local workspace")
        targets[name] = target
        try:
            selected[name] = locate(code, workspace, name).parent.parent
            print(f"{name}: 현재 workspace의 검증된 환경을 재사용합니다.", flush=True)
            continue
        except FileNotFoundError:
            print(f"{name}: 필요한 환경 경로 {target}", flush=True)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"기존의 불완전한 환경은 덮어쓰지 않습니다: {target}")
        candidates = set()
        for root in roots:
            directory = root / "environments" / target.name
            marker = directory / "vlade-runtime.json"
            if not marker.is_file() or not interpreter(directory).is_file():
                continue
            try:
                matches = json.loads(marker.read_text(encoding="utf-8")) == spec
            except (OSError, ValueError):
                continue
            resolved = directory.resolve()
            if matches and resolved.is_relative_to(search_root) and not resolved.is_relative_to(search_root / "drive"):
                # Includes the existing AVA patch verification before Student setup.
                locate(code, root, name)
                candidates.add(resolved)
        if len(candidates) > 1:
            raise ValueError(f"{name} 환경 후보가 여러 개입니다. 사용할 경로를 확인해야 합니다: "
                             + ", ".join(str(path) for path in sorted(candidates)))
        if candidates:
            selected[name] = candidates.pop()
            print(f"{name}: 기존 환경 발견 {selected[name]}", flush=True)
        elif name == "student" and allow_student_setup:
            selected[name] = None
        else:
            raise FileNotFoundError(
                f"{name}: 현재 workspace와 {search_root}의 직접 하위 workspace에서 "
                "이 설정과 일치하는 완료 환경을 찾지 못했습니다. "
                + ("Student 분리 설치가 필요하면 --allow-student-setup을 명시하십시오."
                   if name == "student" else "기존 AVA 환경 경로를 확인하십시오."))
    # Resolve both roles before making any link or starting the optional setup.
    for name, directory in selected.items():
        target = targets[name]
        if directory is None:
            print("Student 전용 Python 환경과 패키지를 준비합니다. 호스트/AVA는 재설치하지 않습니다.", flush=True)
            prepare(code, workspace, name)
        elif directory != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(directory, target_is_directory=True)
        selected[name] = locate(code, workspace, name)
    return {name: str(path) for name, path in selected.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, default=Path("/content"))
    parser.add_argument("--allow-student-setup", action="store_true")
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[3]
    from vlade.utils.source_identity import source_identity
    source_identity(code, require_clean=True)
    print(json.dumps(reconnect(code, args.workspace, search_root=args.search_root,
        allow_student_setup=args.allow_student_setup), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
