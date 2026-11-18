"""Local controls and reports for the experimental S0 fake-quantization pilot."""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from uuid import uuid4

METHODS = {"bf16": "BF16", "actquant": "ActQuant", "qvla": "QVLA", "quant": "QuantVLA"}


def mode_label(mode: str) -> str:
    return " + ".join(METHODS.get(stage, stage) for stage in mode.split("+"))


class EvaluationWindow:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root, self.args = root, args
        self.events = queue.Queue()
        self.running = False
        self.reports = {}
        self.methods = {name: tk.BooleanVar() for name in METHODS if name != "bf16"}
        self.queries = tk.StringVar(value="128")
        self.rollouts = tk.StringVar(value="4")
        self.status = tk.StringVar(value="실행할 기법을 선택하세요.")
        self.history = tk.StringVar()

        title = "S0 fake quantization 실험"
        root.title(title)
        root.geometry("1050x580")
        root.minsize(905, 500)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.configure(bg="#edf2f6")
        style = ttk.Style(root)
        style.configure("TFrame", background="#edf2f6")
        style.configure("TLabel", background="#edf2f6", foreground="#142d40", font=("맑은 고딕", 10))
        style.configure("TCheckbutton", background="#edf2f6", font=("맑은 고딕", 10))
        style.configure("Title.TLabel", font=("맑은 고딕", 22, "bold"))
        style.configure("Run.TButton", font=("맑은 고딕", 11, "bold"), padding=(14, 7))
        style.configure("Treeview", rowheight=32, font=("맑은 고딕", 10))
        style.configure("Treeview.Heading", font=("맑은 고딕", 10, "bold"))

        panel = ttk.Frame(root, padding=24)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text=title, style="Title.TLabel").pack(anchor="w")
        ttk.Label(panel, text="LIBERO+ validation · 정규화 액션 MAE · 낮을수록 좋음").pack(anchor="w", pady=(2, 16))
        controls = ttk.Frame(panel)
        controls.pack(fill="x")
        self.inputs = []
        for key, variable in self.methods.items():
            control = ttk.Checkbutton(controls, text=METHODS[key], variable=variable)
            control.pack(side="left", padx=(0, 22))
            self.inputs.append(control)
        ttk.Label(controls, text="검증 쿼리").pack(side="left")
        count = ttk.Spinbox(controls, from_=1, to=100000, width=7, textvariable=self.queries)
        count.pack(side="left", padx=(6, 0))
        self.inputs.append(count)
        ttk.Label(controls, text="Spatial 롤아웃").pack(side="left", padx=(12, 0))
        trials = ttk.Spinbox(controls, from_=0, to=2402, width=5, textvariable=self.rollouts)
        trials.pack(side="left", padx=(6, 0))
        self.inputs.append(trials)
        self.run_button = ttk.Button(controls, text="평가 실행", style="Run.TButton", command=self.start)
        self.run_button.pack(side="right")
        ttk.Label(panel, text="선택한 기법을 한 모델에 적용합니다. 롤아웃 0은 MAE만 측정합니다.").pack(anchor="w", pady=(8, 14))

        toolbar = ttk.Frame(panel)
        toolbar.pack(fill="x", pady=(0, 7))
        ttk.Label(toolbar, text="저장된 평가").pack(side="left", padx=(0, 8))
        self.history_box = ttk.Combobox(toolbar, state="readonly", textvariable=self.history, width=44)
        self.history_box.pack(side="left")
        self.history_box.bind("<<ComboboxSelected>>", lambda _event: self.show_selected())
        ttk.Button(toolbar, text="결과 폴더", command=self.open_results).pack(side="right")
        columns = ("method", "mae", "delta", "success", "p50", "memory")
        self.table = ttk.Treeview(panel, columns=columns, show="headings", height=4, selectmode="none")
        for column, label in zip(columns, ("조합", "액션 MAE", "BF16 대비 MAE 차이", "롤아웃 성공률", "p50 (ms)", "피크 할당 (GiB)")):
            self.table.heading(column, text=label)
            self.table.column(column, anchor="center", width=230 if column == "method" else 124, minwidth=105)
        self.table.pack(fill="x")
        self.details = ttk.Label(panel, text="", wraplength=690)
        self.details.pack(anchor="w", pady=(8, 0))
        self.progress = ttk.Progressbar(panel, mode="indeterminate")
        self.progress.pack(fill="x", pady=(14, 7))
        ttk.Label(panel, textvariable=self.status, wraplength=690).pack(anchor="w")
        self.reload_history()
        self.root.after(150, self.poll)

    def open_results(self) -> None:
        self.args.results.mkdir(parents=True, exist_ok=True)
        os.startfile(self.args.results)

    def reload_history(self, preferred: Path | None = None) -> None:
        if self.args.results.is_dir():
            self.reports = {p.stem: p for p in sorted(
                self.args.results.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
            )}
        self.history_box.configure(values=list(self.reports))
        if self.reports:
            self.history.set(preferred.stem if preferred else next(iter(self.reports)))
            self.show_selected()

    def show_selected(self) -> None:
        path = self.reports.get(self.history.get())
        if path is None:
            return
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            runs = report["runs"]
            baseline = runs["bf16"]["action_mae_vs_ground_truth"]
            samples = runs["bf16"]["samples"]["validation"]
            rows = []
            rollout_errors = set()
            for mode, run in runs.items():
                mae = run["action_mae_vs_ground_truth"]
                rollout = run.get("rollout", {})
                rate = "미측정"
                if rollout.get("status") == "complete":
                    rate = f"{rollout['success_rate']:.1%} ({rollout['successes']}/{rollout['episodes']})"
                elif rollout:
                    rate = "실행 불가" if rollout["status"] == "unavailable" else "실행 실패"
                    rollout_errors.add(rollout["reason"])
                rows.append((mode_label(mode), f"{mae:.6f}", f"{mae - baseline:+.6f}", rate,
                             f"{run['forward_p50_ms']:.1f}", f"{run['peak_allocated_bytes'] / 1024**3:.2f}"))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.table.delete(*self.table.get_children())
            self.details.configure(text="")
            self.status.set(f"결과를 읽을 수 없습니다. {exc}")
            return
        self.table.delete(*self.table.get_children())
        for row in rows:
            self.table.insert("", "end", values=row)
        detail = f"step {report['checkpoint_step']} · 검증 쿼리 {len(samples)}개 · 에피소드 {len({s[0] for s in samples})}개"
        if len(rows) > 1:
            detail += " · Fake quantization"
        if rollout_errors:
            detail += "\n" + "\n".join(sorted(rollout_errors))
        self.details.configure(text=detail)
        if rollout_errors:
            self.status.set("MAE 저장 완료 · 롤아웃 미완료")
        elif any(run.get("rollout", {}).get("status") == "complete" for run in runs.values()):
            self.status.set("MAE 및 롤아웃 성공률 저장 완료")
        else:
            self.status.set("MAE 저장 완료")

    def start(self) -> None:
        if self.running:
            return
        try:
            count = int(self.queries.get())
            rollouts = int(self.rollouts.get())
            if not 1 <= count <= 100000 or not 0 <= rollouts <= 2402:
                raise ValueError()
        except ValueError:
            self.status.set("검증 쿼리는 1~100000, 롤아웃은 0~2402 사이의 정수를 입력하세요.")
            return
        on = [key for key, value in self.methods.items() if value.get()]
        label = "-".join(on) or "bf16"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.args.results / f"{stamp}-{label}-{uuid4().hex[:6]}.json"
        python = Path(sys.executable)
        if python.name.lower() == "pythonw.exe":
            python = python.with_name("python.exe")
        command = [str(python), "-B", "-u", str(Path(__file__).with_name("run_s0_quantization.py")),
                   "--checkpoint", str(self.args.checkpoint), "--data", str(self.args.data),
                   "--output", str(path), "--eval-queries", str(count), "--rollouts", str(rollouts)]
        if self.args.registry is not None:
            command.extend(("--registry", str(self.args.registry)))
        for method in self.methods:
            command.extend((f"--{method}", "on" if method in on else "off"))
        self.table.delete(*self.table.get_children())
        self.details.configure(text="")
        self.set_running(True)
        self.status.set("체크포인트와 검증 데이터 확인 중…")
        threading.Thread(target=self.evaluate, args=(command, path), daemon=True).start()

    def set_running(self, running: bool) -> None:
        self.running = running
        for widget in [self.run_button, *self.inputs]:
            widget.state(["disabled" if running else "!disabled"])
        self.history_box.configure(state="disabled" if running else "readonly")
        if running:
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.configure(value=0)

    def evaluate(self, command: list[str], path: Path) -> None:
        tail = ""
        try:
            self.args.scratch.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8",
                       TEMP=str(self.args.scratch), TMP=str(self.args.scratch),
                       HF_HOME=str(self.args.scratch / "hf"))
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  encoding="utf-8", errors="replace", env=env,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
                for line in process.stdout:
                    if not line.strip():
                        continue
                    tail = line.strip()
                    if tail.startswith("Running "):
                        name = tail.removeprefix("Running ")
                        self.events.put(("status", f"{mode_label(name)} 평가 중…"))
                    elif tail.startswith("Rollout "):
                        self.events.put(("status", tail))
                    elif ": action MAE " in tail:
                        name, mae = tail.split(": action MAE ", 1)
                        self.events.put(("status", f"{mode_label(name)} 완료 · 액션 MAE {mae}"))
                if process.wait() != 0 or not path.is_file():
                    self.events.put(("error", tail or "평가 프로세스가 종료됐습니다."))
                    return
            self.events.put(("done", path))
        except OSError as exc:
            self.events.put(("error", str(exc)))

    def poll(self) -> None:
        while not self.events.empty():
            event, value = self.events.get_nowait()
            if event == "status":
                self.status.set(value)
            elif event == "error":
                self.set_running(False)
                self.status.set(f"실행 실패 · {value}")
            else:
                self.set_running(False)
                self.reload_history(value)
        self.root.after(150, self.poll)

    def close(self) -> None:
        if self.running:
            self.status.set("평가가 완료되면 창을 닫을 수 있습니다.")
        else:
            self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "data"):
        parser.add_argument(f"--{name}", required=True, help="HF asset URI or verified HF cache path")
    for name in ("results", "scratch"):
        parser.add_argument(f"--{name}", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--registry", type=lambda p: Path(p).resolve(), help="Pinned VLADE HF asset registry JSON")
    args = parser.parse_args()
    root = tk.Tk()
    EvaluationWindow(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
