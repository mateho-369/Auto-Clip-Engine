"""The pipeline as data: stages, dependencies, resources, hardware gating.

Everything the scheduler needs to know is in `STAGES`, so adding/renaming a stage
is a one-line change and the UI stepper, the API status payload and the run graph
all follow automatically. Two rules are encoded here:

* a stage lists the stages whose **outputs are its inputs** (the directed graph in
  the brief); `video` deliberately does *not* depend on the voice branch — it
  starts from Stage 1's estimate so voice and video run concurrently, and
  `video_fit` (a cheap CPU re-time) reconciles them with Stage 3b's real length;
* `resource` + `requires_gpu` drive the semaphores/VRAM lock that keep an 8GB card
  from ever hosting two models at once.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StageSpec:
    key: str
    title: str
    emoji: str
    role: str = ""                                  # LLM role, or engine module name
    resource: str = "io"                            # llm | tts | gpu | cpu | io
    per_scene: bool = True
    requires_gpu: bool = False
    depends: tuple = ()
    blurb: str = ""
    model: str = ""
    outputs: tuple = ()                             # asset kinds this stage writes
    deferrable: bool = False                         # can be queued for Machine A
    retryable: bool = True


STAGES = [
    StageSpec("script", "Script ready", "🎬", role="auto_idea", resource="llm", per_scene=False,
              depends=(), blurb="Mode A: Director script locked · Mode B: Controller writes it",
              model="sailor2:8b", outputs=("script",)),
    StageSpec("breakdown", "1 · Scene breakdown", "🧩", role="controller", resource="llm",
              per_scene=False, depends=("script",),
              blurb="Mechanical segmentation into scenes + visual prompt, mood tag, duration",
              model="sailor2:8b", outputs=("scenes",)),
    StageSpec("voice_base", "3a · Khmer voice", "🗣️", role="tts", resource="tts",
              depends=("breakdown",),
              blurb="sherpa-onnx VITS (vits-mms-khm) speech for the scene text",
              model="vits-mms-khm", outputs=("voice",)),
    StageSpec("voice_final", "3b · Your timbre (RVC)", "🎙️", role="rvc", resource="gpu",
              depends=("voice_base",), requires_gpu=True,
              blurb="Retrieval-based Voice Conversion → the narration in your own voice",
              model="RVC voice profile", outputs=("voice_final",)),
    StageSpec("video", "4 · Animator (Wan)", "🎞️", role="video", resource="gpu", requires_gpu=True,
              depends=("breakdown",), deferrable=True,
              blurb="Wan2.1 1.3B (or Wan2.2 5B) at 480p, silent, duration from the estimate",
              model="Wan2.1-T2V-1.3B / ComfyUI", outputs=("video",)),
    StageSpec("video_fit", "4b · Duration match", "⏱️", role="media", resource="cpu",
              depends=("video", "voice_final"), deferrable=True,
              blurb="Trim / freeze so the picture matches the finished voice exactly",
              model="ffmpeg", outputs=("video_fit",)),
    StageSpec("sfx", "5 · SFX Director", "🔊", role="sfx", resource="gpu", requires_gpu=True,
              depends=("video",), deferrable=True,
              blurb="MMAudio video-to-audio ambience from the mood tag (below-8GB mode)",
              model="MMAudio small / ComfyUI", outputs=("ambient",)),
    StageSpec("qa", "6 · QA Reviewer", "✅", role="qa", resource="llm",
              depends=("voice_final", "video_fit", "sfx"),
              blurb="Length mismatch, silence gaps, clipping, engine honesty + tone review",
              model="sailor2:8b", outputs=("qa",)),
    StageSpec("assemble", "7 · Final Assembly", "🎁", role="assembly", resource="cpu", per_scene=False,
              depends=("qa",), blurb="Mix voice + ambience under it, concat scenes, encode MP4/SRT",
              model="ffmpeg", outputs=("final",)),
]

STAGE_BY_KEY = {s.key: s for s in STAGES}
ORDER = [s.key for s in STAGES]

TERMINAL_OK = ("done", "skipped", "deferred")
TERMINAL_BAD = ("failed", "blocked", "cancelled")


def stage_spec(key):
    s = STAGE_BY_KEY.get(key)
    if s is None:
        raise KeyError(f"unknown stage '{key}'")
    return s


def job_key(stage, scene_idx):
    return f"{stage}#{int(scene_idx)}"


def parse_job_key(key):
    stage, idx = key.rsplit("#", 1)
    return stage, int(idx)


@dataclass
class Job:
    stage: str
    scene_idx: int = -1
    deps: tuple = ()
    attempts: int = 0

    @property
    def key(self):
        return job_key(self.stage, self.scene_idx)

    @property
    def spec(self):
        return STAGE_BY_KEY[self.stage]


def build_graph(scene_count, plan=None, cfg=None, only=None, ignore_done=()):
    """Jobs for a run. `only` = regenerate these stages (+everything downstream).

    Returns (jobs: dict[key → Job], order: list of keys). A `defer` engine on a
    deferrable stage keeps the job in the graph: it is resolved by the scheduler to
    a `deferred` status, which downstream stages treat as satisfied.
    """
    scene_count = max(0, int(scene_count))
    jobs = {}

    def scenes_for(s):
        return range(scene_count) if (s.per_scene and scene_count) else [-1]

    for s in STAGES:
        for idx in scenes_for(s):
            jobs[job_key(s.key, idx)] = Job(s.key, idx)
    # resolve dependencies
    for key, job in jobs.items():
        deps = []
        for dep_key in job.spec.depends:
            ds = STAGE_BY_KEY[dep_key]
            if ds.per_scene and scene_count:
                if job.scene_idx == -1:                     # run-level ← per-scene: all of them
                    deps += [job_key(dep_key, i) for i in range(scene_count)]
                else:
                    deps.append(job_key(dep_key, job.scene_idx))
            else:
                deps.append(job_key(dep_key, -1))
        jobs[key] = Job(job.stage, job.scene_idx, tuple(dict.fromkeys(deps)))
    if only:
        seeds = {k for k in jobs if parse_job_key(k)[0] in set(only)}
        keep = _downstream_closure(jobs, seeds)
        jobs = {k: v for k, v in jobs.items() if k in keep}
        # deps pointing at inherited (already-done) jobs are satisfied by the scheduler
    if ignore_done:
        for k in list(jobs):
            st, idx = parse_job_key(k)
            if (st, idx) in set(ignore_done):
                del jobs[k]
    if not scene_count:
        # Nothing has been segmented yet: only the run-level text stages can run.
        # The scheduler re-expands the graph with real scene jobs after Stage 1,
        # which is what keeps a first-ever run from launching voice#-1.
        jobs = _prune_unresolvable(jobs)
    return jobs, _topo(list(jobs.values()))


def _prune_unresolvable(jobs):
    """Drop per-scene jobs and anything that waits on them (used for scene_count=0)."""
    keep = {k: j for k, j in jobs.items() if j.scene_idx == -1 and not j.spec.per_scene}
    changed = True
    while changed:
        changed = False
        for k, j in list(keep.items()):
            if any(d not in keep for d in j.deps if d in jobs):
                del keep[k]
                changed = True
    return keep


def _downstream_closure(jobs, seeds):
    """Every job reachable from `seeds` (transitively), via the deps we just built."""
    dependents = {k: set() for k in jobs}
    for k, job in jobs.items():
        for d in job.deps:
            if d in dependents:
                dependents[d].add(k)
    seen, stack = set(seeds), list(seeds)
    while stack:
        cur = stack.pop()
        for nxt in dependents.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen | set(seeds)


def _topo(jobs):
    """Order for stable display / deterministic ready-selection."""
    rank = {s.key: i for i, s in enumerate(STAGES)}
    return sorted((j.key for j in jobs), key=lambda k: (rank[parse_job_key(k)[0]],
                                                        parse_job_key(k)[1]))


def ready_jobs(jobs, done, failed):
    """Jobs whose dependencies are all terminal (ok or failed-and-tolerated)."""
    out = []
    for key, job in jobs.items():
        if key in done or key in failed:
            continue
        if all((d in done) for d in job.deps):
            out.append(job)
    return out


def blocked_jobs(jobs, done, failed):
    """Jobs that can never run because a hard dependency failed."""
    out = []
    for key, job in jobs.items():
        if key in done or key in failed:
            continue
        if any(d in failed for d in job.deps):
            out.append(job)
    return out


def stage_labels():
    return [{"key": s.key, "title": s.title, "emoji": s.emoji, "role": s.role,
             "blurb": s.blurb, "model": s.model, "per_scene": s.per_scene,
             "requires_gpu": s.requires_gpu, "resource": s.resource,
             "depends": list(s.depends), "deferrable": s.deferrable} for s in STAGES]
