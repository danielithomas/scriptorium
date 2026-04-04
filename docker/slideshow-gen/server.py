#!/usr/bin/env python3
"""Slideshow Generator — FastAPI Service

Persistent HTTP API wrapping the slideshow pipeline.
Accepts jobs via REST, runs them sequentially on the GPU,
and serves completed videos.
"""

import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# ── Configuration ──────────────────────────────────────────────

JOBS_DIR = Path(os.environ.get("JOBS_DIR", "/data/jobs"))
MAX_STORED_JOBS = int(os.environ.get("MAX_STORED_JOBS", "20"))
HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
PORT = int(os.environ.get("SERVER_PORT", "8080"))

JOBS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("slideshow")

# ── Job State ──────────────────────────────────────────────────

STAGES = [
    "Narration Synthesis",
    "Image Preparation",
    "Music Generation",
    "Audio Mixing",
    "Video Segments + Overlays",
    "Final Assembly",
]

job_queue: Queue = Queue()
jobs_lock = threading.Lock()


def _status_path(job_id: str) -> Path:
    return JOBS_DIR / job_id / "status.json"


def _read_status(job_id: str) -> Optional[dict]:
    p = _status_path(job_id)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _write_status(job_id: str, status: dict):
    p = _status_path(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(status, f, indent=2)


def _list_jobs() -> list[dict]:
    """List all jobs sorted by creation time (newest first)."""
    results = []
    if not JOBS_DIR.exists():
        return results
    for d in JOBS_DIR.iterdir():
        if d.is_dir():
            st = _read_status(d.name)
            if st:
                results.append(st)
    results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return results


def _cleanup_old_jobs():
    """Remove oldest completed/failed jobs beyond MAX_STORED_JOBS."""
    all_jobs = _list_jobs()
    if len(all_jobs) <= MAX_STORED_JOBS:
        return
    # Keep running/queued, prune oldest completed/failed
    removable = [j for j in all_jobs if j["status"] in ("completed", "failed")]
    to_remove = removable[MAX_STORED_JOBS:]  # oldest beyond limit
    for j in to_remove:
        jid = j["job_id"]
        job_dir = JOBS_DIR / jid
        if job_dir.exists():
            shutil.rmtree(job_dir)
            log.info(f"Cleaned up old job {jid}")


# ── Pipeline Runner ────────────────────────────────────────────

def _run_pipeline(job_id: str):
    """Execute the 6-stage pipeline for a job."""
    job_dir = JOBS_DIR / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = input_dir / "script.json"
    output_path = output_dir / "final.mp4"
    working_dir = job_dir / "working"
    working_dir.mkdir(parents=True, exist_ok=True)
    (working_dir / "slides").mkdir(exist_ok=True)
    (working_dir / "narration").mkdir(exist_ok=True)
    (working_dir / "segments").mkdir(exist_ok=True)

    # Read script for options
    with open(script_path) as f:
        script = json.load(f)

    # Check for optional files
    voice_ref = input_dir / "voice_ref.wav"
    music_txt = input_dir / "music.txt"

    env = {**os.environ}
    music_vol = env.get("MUSIC_VOLUME", "0.2")

    stages = [
        # Narration FIRST — determines actual slide durations
        (1, "Narration Synthesis", [
            "python3", "/app/scripts/stage_narration.py",
            str(script_path), str(working_dir / "narration"),
        ] + (["--voice-ref=" + str(voice_ref)] if voice_ref.exists() else [])),
        (2, "Image Preparation", [
            "python3", "/app/scripts/stage_images.py",
            str(script_path), str(working_dir / "slides"),
        ]),
        (3, "Music Generation", [
            "python3", "/app/scripts/stage_music.py",
            str(script_path), str(working_dir / "music.wav"),
        ] + (["--prompt=" + music_txt.read_text().strip()] if music_txt.exists() else [])),
        (4, "Audio Mixing", [
            "python3", "/app/scripts/stage_mix.py",
            str(script_path), str(working_dir),
            f"--volume={music_vol}",
        ]),
        (5, "Video Segments + Overlays", [
            "python3", "/app/scripts/stage_segments.py",
            str(script_path), str(working_dir),
        ]),
        (6, "Final Assembly", [
            "python3", "/app/scripts/stage_assemble.py",
            str(script_path), str(working_dir), str(output_path),
        ]),
    ]

    status = _read_status(job_id)
    status["status"] = "running"
    status["started_at"] = datetime.now(timezone.utc).isoformat()
    _write_status(job_id, status)

    for stage_num, stage_name, cmd in stages:
        log.info(f"[{job_id}] Stage {stage_num}/6: {stage_name}")
        status["progress"] = {"stage": stage_num, "total": 6, "name": stage_name}
        _write_status(job_id, status)

        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min max per stage
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Stage {stage_num} failed (exit {result.returncode}):\n"
                    f"stdout: {result.stdout[-2000:]}\n"
                    f"stderr: {result.stderr[-2000:]}"
                )
            log.info(f"[{job_id}] Stage {stage_num}/6 complete")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Stage {stage_num} ({stage_name}) timed out after 1800s")

    # Cleanup working dir
    shutil.rmtree(working_dir, ignore_errors=True)

    status["status"] = "completed"
    status["completed_at"] = datetime.now(timezone.utc).isoformat()
    status["progress"] = {"stage": 6, "total": 6, "name": "Complete"}
    _write_status(job_id, status)
    log.info(f"[{job_id}] Pipeline complete → {output_path}")


def _worker():
    """Background worker — processes jobs sequentially."""
    while True:
        job_id = job_queue.get()
        try:
            _run_pipeline(job_id)
        except Exception as e:
            log.error(f"[{job_id}] Pipeline failed: {e}")
            status = _read_status(job_id) or {}
            status["status"] = "failed"
            status["error"] = str(e)
            status["completed_at"] = datetime.now(timezone.utc).isoformat()
            _write_status(job_id, status)
            # Cleanup working dir on failure too
            working = JOBS_DIR / job_id / "working"
            shutil.rmtree(working, ignore_errors=True)
        finally:
            job_queue.task_done()
            _cleanup_old_jobs()


# Start background worker thread
worker_thread = threading.Thread(target=_worker, daemon=True)
worker_thread.start()

# ── Restore queued jobs on startup ─────────────────────────────

def _restore_jobs():
    """Re-queue any jobs that were queued/running when the service stopped."""
    for st in _list_jobs():
        if st["status"] in ("queued", "running"):
            jid = st["job_id"]
            log.info(f"Restoring interrupted job {jid}")
            st["status"] = "queued"
            _write_status(jid, st)
            job_queue.put(jid)

_restore_jobs()

# ── FastAPI App ────────────────────────────────────────────────

app = FastAPI(
    title="Slideshow Generator",
    description="AI-powered slideshow video generator with TTS narration and music",
    version="2.0.0",
)


@app.get("/health")
def health():
    """Service health + GPU status."""
    gpu_info = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            gpu_info = {
                "name": parts[0],
                "memory_used_mb": int(parts[1]),
                "memory_total_mb": int(parts[2]),
                "utilization_pct": int(parts[3]),
            }
    except Exception:
        pass

    queue_size = job_queue.qsize()
    running = [j for j in _list_jobs() if j["status"] == "running"]

    return {
        "status": "healthy",
        "gpu": gpu_info,
        "queue_depth": queue_size,
        "running_jobs": len(running),
        "device": os.environ.get("DEVICE", "cpu"),
        "encoder": os.environ.get("FFMPEG_ENCODER", "libx264"),
    }


@app.post("/generate")
async def generate(
    script: UploadFile = File(..., description="script.json file"),
    images: list[UploadFile] = File(default=[], description="Slide image files"),
    voice_ref: Optional[UploadFile] = File(default=None, description="Voice reference WAV"),
    logo: Optional[UploadFile] = File(default=None, description="Logo image for intro/outro slides"),
    music_prompt: Optional[str] = Form(default=None, description="Music generation prompt"),
    no_music: bool = Form(default=False, description="Skip music generation"),
    no_narration: bool = Form(default=False, description="Skip TTS narration"),
    subtitles: bool = Form(default=False, description="Generate SRT subtitles"),
):
    """Submit a new slideshow generation job."""
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    input_dir = job_dir / "input"
    images_dir = input_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Save script.json
    script_content = await script.read()
    try:
        script_data = json.loads(script_content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in script file")
    (input_dir / "script.json").write_bytes(script_content)

    # Save images
    for img in images:
        img_content = await img.read()
        (images_dir / img.filename).write_bytes(img_content)

    # Save voice reference
    if voice_ref:
        voice_content = await voice_ref.read()
        (input_dir / "voice_ref.wav").write_bytes(voice_content)

    # Save logo for intro/outro
    if logo:
        logo_content = await logo.read()
        logo_filename = logo.filename or "logo.png"
        (images_dir / logo_filename).write_bytes(logo_content)
        # Auto-add intro/outro to script if not already present
        if "intro" not in script_data:
            script_data["intro"] = {"image": logo_filename, "duration": 3, "background": "white"}
        if "outro" not in script_data:
            script_data["outro"] = {"image": logo_filename, "duration": 3, "background": "white"}
        # Re-write script with intro/outro added
        (input_dir / "script.json").write_text(json.dumps(script_data, indent=2))

    # Save music prompt
    if music_prompt:
        (input_dir / "music.txt").write_text(music_prompt)

    # Create job status
    status = {
        "job_id": job_id,
        "status": "queued",
        "progress": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "options": {
            "no_music": no_music,
            "no_narration": no_narration,
            "subtitles": subtitles,
            "has_voice_ref": voice_ref is not None,
            "slide_count": len(script_data.get("slides", [])),
            "title": script_data.get("title", "Untitled"),
        },
    }
    _write_status(job_id, status)

    # Queue the job
    job_queue.put(job_id)
    log.info(f"Queued job {job_id}: {status['options']['title']} ({status['options']['slide_count']} slides)")

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs")
def list_jobs(limit: int = 20):
    """List recent jobs."""
    all_jobs = _list_jobs()
    return {"jobs": all_jobs[:limit]}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Get job status and progress."""
    status = _read_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@app.get("/jobs/{job_id}/output")
def get_output(job_id: str):
    """Download the completed video."""
    status = _read_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    if status["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is {status['status']}, not completed",
        )

    output_path = JOBS_DIR / job_id / "output" / "final.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    title = status.get("options", {}).get("title", "slideshow")
    safe_title = "".join(c if c.isalnum() or c in "- _" else "_" for c in title)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"{safe_title}.mp4",
    )


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    """Delete a completed/failed job, or cancel a queued one."""
    status = _read_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    if status["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a running job. Wait for completion or restart the service.",
        )

    job_dir = JOBS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    log.info(f"Deleted job {job_id}")
    return {"deleted": job_id}


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info(f"Starting Slideshow Generator API on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
