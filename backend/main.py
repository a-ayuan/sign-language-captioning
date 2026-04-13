from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .inference_service import InferenceService
from .schemas import InferenceResponse


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "transformer_wlasl_ctc.yaml"
CHECKPOINT_PATH = ROOT / "outputs" / "runs" / "transformer_wlasl_ctc" / "checkpoints" / "best.pt"
FRONTEND_DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="Sign Language Captioning UI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service: InferenceService | None = None


@app.on_event("startup")
def startup_event() -> None:
    global service
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Config file not found: {CONFIG_PATH}")
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint file not found: {CHECKPOINT_PATH}")
    service = InferenceService(str(CONFIG_PATH), str(CHECKPOINT_PATH))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/api/predict-video", response_model=InferenceResponse)
async def predict_video(file: UploadFile = File(...)) -> InferenceResponse:
    if service is None:
        raise HTTPException(status_code=500, detail="Inference service not initialized")

    allowed_suffixes = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    try:
        return await service.run_uploaded_video(file)
    except Exception as exc:
        print("predict-video error:", repr(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")