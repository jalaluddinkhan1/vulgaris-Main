"""VULGARIS production inference server with auth, metrics, degradation, and versioning."""
import os
import sys
import uuid
import time
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

# Ensure project root is importable when running as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve.auth import require_api_key
from serve.logging_setup import setup_logging, get_logger
from serve import metrics as _metrics
from serve.degradation import (
    CanaryController, DeploymentMode,
    DegradationController, DegradationLevel,
)
from serve.audit import AuditLogger
from serve.versioning import ModelVersionRegistry

from .streaming import StreamingInference
from model.vulgaris import Vulgaris
from config import ModelConfig

# ---------------------------------------------------------------------------
# Module-level singletons (initialised at import time so uvicorn workers share
# a consistent startup path; the FastAPI lifespan hook re-calls _setup_model).
# ---------------------------------------------------------------------------

_log = get_logger("server")

_degradation = DegradationController()
_canary = CanaryController(mode=DeploymentMode.PRIMARY)
_audit = AuditLogger(
    path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "audit",
        "predictions.jsonl",
    )
)

_version_registry = ModelVersionRegistry(
    index_path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints",
        "version_index.json",
    )
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    data: List[List[float]]   # (in_channels, T) or (T,) for single channel
    domain_id: int = 0
    mode: str = "predictive"
    return_uncertainty: bool = True
    return_attribution: bool = False


class PredictResponse(BaseModel):
    prediction: List[float]
    uncertainty: Optional[float]
    attribution: Optional[List[float]]
    latency_ms: float
    step: int
    degradation_level: str


class StreamRequest(BaseModel):
    domain_id: int = 0
    mode: str = "predictive"


class StepRequest(BaseModel):
    values: List[float]       # (in_channels,) single timestep
    domain_id: int = 0


class RegisterDomainRequest(BaseModel):
    domain_name: str
    metadata: dict = {}


class ExplainRequest(BaseModel):
    session_id: Optional[str] = None
    n_rules: int = 5


class CounterfactualRequest(BaseModel):
    data: List[List[float]]
    target: List[float]
    domain_id: int = 0
    n_steps: int = 50
    lr: float = 0.01


class VersionRegisterRequest(BaseModel):
    path: str
    tag: str
    metadata: Optional[dict] = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _build_config() -> ModelConfig:
    """Build ModelConfig from environment variables."""
    cfg = ModelConfig()
    cfg.input_dim = int(os.environ.get("VULGARIS_INPUT_DIM", cfg.input_dim))
    cfg.n_classes = int(os.environ.get("VULGARIS_N_CLASSES", cfg.n_classes))
    cfg.output_dim = cfg.n_classes if cfg.n_classes > 0 else cfg.input_dim
    d_model = int(os.environ.get("VULGARIS_D_MODEL", 64))
    cfg.ase.latent_dim = d_model * 4
    cfg.sssr.state_dim = d_model * 4
    cfg.sssr.d_inner = d_model * 8
    cfg.hmb.embed_dim = d_model * 4
    cfg.hmb.compress_dim = d_model
    return cfg


def _setup_model(config: ModelConfig) -> Vulgaris:
    """Load Vulgaris model; optionally restore weights from .npz checkpoint."""
    model = Vulgaris(config)
    ckpt_path = os.environ.get("VULGARIS_CHECKPOINT", "")
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            data = np.load(ckpt_path, allow_pickle=False)
            params = model.parameters()
            for i, p in enumerate(params):
                key = f"param_{i}"
                if key in data:
                    p.data = data[key].astype(np.float32)
            _log.info("Checkpoint loaded", extra={"ctx_path": ckpt_path})
        except Exception as exc:
            _log.warning(
                "Checkpoint load failed — using random weights",
                extra={"ctx_path": ckpt_path, "ctx_error": str(exc)},
            )
    return model


# Initialise app-level state
_config = _build_config()
_model = _setup_model(_config)
_engines: Dict[str, StreamingInference] = {}
_domain_registry: Dict[str, dict] = {}
_next_domain_idx: int = 0

app = FastAPI(title="VULGARIS Inference API", version="1.0.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_check(request: Request):
    valid, key = require_api_key(request)
    if not valid:
        _metrics.requests_errors.inc()
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _make_engine(mode: str) -> StreamingInference:
    return StreamingInference(
        model=_model,
        batch_size=1,
        normalize_input=True,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request):
    _auth_check(request)

    request_id = str(uuid.uuid4())
    t0 = time.time()
    _metrics.requests_total.inc()
    _metrics.active_requests.inc()

    deg_level = _degradation.level

    try:
        # ALERT_ONLY: return anomaly score only, no classification
        if deg_level == DegradationLevel.ALERT_ONLY:
            data = np.array(req.data, dtype=np.float32)
            if data.ndim == 1:
                data = data[None, :]
            anomaly_score = float(np.mean(np.abs(data)))
            latency_s = time.time() - t0
            _degradation.record(latency_s, is_error=False)
            _metrics.request_latency.observe(latency_s)
            _log.info(
                "predict alert_only",
                extra={
                    "ctx_request_id": request_id,
                    "ctx_domain": req.domain_id,
                    "ctx_latency_ms": round(latency_s * 1000, 2),
                    "ctx_degradation": deg_level.value,
                },
            )
            return PredictResponse(
                prediction=[anomaly_score],
                uncertainty=None,
                attribution=None,
                latency_ms=round(latency_s * 1000, 2),
                step=0,
                degradation_level=deg_level.value,
            )

        data = np.array(req.data, dtype=np.float32)
        if data.ndim == 1:
            data = data[None, :]
        x_window = data[None, :, :]  # (1, in_channels, T)

        engine = _make_engine(req.mode)
        engine.switch_mode(req.mode)

        # REDUCED: skip ESE + DAH (domain_idx forced to 0)
        domain_idx = 0 if deg_level == DegradationLevel.REDUCED else req.domain_id
        result = engine.process_window(x_window, domain_idx=domain_idx)

        pred = result["final_prediction"]
        pred_list = pred.flatten().tolist()
        uncertainty = float(result["uncertainty"]) if req.return_uncertainty else None

        attribution = None
        if req.return_attribution and deg_level == DegradationLevel.FULL:
            try:
                rules = _model.ese.get_rules(max_rules=5)
                if rules:
                    attribution = [float(r.get("importance", 0.0)) for r in rules[: len(pred_list)]]
                    while len(attribution) < len(pred_list):
                        attribution.append(0.0)
                else:
                    attribution = [0.0] * len(pred_list)
            except Exception:
                attribution = [0.0] * len(pred_list)

        latency_s = time.time() - t0
        _degradation.record(latency_s, is_error=False)
        _metrics.request_latency.observe(latency_s)

        _log.info(
            "predict ok",
            extra={
                "ctx_request_id": request_id,
                "ctx_domain": req.domain_id,
                "ctx_latency_ms": round(latency_s * 1000, 2),
                "ctx_degradation": deg_level.value,
            },
        )

        _audit.record(
            x_raw=data,
            prediction=np.array(pred_list),
            step=result["steps"],
            domain_idx=domain_idx,
        )

        return PredictResponse(
            prediction=pred_list,
            uncertainty=uncertainty,
            attribution=attribution,
            latency_ms=round(result["total_latency_ms"], 2),
            step=result["steps"],
            degradation_level=deg_level.value,
        )

    except MemoryError:
        latency_s = time.time() - t0
        _metrics.requests_errors.inc()
        _degradation.record(latency_s, is_error=True)
        _log.error(
            "predict OOM",
            extra={"ctx_request_id": request_id, "ctx_latency_ms": round(latency_s * 1000, 2)},
        )
        raise HTTPException(status_code=503, detail="Out of memory — try a smaller input window")

    except Exception as exc:
        latency_s = time.time() - t0
        _metrics.requests_errors.inc()
        _degradation.record(latency_s, is_error=True)
        _log.error(
            "predict error",
            extra={
                "ctx_request_id": request_id,
                "ctx_error": str(exc),
                "ctx_latency_ms": round(latency_s * 1000, 2),
            },
        )
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        _metrics.active_requests.dec()


@app.post("/stream/start")
async def stream_start(req: StreamRequest, request: Request):
    _auth_check(request)
    session_id = str(uuid.uuid4())
    engine = _make_engine(req.mode)
    engine.switch_mode(req.mode)
    engine.set_domain(req.domain_id)
    _engines[session_id] = engine
    _log.info("stream started", extra={"ctx_session_id": session_id, "ctx_domain": req.domain_id})
    return {"session_id": session_id, "domain_id": req.domain_id, "mode": req.mode}


@app.post("/stream/{session_id}/step")
async def stream_step(session_id: str, req: StepRequest, request: Request):
    _auth_check(request)
    if session_id not in _engines:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    engine = _engines[session_id]
    x_t = np.array(req.values, dtype=np.float32)
    t0 = time.time()
    _metrics.requests_total.inc()
    _metrics.active_requests.inc()
    try:
        result = engine.step(x_t, domain_idx=req.domain_id)
        latency_s = time.time() - t0
        _metrics.request_latency.observe(latency_s)
        _degradation.record(latency_s, is_error=False)
        return {
            "prediction": result["prediction"].flatten().tolist(),
            "uncertainty": result["uncertainty"],
            "latency_ms": result["latency_ms"],
            "step": result["step"],
        }
    except Exception as exc:
        _metrics.requests_errors.inc()
        _degradation.record(time.time() - t0, is_error=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _metrics.active_requests.dec()


@app.get("/stream/{session_id}/stats")
async def stream_stats(session_id: str, request: Request):
    _auth_check(request)
    if session_id not in _engines:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    engine = _engines[session_id]
    latency_stats = engine.get_latency_stats()
    return {
        "session_id": session_id,
        "step_count": engine.step_count,
        "latency": latency_stats,
        "mode": engine.mode,
    }


@app.delete("/stream/{session_id}")
async def stream_end(session_id: str, request: Request):
    _auth_check(request)
    if session_id not in _engines:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    del _engines[session_id]
    _log.info("stream closed", extra={"ctx_session_id": session_id})
    return {"status": "closed", "session_id": session_id}


@app.get("/health")
async def health():
    deg_status = _degradation.status()
    return {
        "status": "ok",
        "active_sessions": len(_engines),
        "model_params": _model.n_base_params(),
        "timestamp": time.time(),
        "degradation": deg_status,
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    return PlainTextResponse(
        content=_metrics.REGISTRY.exposition_text(),
        media_type="text/plain; version=0.0.4",
    )


@app.get("/degradation")
async def degradation_status():
    return _degradation.status()


@app.get("/versions")
async def list_versions(request: Request):
    _auth_check(request)
    active = _version_registry.get_active()
    return {
        "versions": _version_registry.list_versions(),
        "active": active,
    }


@app.post("/versions/register")
async def register_version(req: VersionRegisterRequest, request: Request):
    _auth_check(request)
    try:
        entry = _version_registry.register(req.path, req.tag, req.metadata)
        _log.info("version registered", extra={"ctx_tag": req.tag, "ctx_path": req.path})
        return entry
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/versions/{tag}/promote")
async def promote_version(tag: str, request: Request):
    _auth_check(request)
    try:
        entry = _version_registry.promote(tag)
        _log.info("version promoted", extra={"ctx_tag": tag})
        return {"status": "promoted", "version": entry}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/versions/rollback")
async def rollback_version(request: Request):
    _auth_check(request)
    entry = _version_registry.rollback()
    if entry is None:
        raise HTTPException(status_code=409, detail="No previous version to roll back to")
    _log.info("version rolled back", extra={"ctx_tag": entry.get("tag")})
    return {"status": "rolled_back", "version": entry}


@app.get("/domains")
async def list_domains(request: Request):
    _auth_check(request)
    return {
        "domains": _domain_registry,
        "n_domains": len(_domain_registry),
        "configured_domains": _config.dah.n_domains,
    }


@app.post("/domains/register")
async def register_domain(req: RegisterDomainRequest, request: Request):
    _auth_check(request)
    global _next_domain_idx
    if req.domain_name in _domain_registry:
        existing = _domain_registry[req.domain_name]
        return {"domain_name": req.domain_name, "domain_idx": existing["domain_idx"], "status": "existing"}
    domain_idx = _next_domain_idx
    if domain_idx >= _config.dah.n_domains:
        raise HTTPException(status_code=400, detail="Maximum number of domains reached")
    _domain_registry[req.domain_name] = {
        "domain_idx": domain_idx,
        "metadata": req.metadata,
        "registered_at": time.time(),
    }
    _next_domain_idx += 1
    return {"domain_name": req.domain_name, "domain_idx": domain_idx, "status": "registered"}


@app.post("/explain")
async def explain(req: ExplainRequest, request: Request):
    _auth_check(request)
    try:
        rules = _model.ese.get_rules(max_rules=req.n_rules)
        return {
            "rules": rules,
            "n_rules": len(rules),
            "session_id": req.session_id,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/counterfactual")
async def counterfactual(req: CounterfactualRequest, request: Request):
    _auth_check(request)
    try:
        data = np.array(req.data, dtype=np.float32)
        target = np.array(req.target, dtype=np.float32)
        if data.ndim == 1:
            data = data[None, :]
        x_window = data[None, :, :]   # (1, in_channels, T)
        target_t = target[None, :]    # (1, output_dim)

        cf_data = x_window.copy()
        lr = req.lr

        for step_i in range(req.n_steps):
            engine_cf = StreamingInference(
                model=_model, batch_size=1,
                normalize_input=False, mode="predictive",
            )
            result_cf = engine_cf.process_window(cf_data, domain_idx=req.domain_id)
            pred = result_cf["final_prediction"]
            loss = float(np.mean((pred - target_t) ** 2))
            if loss < 1e-6:
                break
            eps = 1e-3
            grad = np.zeros_like(cf_data)
            for i in range(min(cf_data.shape[1], 4)):
                for t in range(cf_data.shape[2]):
                    cf_plus = cf_data.copy()
                    cf_plus[0, i, t] += eps
                    r_plus = StreamingInference(
                        model=_model, batch_size=1,
                        normalize_input=False, mode="predictive",
                    ).process_window(cf_plus, domain_idx=req.domain_id)
                    loss_plus = float(np.mean((r_plus["final_prediction"] - target_t) ** 2))
                    grad[0, i, t] = (loss_plus - loss) / eps
            cf_data = cf_data - lr * grad

        return {
            "counterfactual": cf_data.tolist(),
            "original": x_window.tolist(),
            "target": target.tolist(),
            "n_steps": req.n_steps,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Audit + deployment endpoints
# ---------------------------------------------------------------------------


@app.get("/audit")
async def audit_tail(request: Request, n: int = 100):
    _auth_check(request)
    return {
        "records": _audit.tail(n),
        "total_written": _audit.count(),
    }


@app.post("/deployment/mode")
async def set_deployment_mode(request: Request):
    _auth_check(request)
    body = await request.json()
    raw_mode = body.get("mode", "primary")
    try:
        mode = DeploymentMode(raw_mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode '{raw_mode}'. Valid: primary, canary, shadow",
        )
    canary_pct = float(body.get("canary_pct", _canary.canary_pct))
    _canary.canary_pct = canary_pct
    _canary.set_mode(mode)
    _log.info("deployment mode changed", extra={"ctx_mode": mode.value, "ctx_canary_pct": canary_pct})
    return {"mode": mode.value, "canary_pct": canary_pct}


@app.get("/deployment/shadow-stats")
async def shadow_stats(request: Request):
    _auth_check(request)
    return _canary.shadow_stats()


# ---------------------------------------------------------------------------
# Legacy class-based wrapper kept for backward-compatibility
# ---------------------------------------------------------------------------


class InferenceServer:
    """Thin wrapper retained for any code that instantiates InferenceServer directly."""

    def __init__(self, model_path: str, config: ModelConfig,
                 host: str = "0.0.0.0", port: int = 8000):
        global _model, _config
        self.host = host
        self.port = port
        _config = config
        _model = _setup_model(config)
        self.app = app

    def run(self):
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)
