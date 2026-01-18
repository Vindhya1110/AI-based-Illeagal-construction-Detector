from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from typing import Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import json

# Mount titiler programmatically
try:
	from titiler.application.main import app as titiler
except Exception:
	titiler = None

app = FastAPI(title="HYD Change Detection Backend")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

if titiler is not None:
	app.mount("/cog", titiler)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

class DetectReq(BaseModel):
	# Accept local file paths or http(s) URLs (optional). If missing, use defaults.
	t0_path: Optional[str] = None
	t1_path: Optional[str] = None
	coordinates: Optional[str] = None
	radius_km: Optional[float] = None

@app.get("/health")
async def health():
	return {"ok": True}

@app.post("/detect")
async def detect(req: DetectReq):
    # 1) Prepare tile URLs for the viewer
	def to_cog_tiles(path: str) -> str:
		# If path is local, prefix file://
		if path.startswith("http://") or path.startswith("https://"):
			url = path
		else:
			url = f"file:///{os.path.abspath(path).replace('\\', '/')}"
		return f"http://localhost:8000/cog/tiles/{{z}}/{{x}}/{{y}}.png?url={url}"

    # Defaults when user doesn't provide datasets
    default_t0 = os.getenv("DEFAULT_T0", "/data/GEE_Exports/HYD_T0_S2_2016_clean.tif")
    default_t1 = os.getenv("DEFAULT_T1", "/data/GEE_Exports/HYD_T1_S2_2022_clean.tif")
    t0_path = req.t0_path or default_t0
    t1_path = req.t1_path or default_t1

    t0_tiles = to_cog_tiles(t0_path)
    t1_tiles = to_cog_tiles(t1_path)

    # 2) Run a lightweight change mask using ΔNDBI guarded by NDVI decrease
    def read_align(src_path: str) -> Tuple[np.ndarray, rasterio.profiles.Profile]:
        with rasterio.open(src_path) as src:
            data = src.read()  # [bands, H, W]
            profile = src.profile
        return data, profile

    def reproject_to(dst_profile, src_path: str) -> np.ndarray:
        with rasterio.open(src_path) as src:
            dst = np.zeros((src.count, dst_profile["height"], dst_profile["width"]), dtype=src.dtypes[0])
            for b in range(src.count):
                reproject(
                    source=rasterio.band(src, b + 1),
                    destination=dst[b],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_profile["transform"],
                    dst_crs=dst_profile["crs"],
                    resampling=Resampling.bilinear,
                )
        return dst

    # Load T0, align T1 to T0 grid
    t0_abs = t0_path if t0_path.startswith("/") or t0_path.startswith("http") else os.path.abspath(t0_path)
    t1_abs = t1_path if t1_path.startswith("/") or t1_path.startswith("http") else os.path.abspath(t1_path)
    t0_data, t0_prof = read_align(t0_abs)
    t1_data = reproject_to(t0_prof, t1_abs)

    # Expect export order: [B2,B3,B4,B8,B11,NDVI,NDBI]
    def safe_band(arr: np.ndarray, idx: int, fallback: np.ndarray) -> np.ndarray:
        return arr[idx] if arr.shape[0] > idx else fallback

    # Prepare indices
    # If NDVI/NDBI not present, approximate from bands
    # NDVI = (B8 - B4) / (B8 + B4)
    # NDBI = (B11 - B8) / (B11 + B8)
    def compute_indices(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        num_bands = stack.shape[0]
        if num_bands >= 7:
            ndvi = stack[5].astype("float32")
            ndbi = stack[6].astype("float32")
        else:
            b4 = safe_band(stack, 2, stack[0]).astype("float32")
            b8 = safe_band(stack, 3, stack[0]).astype("float32")
            b11 = safe_band(stack, 4, stack[0]).astype("float32")
            ndvi = (b8 - b4) / np.clip(b8 + b4, 1e-6, None)
            ndbi = (b11 - b8) / np.clip(b11 + b8, 1e-6, None)
        return ndvi, ndbi

    ndvi0, ndbi0 = compute_indices(t0_data)
    ndvi1, ndbi1 = compute_indices(t1_data)

    d_ndbi = ndbi1 - ndbi0
    d_ndvi = ndvi1 - ndvi0

    # Heuristic: urbanization -> NDBI increases and NDVI decreases
    score = d_ndbi - 0.5 * np.maximum(d_ndvi, 0)

    # Threshold by high percentile to keep top changes, remove tiny speckles
    th = float(np.nanpercentile(score, 99))
    mask = (score > max(th, 0.1)).astype(np.uint8)

    # Vectorize mask
    geoms = []
    for geom, val in shapes(mask, mask=mask == 1, transform=t0_prof["transform"]):
        if val != 1:
            continue
        geoms.append(shape(geom))

    # Filter small polygons (< 200 m²) using projected CRS (UTM 44N)
    from shapely.ops import transform as shp_transform
    import pyproj

    proj_to = pyproj.Transformer.from_crs(t0_prof["crs"], "EPSG:32644", always_xy=True).transform
    proj_from = pyproj.Transformer.from_crs("EPSG:32644", t0_prof["crs"], always_xy=True).transform

    features = []
    for g in geoms:
        g_utm = shp_transform(proj_to, g)
        area_m2 = g_utm.area
        if area_m2 < 200:
            continue
        risk = "High" if area_m2 > 2000 else ("Medium" if area_m2 > 600 else "Low")
        features.append({
            "type": "Feature",
            "geometry": mapping(g),
            "properties": {"area_m2": round(float(area_m2), 2), "risk": risk},
        })

    fc = {"type": "FeatureCollection", "features": features}
    cases_path = os.path.join(STATIC_DIR, "HYD_change_cases.geojson")
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(fc, f)

    cases_url = f"http://localhost:8000/static/HYD_change_cases.geojson"

    return {
        "t0_tiles": t0_tiles,
        "t1_tiles": t1_tiles,
        "cases_geojson": cases_url,
        "center": {"lat": 17.4102, "lon": 78.4747},
    }

# Simple static files with Uvicorn
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ===================== Pretrained model (optional) =====================
MODEL_READY = False
try:
    import torch
    from mmseg.apis import init_model, inference_model
    MODEL_CFG = os.getenv(
        "CHANGEFORMER_CFG",
        "https://download.openmmlab.com/mmseg/changeformer/changeformerB_scratch_512x512_40k_levir-cd.py",
    )
    MODEL_CKPT = os.getenv(
        "CHANGEFORMER_CKPT",
        "https://download.openmmlab.com/mmseg/changeformer/changeformerB_scratch_512x512_40k_levir-cd_20230719.pth",
    )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _model = None

    def get_model():
        global _model, MODEL_READY
        if _model is None:
            _model = init_model(MODEL_CFG, MODEL_CKPT, device=DEVICE)
            MODEL_READY = True
        return _model
except Exception:
    _model = None
    MODEL_READY = False


def _read_rgb_scaled(path: str):
    with rasterio.open(path) as src:
        # Expect export order: B2,B3,B4,... so RGB=B4,B3,B2
        b2 = src.read(1).astype("float32")
        b3 = src.read(2).astype("float32")
        b4 = src.read(3).astype("float32")
        rgb = np.stack([b4, b3, b2], axis=0)
        rgb = np.clip(rgb / 0.3, 0, 1)
        profile = src.profile
    return rgb, profile


@app.post("/detect-ml")
async def detect_ml(req: DetectReq):
    if not MODEL_READY:
        return {"detail": "Model not available on server. Install mmseg/torch or use Docker image with ML."}

    default_t0 = os.getenv("DEFAULT_T0", "/data/GEE_Exports/HYD_T0_S2_2016_clean.tif")
    default_t1 = os.getenv("DEFAULT_T1", "/data/GEE_Exports/HYD_T1_S2_2022_clean.tif")
    t0_path = req.t0_path or default_t0
    t1_path = req.t1_path or default_t1

    t0_abs = t0_path if t0_path.startswith("/") or t0_path.startswith("http") else os.path.abspath(t0_path)
    t1_abs = t1_path if t1_path.startswith("/") or t1_path.startswith("http") else os.path.abspath(t1_path)

    # Align T1 to T0 grid
    _, t0_prof = _read_rgb_scaled(t0_abs)
    t1_aligned = reproject_to(t0_prof, t1_abs)
    # Build RGB arrays
    t0_full, _ = _read_rgb_scaled(t0_abs)
    # t1_aligned is full stack; build RGB as B4,B3,B2 order indices 2,1,0 after alignment
    b2 = t1_aligned[0].astype("float32"); b3 = t1_aligned[1].astype("float32"); b4 = t1_aligned[2].astype("float32")
    t1_full = np.stack([np.clip(b4/0.3,0,1), np.clip(b3/0.3,0,1), np.clip(b2/0.3,0,1)], axis=0)

    H, W = t0_full.shape[1], t0_full.shape[2]
    tile = 512
    stride = 480
    pred_mask = np.zeros((H, W), dtype=np.uint8)

    model = get_model()

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2 = min(y + tile, H)
            x2 = min(x + tile, W)
            img0 = t0_full[:, y:y2, x:x2].transpose(1,2,0)
            img1 = t1_full[:, y:y2, x:x2].transpose(1,2,0)
            data = {'img': [img0, img1]}
            try:
                out = inference_model(model, data)
                logits = out.pred_sem_seg.data[0].softmax(dim=0)
                prob = logits[1].cpu().numpy()
                mask = (prob > 0.5).astype(np.uint8)
            except Exception:
                mask = np.zeros((y2 - y, x2 - x), dtype=np.uint8)
            pred_mask[y:y2, x:x2] = np.maximum(pred_mask[y:y2, x:x2], mask)

    # Vectorize and save like /detect
    features = []
    for geom, val in shapes(pred_mask, mask=pred_mask == 1, transform=t0_prof["transform"]):
        if val != 1:
            continue
        g = shape(geom)
        features.append({"type": "Feature", "geometry": mapping(g), "properties": {"risk": "High"}})

    fc = {"type": "FeatureCollection", "features": features}
    cases_path = os.path.join(STATIC_DIR, "HYD_change_cases_ml.geojson")
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(fc, f)

    def to_cog_tiles(path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            url = path
        else:
            url = f"file:///{os.path.abspath(path).replace('\\', '/')}"
        return f"http://localhost:8000/cog/tiles/{{z}}/{{x}}/{{y}}.png?url={url}"

    return {
        "t0_tiles": to_cog_tiles(t0_path),
        "t1_tiles": to_cog_tiles(t1_path),
        "cases_geojson": f"http://localhost:8000/static/HYD_change_cases_ml.geojson",
        "center": {"lat": 17.4102, "lon": 78.4747},
    }
