import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { runDetection, buildViewerUrl } from "../services/api";

export default function SiteCheck() {
  const navigate = useNavigate();
  const [coordinates, setCoordinates] = useState("");
  const [radius, setRadius] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      // Expect local file paths or http(s) URLs to your two GeoTIFFs (COGs or plain tifs)
      const payload = {
        // t0/t1 omitted — backend will use defaults
        coordinates,
        radius_km: radius ? Number(radius) : undefined,
      };
      const res = await runDetection(payload);
      const url = buildViewerUrl({
        t0Tiles: res.t0_tiles,
        t1Tiles: res.t1_tiles,
        casesUrl: res.cases_geojson,
        lat: res.center?.lat,
        lon: res.center?.lon,
        z: 12,
      });
      navigate(url);
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="d-flex flex-column align-items-center px-4 pb-5">
      <div className="w-100" style={{ maxWidth: "700px" }}>
        <div className="card p-4 mt-4">
          <h2 className="text-center mb-4">🛰 Satellite Site Check</h2>

          <form onSubmit={handleSubmit} className="d-flex flex-column gap-3 mb-4">
            <input
              type="text"
              placeholder="Enter coordinates (e.g., 17.3850, 78.4867)"
              value={coordinates}
              onChange={(e) => setCoordinates(e.target.value)}
              className="form-control"
            />
            <input
              type="number"
              placeholder="Radius (km)"
              value={radius}
              onChange={(e) => setRadius(e.target.value)}
              className="form-control"
            />
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? "Running detection…" : "Run Detection"}
            </button>
          </form>

          <hr />

          {error && (
            <div className="alert alert-danger" role="alert">{error}</div>
          )}

          <div className="bg-body-secondary rounded-3 d-flex align-items-center justify-content-center text-secondary" style={{ height: "350px" }}>
            Satellite imagery preview will appear here
          </div>
        </div>
      </div>
    </div>
  );
}
