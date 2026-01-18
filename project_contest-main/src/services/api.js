const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function runDetection(payload) {
  const res = await fetch(`${API_BASE}/detect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Detection failed: ${res.status}`);
  return res.json();
}

export function buildViewerUrl({ t0Tiles, t1Tiles, casesUrl, lat, lon, z }) {
  const params = new URLSearchParams({
    t0: t0Tiles,
    t1: t1Tiles,
    cases: casesUrl || "",
    lat: String(lat ?? 17.4102),
    lon: String(lon ?? 78.4747),
    z: String(z ?? 12),
  });
  return `/viewer?${params.toString()}`;
}


