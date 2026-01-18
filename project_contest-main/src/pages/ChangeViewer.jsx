import React, { useEffect, useMemo, useRef, useState } from "react";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import "leaflet-side-by-side";

// Utility to read query params with defaults
function useQueryParams() {
  return useMemo(() => new URLSearchParams(window.location.search), []);
}

const defaultCenter = [17.4102, 78.4747];
const defaultZoom = 12;

export default function ChangeViewer() {
  const mapRef = useRef(null);
  const sideRef = useRef(null);
  const [error, setError] = useState(null);
  const params = useQueryParams();

  const t0Url = params.get("t0");
  const t1Url = params.get("t1");
  const geojsonUrl = params.get("cases");
  const lat = parseFloat(params.get("lat") || `${defaultCenter[0]}`);
  const lon = parseFloat(params.get("lon") || `${defaultCenter[1]}`);
  const zoom = parseInt(params.get("z") || `${defaultZoom}`, 10);

  useEffect(() => {
    try {
      const map = L.map("map-change-viewer", {
        center: [lat, lon],
        zoom,
      });
      mapRef.current = map;

      // Basemap
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "&copy; OpenStreetMap contributors",
      }).addTo(map);

      // Before/After layers
      const t0 = t0Url
        ? L.tileLayer(t0Url, { attribution: "T0" })
        : L.tileLayer(
            "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            { attribution: "T0 placeholder" }
          );

      const t1 = t1Url
        ? L.tileLayer(t1Url, { attribution: "T1" })
        : L.tileLayer(
            "https://{s}.tile.openstreetmap.de/{z}/{x}/{y}.png",
            { attribution: "T1 placeholder" }
          );

      t0.addTo(map);
      t1.addTo(map);
      sideRef.current = L.control.sideBySide(t0, t1).addTo(map);

      // Load GeoJSON cases overlay if provided
      if (geojsonUrl) {
        fetch(geojsonUrl)
          .then((r) => r.json())
          .then((gj) => {
            const style = (f) => {
              const risk = (f.properties?.risk || "").toLowerCase();
              const color = risk === "high" ? "#d73027" : risk === "medium" ? "#fc8d59" : "#1a9850";
              return { color, weight: 2, fillOpacity: 0.15 };
            };

            const layer = L.geoJSON(gj, {
              style,
              onEachFeature: (feature, layer) => {
                const p = feature.properties || {};
                const html = `<b>Risk:</b> ${p.risk || "N/A"}<br/>` +
                  `<b>Area (m²):</b> ${p.area_m2 || ""}<br/>` +
                  (p.id ? `<b>ID:</b> ${p.id}<br/>` : "");
                layer.bindPopup(html);
              },
            });
            layer.addTo(map);

            // Add a simple legend
            const legend = L.control({ position: "bottomright" });
            legend.onAdd = () => {
              const div = L.DomUtil.create("div", "leaflet-control leaflet-bar");
              div.style.background = "white";
              div.style.padding = "8px";
              div.innerHTML = `
                <div style="font-weight:600;margin-bottom:4px">Risk</div>
                <div><span style="background:#d73027;width:12px;height:12px;display:inline-block;margin-right:6px"></span>High</div>
                <div><span style="background:#fc8d59;width:12px;height:12px;display:inline-block;margin-right:6px"></span>Medium</div>
                <div><span style="background:#1a9850;width:12px;height:12px;display:inline-block;margin-right:6px"></span>Low</div>
              `;
              return div;
            };
            legend.addTo(map);
          })
          .catch((e) => setError(String(e)));
      }

      return () => {
        map.remove();
      };
    } catch (e) {
      setError(String(e));
    }
  }, [t0Url, t1Url, geojsonUrl, lat, lon, zoom]);

  return (
    <div style={{ height: "calc(100vh - 80px)", width: "100%" }}>
      {error && (
        <div className="alert alert-danger" role="alert" style={{ margin: 8 }}>
          {error}
        </div>
      )}
      <div id="map-change-viewer" style={{ height: "100%", width: "100%" }} />
    </div>
  );
}


