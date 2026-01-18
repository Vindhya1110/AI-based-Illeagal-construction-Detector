import React, { useState } from "react";
import { Alert, Button, Form } from "react-bootstrap";

export default function AddOn() {
  const [landCoords, setLandCoords] = useState("");
  const [landRadius, setLandRadius] = useState("");
  const [waterCoords, setWaterCoords] = useState("");
  const [waterRadius, setWaterRadius] = useState("");
  const [showConfirm, setShowConfirm] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setShowConfirm(true);
  };

  const handleConfirm = (confirm) => {
    if (confirm) {
      console.log("🌍 Land Region → Coordinates:", landCoords, "| Radius (km):", landRadius);
      console.log("🌊 Water-Body Region → Coordinates:", waterCoords, "| Radius (km):", waterRadius);
      alert("✅ Regions successfully submitted! Check console for details.");
      setSubmitted(true);
    }
    setShowConfirm(false);
  };

  return (
    <div className="d-flex flex-column align-items-center px-4 pb-5">
      <div className="w-100" style={{ maxWidth: "750px" }}>
        <div className="card p-4 mt-4 shadow-sm">
          <h2 className="text-center mb-4 fw-bold text-primary">🌐 Add-On Region Setup</h2>

          <Form onSubmit={handleSubmit} className="d-flex flex-column gap-4">
            {/* Land Region */}
            <div className="border rounded p-3 bg-body-secondary">
              <h4 className="fw-semibold mb-3">🏔 Land Region</h4>
              <div className="d-flex flex-column flex-sm-row gap-3">
                <Form.Control
                  type="text"
                  placeholder="Enter land coordinates (e.g., 17.3850, 78.4867)"
                  value={landCoords}
                  onChange={(e) => setLandCoords(e.target.value)}
                  required
                />
                <Form.Control
                  type="number"
                  step="any" // ✅ allows decimals
                  placeholder="Radius (km)"
                  value={landRadius}
                  onChange={(e) => setLandRadius(e.target.value)}
                  required
                />
              </div>
            </div>

            {/* Water-Body Region */}
            <div className="border rounded p-3 bg-body-secondary">
              <h4 className="fw-semibold mb-3">💧 Water-Body</h4>
              <div className="d-flex flex-column flex-sm-row gap-3">
                <Form.Control
                  type="text"
                  placeholder="Enter water-body coordinates (e.g., 16.5062, 80.6480)"
                  value={waterCoords}
                  onChange={(e) => setWaterCoords(e.target.value)}
                  required
                />
                <Form.Control
                  type="number"
                  step="any" // ✅ allows decimals
                  placeholder="Radius (km)"
                  value={waterRadius}
                  onChange={(e) => setWaterRadius(e.target.value)}
                  required
                />
              </div>
            </div>

            {/* Submit Button */}
            <Button variant="primary" type="submit" className="mt-3 fw-semibold">
              Submit
            </Button>
          </Form>

          {/* Confirmation Alert */}
          {showConfirm && (
            <Alert
              variant="warning"
              className="mt-4 d-flex justify-content-between align-items-center"
            >
              <div>⚠️ Are you sure you want to submit these regions?</div>
              <div className="d-flex gap-2">
                <Button
                  variant="danger"
                  size="sm"
                  onClick={() => handleConfirm(false)}
                >
                  No
                </Button>
                <Button
                  variant="success"
                  size="sm"
                  onClick={() => handleConfirm(true)}
                >
                  Yes
                </Button>
              </div>
            </Alert>
          )}

          {/* Submitted Message */}
          {submitted && (
            <Alert
              variant="success"
              className="mt-3 text-center fw-semibold"
            >
              ✅ Data submitted successfully!
            </Alert>
          )}
        </div>
      </div>
    </div>
  );
}
