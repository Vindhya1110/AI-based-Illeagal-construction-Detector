import React, { useState } from "react";
import { FileText } from "lucide-react";
import { jsPDF } from "jspdf";
import "jspdf-autotable";

export default function Reports() {
  const [reportData] = useState({
    siteName: "Construction Site #102",
    coordinates: "17.3850° N, 78.4867° E",
    radius: "1.5 km",
    detectedOn: "2025-11-01",
    status: "Illegal construction suspected",
    remarks:
      "Further inspection required. Nearby structures show deviation from zoning laws.",
  });

  const handleDownload = () => {
    const doc = new jsPDF();
    doc.setFont("helvetica", "bold");
    doc.setFontSize(18);
    doc.text("Site Report Summary", 70, 20);
    doc.setFontSize(12);

    const tableData = Object.entries(reportData).map(([key, value]) => [key, value]);
    doc.autoTable({
      head: [["Field", "Details"]],
      body: tableData,
      theme: "grid",
      styles: { fontSize: 12 },
      margin: { top: 30 },
    });

    doc.save("site_report.pdf");
  };

  return (
    <div className="container-fluid py-5">
      <div className="card mx-auto shadow-lg p-4" style={{ maxWidth: 800 }}>
        <div className="text-center mb-4">
          <FileText size={40} className="text-warning mb-2" />
          <h2 className="fw-bold">Site Report Summary</h2>
        </div>

        <table className="table table-bordered align-middle">
          <tbody>
            {Object.entries(reportData).map(([key, value]) => (
              <tr key={key}>
                <th style={{ width: "30%", textTransform: "capitalize" }}>{key}</th>
                <td>{value}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="text-center mt-4">
          <button onClick={handleDownload} className="btn btn-primary px-4 py-2">
            Download PDF Report
          </button>
        </div>
      </div>
    </div>
  );
}
