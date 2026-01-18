import React from "react";
import { Container, Row, Col, Card, Button } from "react-bootstrap";
import { useNavigate } from "react-router-dom";

function Home() {
  const navigate = useNavigate();

  const handleProceed = () => navigate("/site-check");
  const handleWait = () => window.scrollTo(0, 0);

  return (
    <Container fluid className="py-5">
      <Row className="justify-content-center g-4">
        {/* Welcome Section */}
        <Col xs={12} md={8} lg={7}>
          <Card className="shadow-sm border-0 rounded-4 h-100">
            <Card.Body className="p-4 d-flex flex-column justify-content-between">
              <div>
                <h2 className="fw-bold mb-3">Welcome 👋</h2>
                <p className="text-secondary mb-4">
                  Empowering sustainable urban planning through AI-driven satellite imagery.
                  Detect and report illegal constructions effortlessly with{" "}
                  <b>AI-UrbanVision</b>.
                </p>
              </div>

              <div className="d-flex justify-content-center gap-3 mt-3">
                <Button variant="primary" className="px-4 py-2" onClick={handleProceed}>
                  🚀 Proceed
                </Button>
                <Button variant="outline-secondary" className="px-4 py-2" onClick={handleWait}>
                  ⏳ Wait
                </Button>
              </div>
            </Card.Body>
          </Card>
        </Col>

        {/* Recent Activity Section */}
        <Col xs={12} md={4} lg={4}>
          <Card className="shadow-sm border-0 rounded-4 h-100">
            <Card.Body className="p-4">
              <h5 className="fw-bold mb-3">Recent Activity 🕒</h5>
              <ul className="list-unstyled mb-0">
                <li className="mb-2">✔ Checked: <b>Banjara Hills Zone</b></li>
                <li className="mb-2">⚠ Flagged: <b>2 Sites near Kukatpally</b></li>
                <li>📅 Last Scan: <b>30 Oct 2025</b></li>
              </ul>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}

export default Home;
