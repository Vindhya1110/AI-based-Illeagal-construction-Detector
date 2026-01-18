// src/components/CustomNavbar.jsx
import React, { useState, useRef, useContext } from "react";
import { Navbar, Nav, Container, Image } from "react-bootstrap";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { signOut } from "firebase/auth";
import { auth } from "../firebase";
import ProfileCard from "./ProfileCard";
import { ThemeContext } from "../contexts/ThemeContext";

export default function CustomNavbar() {
  const [expanded, setExpanded] = useState(false);
  const [showPopup, setShowPopup] = useState(false);
  const popupRef = useRef(null);
  const location = useLocation();
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  const { theme, toggleTheme } = useContext(ThemeContext);

  const darkMode = theme === "dark";

  const handleLogout = async () => {
    try {
      await signOut(auth);
      navigate("/login");
    } catch (err) {
      console.error("Logout failed:", err);
    }
  };

  return (
    <Navbar
      bg={darkMode ? "dark" : "light"}
      variant={darkMode ? "dark" : "light"}
      expand="lg"
      expanded={expanded}
      fixed="top"
      className="shadow-sm"
    >
      <Container fluid>
        <Navbar.Brand
          as={Link}
          to="/"
          className="fw-bold text-primary fs-4"
          onClick={() => setExpanded(false)}
        >
          Site4U
        </Navbar.Brand>

        <Navbar.Toggle
          aria-controls="basic-navbar-nav"
          onClick={() => setExpanded((e) => !e)}
        />

        <Navbar.Collapse id="basic-navbar-nav">
          {currentUser && (
            <Nav className="mx-auto text-center">
              <Nav.Link
                as={Link}
                to="/"
                onClick={() => setExpanded(false)}
                active={location.pathname === "/"}
              >
                Home
              </Nav.Link>
              <Nav.Link
                as={Link}
                to="/site-check"
                onClick={() => setExpanded(false)}
                active={location.pathname === "/site-check"}
              >
                Site Check
              </Nav.Link>
              <Nav.Link
                as={Link}
                to="/dashboard"
                onClick={() => setExpanded(false)}
                active={location.pathname === "/dashboard"}
              >
                Dashboard
              </Nav.Link>
              <Nav.Link
                as={Link}
                to="/reports"
                onClick={() => setExpanded(false)}
                active={location.pathname === "/reports"}
              >
                Reports
              </Nav.Link>
              <Nav.Link
                as={Link}
                to="/add-on"
                onClick={() => setExpanded(false)}
                active={location.pathname === "/add-on"}
              >
                Add-On
              </Nav.Link>
            </Nav>
          )}

          <div className="d-flex align-items-center gap-3 me-3 position-relative" ref={popupRef}>
            {/* Theme Toggle (clickable) */}
            <div onClick={toggleTheme} role="button" aria-label="Toggle theme" style={{
              width: 50,
              height: 26,
              borderRadius: 26,
              background: darkMode ? "#333" : "#ccc",
              position: "relative",
              cursor: "pointer",
              transition: "background 0.3s",
            }}>
              <div style={{
                position: "absolute",
                top: 3,
                left: darkMode ? 26 : 3,
                width: 20,
                height: 20,
                borderRadius: "50%",
                background: darkMode ? "#f6d365" : "#222",
                transition: "left 0.3s, background 0.3s",
              }} />
            </div>

            {/* Profile */}
            {currentUser ? (
              <>
                <Image
                  src="https://cdn-icons-png.flaticon.com/512/149/149071.png"
                  roundedCircle
                  width="35"
                  height="35"
                  alt="Profile"
                  onClick={() => setShowPopup((s) => !s)}
                  style={{ cursor: "pointer" }}
                />
                {showPopup && (
                  <div style={{ position: "absolute", top: "60px", right: 0, zIndex: 2000 }}>
                    <ProfileCard user={currentUser} handleLogout={handleLogout} />
                  </div>
                )}
              </>
            ) : (
              <div className="d-flex gap-3">
                <Nav.Link as={Link} to="/login" className="fw-semibold">Login</Nav.Link>
                <Nav.Link as={Link} to="/signup" className="fw-semibold">Sign Up</Nav.Link>
              </div>
            )}
          </div>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
}
