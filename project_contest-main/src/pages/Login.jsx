import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Form, Button } from "react-bootstrap";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "../firebase";

function ToggleSwitch({ toggled, onClick, color }) {
  return (
    <div
      style={{
        position: "fixed",
        top: 32,
        right: 48,
        zIndex: 10,
        cursor: "pointer",
      }}
      onClick={onClick}
    >
      <div style={{
        width: 56, height: 32, borderRadius: 20, background: toggled ? color : "#e3e4ea",
        display: "flex", alignItems: "center", transition: "background 0.3s"
      }}>
        <div style={{
          width: 28, height: 28, borderRadius: "50%", background: "#fff",
          transform: toggled ? "translateX(24px)" : "translateX(4px)",
          transition: "transform 0.3s",
        }}/>
      </div>
    </div>
  );
}

function Login() {
  const [darkMode, setDarkMode] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const bootstrapBlue = "#0d6efd";

  const theme = {
    background: darkMode ? "#121728" : "#f5f7fa",
    cardBg: darkMode ? "#171e37" : "#fff",
    text: darkMode ? "#fff" : "#181b2c",
    inputBg: darkMode ? "#20263d" : "#f2f3f8",
    muted: darkMode ? "#adb0be" : "#6f7895",
    highlight: bootstrapBlue,
    buttonBg: bootstrapBlue,
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      await signInWithEmailAndPassword(auth, email, password);
      navigate("/home");
    } catch (error) {
      alert(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: "100vh",
      width: "100vw",
      background: theme.background,
      color: theme.text,
      display: "flex",
      alignItems: "center",
      justifyContent: "center"
    }}>
      <ToggleSwitch toggled={darkMode} onClick={() => setDarkMode((d) => !d)} color={bootstrapBlue} />

      <div className="card p-5" style={{ width: "100%", maxWidth: 430, background: theme.cardBg, color: theme.text }}>
        <h2 className="fw-bold text-center mb-4" style={{ fontSize: "2rem" }}>Sign in to your account</h2>
        <Form onSubmit={handleLogin}>
          <Form.Group className="mb-3">
            <Form.Label style={{ color: theme.text }}>Email address</Form.Label>
            <Form.Control
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              style={{ background: theme.inputBg, border: "none", color: theme.text }}
            />
          </Form.Group>
          <Form.Group className="mb-3">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <Form.Label style={{ color: theme.text, marginBottom: 0 }}>Password</Form.Label>
              <a href="#" style={{ color: theme.highlight }}>Forgot password?</a>
            </div>
            <Form.Control
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              style={{ background: theme.inputBg, border: "none", color: theme.text }}
            />
          </Form.Group>
          <Button type="submit" variant="primary" className="w-100" disabled={loading}>
            {loading ? "Signing in..." : "Sign in"}
          </Button>
        </Form>

        <div className="text-center mt-3">
          <span style={{ color: theme.muted }}>
            New user?{" "}
            <span style={{ color: theme.highlight, cursor: "pointer" }} onClick={() => navigate("/signup")}>Signup</span>
          </span>
        </div>
      </div>
    </div>
  );
}

export default Login;
