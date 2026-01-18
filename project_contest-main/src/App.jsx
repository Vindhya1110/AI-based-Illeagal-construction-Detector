// src/App.jsx
import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import CustomNavbar from "./components/CustomNavbar";
import Chatbot from "./components/Chatbot";
import Home from "./pages/Home";
import Dashboard from "./pages/Dashboard";
import SiteCheck from "./pages/SiteCheck";
import Reports from "./pages/Reports";
import Login from "./pages/Login";
import SignUp from "./pages/SignUp";
import History from "./pages/History";
import AddOn from "./pages/AddOn";
import ChangeViewer from "./pages/ChangeViewer";
import PrivateRoute from "./components/PrivateRoute";
import { useAuth } from "./contexts/AuthContext";

export default function App() {
  const { currentUser, loading } = useAuth();

  if (loading) return <div className="text-center mt-5">Loading...</div>;

  const isAuthPage = window.location.pathname === "/login" || window.location.pathname === "/signup";

  return (
    <>
      {/* Only show the Navbar if the user is logged in and not on login/signup page */}
      {!isAuthPage && currentUser && <CustomNavbar />}
      <main style={{ marginTop: currentUser && !isAuthPage ? "70px" : "0" }}>
        <Routes>
          <Route
            path="/"
            element={
              currentUser ? <Navigate to="/home" replace /> : <Navigate to="/login" replace />
            }
          />
          <Route
            path="/history"
            element={
              <PrivateRoute>
                <History />
              </PrivateRoute>
            }
          />
          <Route
            path="/add-on"
            element={
              <PrivateRoute>
                <AddOn />
              </PrivateRoute>
            }
          />
          <Route
            path="/login"
            element={currentUser ? <Navigate to="/home" replace /> : <Login />}
          />
          <Route
            path="/signup"
            element={currentUser ? <Navigate to="/home" replace /> : <SignUp />}
          />
          <Route
            path="/home"
            element={
              <PrivateRoute>
                <Home />
              </PrivateRoute>
            }
          />
          <Route
            path="/dashboard"
            element={
              <PrivateRoute>
                <Dashboard />
              </PrivateRoute>
            }
          />
          <Route
            path="/site-check"
            element={
              <PrivateRoute>
                <SiteCheck />
              </PrivateRoute>
            }
          />
          <Route
            path="/reports"
            element={
              <PrivateRoute>
                <Reports />
              </PrivateRoute>
            }
          />
          <Route
            path="/viewer"
            element={
              <PrivateRoute>
                <ChangeViewer />
              </PrivateRoute>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      {!isAuthPage && currentUser && <Chatbot />}
    </>
  );
}
