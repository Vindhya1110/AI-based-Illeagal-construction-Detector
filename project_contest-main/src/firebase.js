// src/firebase.js
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyADp4tDMQmIuxyjayJMvnWuCPcYpx2-G10",
  authDomain: "site4u-e6116.firebaseapp.com",
  projectId: "site4u-e6116",
  storageBucket: "site4u-e6116.firebasestorage.app",
  messagingSenderId: "929848691031",
  appId: "1:929848691031:web:1c5a2cb8ce76845b1f1612",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
