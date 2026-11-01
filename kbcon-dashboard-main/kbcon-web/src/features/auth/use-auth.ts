"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { onAuthStateChanged, signOut, type User } from "firebase/auth";
import { auth } from "@/lib/firebase";

const INACTIVITY_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes inactivity timeout

export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const timeoutIdRef = useRef<NodeJS.Timeout | null>(null);

  const logout = useCallback(async () => {
    if (timeoutIdRef.current) {
      clearTimeout(timeoutIdRef.current);
    }
    await signOut(auth);
  }, []);

  const resetInactivityTimer = useCallback(() => {
    if (timeoutIdRef.current) {
      clearTimeout(timeoutIdRef.current);
    }
    if (auth.currentUser) {
      timeoutIdRef.current = setTimeout(() => {
        void logout();
      }, INACTIVITY_TIMEOUT_MS);
    }
  }, [logout]);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setLoading(false);
      if (currentUser) {
        resetInactivityTimer();
      } else if (timeoutIdRef.current) {
        clearTimeout(timeoutIdRef.current);
      }
    });

    const handleActivity = () => {
      resetInactivityTimer();
    };

    const events = ["mousedown", "keydown", "scroll", "touchstart"];
    events.forEach((evt) =>
      window.addEventListener(evt, handleActivity, { passive: true })
    );

    return () => {
      unsubscribe();
      if (timeoutIdRef.current) {
        clearTimeout(timeoutIdRef.current);
      }
      events.forEach((evt) => window.removeEventListener(evt, handleActivity));
    };
  }, [resetInactivityTimer]);

  return { user, loading, logout };
}
