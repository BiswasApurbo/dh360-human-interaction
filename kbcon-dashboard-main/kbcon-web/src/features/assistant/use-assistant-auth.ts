"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useAuth } from "@/features/auth/use-auth";
import type { AuthorizationSession } from "./types";

interface UseAssistantAuthOptions {
  onAuthorizationLost?: (lostPersonId: string | null) => void;
  onAuthorizationGranted?: (session: AuthorizationSession) => void;
}

export function useAssistantAuth({
  onAuthorizationLost,
  onAuthorizationGranted,
}: UseAssistantAuthOptions = {}) {
  const { user, loading: authLoading } = useAuth();
  const [session, setSession] = useState<AuthorizationSession | null>(null);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const prevSessionRef = useRef<AuthorizationSession | null>(null);
  const callbacksRef = useRef({ onAuthorizationLost, onAuthorizationGranted });
  callbacksRef.current = { onAuthorizationLost, onAuthorizationGranted };

  const getHeaders = useCallback(async (): Promise<HeadersInit> => {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (user) {
      try {
        const token = await user.getIdToken();
        headers["Authorization"] = `Bearer ${token}`;
      } catch (err) {
        console.warn("[AssistantAuth] Failed to acquire Firebase ID token:", err);
      }
    }
    return headers;
  }, [user]);

  // Fetch the currently active session from the BFF
  const fetchCurrentSession = useCallback(async () => {
    if (!user || authLoading) return null;
    try {
      const headers = await getHeaders();
      const res = await fetch("/api/dh360/authorization/sessions/current", {
        headers,
      });
      if (res.ok) {
        const data: AuthorizationSession = await res.json();
        setSession(data);
        return data;
      } else if (res.status === 404) {
        setSession(null);
        return null;
      }
    } catch {
      // Backend temporarily unreachable; stay in general mode
    }
    return null;
  }, [user, authLoading, getHeaders]);

  // Start or synchronize authorization scan on mount or rescan
  const startOrSyncScan = useCallback(async () => {
    if (!user || authLoading) return;
    setScanning(true);
    setError(null);
    try {
      const headers = await getHeaders();
      const res = await fetch("/api/dh360/authorization/sessions", {
        method: "POST",
        headers,
      });

      if (res.status === 201) {
        const data: AuthorizationSession = await res.json();
        setSession(data);
      } else if (res.status === 409) {
        // Active session already running: seamlessly synchronize current session
        await fetchCurrentSession();
      } else {
        // Fallback: poll current or continue gracefully in General mode
        await fetchCurrentSession();
      }
    } catch (err: unknown) {
      console.warn("[AssistantAuth] Scan start failed, operating in general mode:", err);
    } finally {
      setScanning(false);
    }
  }, [user, authLoading, getHeaders, fetchCurrentSession]);

  // Automatic scan initiation on entering the page
  useEffect(() => {
    if (!user || authLoading) return;

    let mounted = true;
    void startOrSyncScan();

    const interval = setInterval(() => {
      if (mounted) {
        void fetchCurrentSession();
      }
    }, 1500);

    return () => {
      mounted = false;
      clearInterval(interval);
      // NOTE: We intentionally DO NOT cancel the session on unmount
      // to preserve continuous background perception across navigation.
    };
  }, [user, authLoading, startOrSyncScan, fetchCurrentSession]);

  // Detect and enforce authorization transitions (privacy boundary)
  useEffect(() => {
    const prev = prevSessionRef.current;
    const current = session;

    const wasAuthorized =
      prev?.status === "authorized" && Boolean(prev?.person_id);
    const isNowAuthorized =
      current?.status === "authorized" && Boolean(current?.person_id);

    if (wasAuthorized && !isNowAuthorized) {
      // Authorization was lost, expired, or face left camera view:
      // Trigger privacy boundary callback immediately
      callbacksRef.current.onAuthorizationLost?.(prev?.person_id || null);
    } else if (
      wasAuthorized &&
      isNowAuthorized &&
      prev?.person_id !== current?.person_id
    ) {
      // Identity changed to a different person: purge previous person's context
      callbacksRef.current.onAuthorizationLost?.(prev?.person_id || null);
      if (current) {
        callbacksRef.current.onAuthorizationGranted?.(current);
      }
    } else if (!wasAuthorized && isNowAuthorized && current) {
      callbacksRef.current.onAuthorizationGranted?.(current);
    }

    prevSessionRef.current = session;
  }, [session]);

  const isAuthorized =
    session?.status === "authorized" && Boolean(session?.person_id);
  const currentPersonId = isAuthorized ? session?.person_id ?? null : null;
  const currentDisplayName = isAuthorized
    ? session?.display_name || session?.person_id || null
    : null;

  return {
    session,
    isAuthorized,
    currentPersonId,
    currentDisplayName,
    scanning,
    error,
    rescan: startOrSyncScan,
    fetchCurrentSession,
    getHeaders,
  };
}
