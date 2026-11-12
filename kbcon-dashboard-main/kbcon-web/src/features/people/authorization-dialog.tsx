"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import {
  AlertCircle,
  AlertTriangle,
  CameraOff,
  CheckCircle2,
  Clock,
  Loader2,
  Play,
  Shield,
  ShieldAlert,
  Square,
  UserCheck,
  UserX,
  X,
} from "lucide-react";
import { useAuth } from "@/features/auth/use-auth";

interface AuthorizationSession {
  authorization_id: string;
  status:
    | "initializing"
    | "active"
    | "processing_face"
    | "no_face"
    | "authorized"
    | "not_authorized"
    | "unknown"
    | "uncertain"
    | "camera_error"
    | "processing_error"
    | "lost"
    | "timeout"
    | "cancelled";
  person_id?: string | null;
  display_name?: string | null;
  created_at_iso?: string;
  updated_at_iso?: string;
  ended_at_iso?: string | null;
}

interface AuthorizationDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

interface AuthorizationDialogContentProps {
  onClose: () => void;
}

function AuthorizationDialogContent({ onClose }: AuthorizationDialogContentProps) {
  const t = useTranslations("People");
  const { user, loading: authLoading } = useAuth();

  const [session, setSession] = useState<AuthorizationSession | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const prevUrlRef = useRef<string | null>(null);
  const isPullingRef = useRef(false);
  const activeRef = useRef(true);
  const previewTimerRef = useRef<NodeJS.Timeout | null>(null);
  const userStoppedRef = useRef(false);

  const getHeaders = useCallback(async (): Promise<HeadersInit> => {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (user) {
      try {
        const token = await user.getIdToken();
        headers["Authorization"] = `Bearer ${token}`;
      } catch (err) {
        console.warn("[AuthorizationDialog] Failed to obtain Firebase ID token:", err);
      }
    }
    return headers;
  }, [user]);

  // Continuous active camera frame pull loop
  const fetchPreviewFrame = useCallback(async () => {
    if (!activeRef.current || isPullingRef.current || !user || authLoading) return;
    isPullingRef.current = true;
    try {
      const headers = await getHeaders();
      const res = await fetch(`/api/dh360/authorization/sessions/preview?_t=${Date.now()}`, {
        headers,
        cache: "no-store",
      });
      if (res.ok && activeRef.current) {
        const blob = await res.blob();
        if (blob.size > 0 && activeRef.current) {
          const url = URL.createObjectURL(blob);
          const oldUrl = prevUrlRef.current;
          prevUrlRef.current = url;
          setPreviewUrl(url);
          if (oldUrl) {
            setTimeout(() => URL.revokeObjectURL(oldUrl), 200);
          }
        }
      }
    } catch {
      // preview frame temporarily unavailable
    } finally {
      isPullingRef.current = false;
      if (activeRef.current) {
        previewTimerRef.current = setTimeout(() => {
          void fetchPreviewFrame();
        }, 0);
      }
    }
  }, [user, authLoading, getHeaders]);

  // Start continuous preview loop on mount
  useEffect(() => {
    activeRef.current = true;
    if (user && !authLoading) {
      void fetchPreviewFrame();
    }
    return () => {
      activeRef.current = false;
      if (previewTimerRef.current) {
        clearTimeout(previewTimerRef.current);
      }
      if (prevUrlRef.current) {
        URL.revokeObjectURL(prevUrlRef.current);
      }
    };
  }, [user, authLoading, fetchPreviewFrame]);

  // Start authorization session
  const startAuthorization = useCallback(async () => {
    if (!user) {
      setError("Please sign in with an authorized account.");
      return;
    }
    setLoading(true);
    setError(null);
    userStoppedRef.current = false;
    try {
      const headers = await getHeaders();
      const res = await fetch("/api/dh360/authorization/sessions", {
        method: "POST",
        headers,
      });
      if (res.status === 409) {
        // Active session already running; sync state immediately without throwing
        const curRes = await fetch("/api/dh360/authorization/sessions/current", { headers });
        if (curRes.ok) {
          const data = await curRes.json();
          setSession(data);
        }
        return;
      }
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        if (res.status === 503) {
          throw new Error("DH360 service is currently unreachable (HTTP 503). Ensure backend is running.");
        }
        if (res.status === 403) {
          throw new Error("Forbidden: SuperAdmin or Admin role required.");
        }
        throw new Error(errData.detail || errData.message || errData.error || `HTTP ${res.status}`);
      }
      const data: AuthorizationSession = await res.json();
      setSession(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to start authorization session");
    } finally {
      setLoading(false);
    }
  }, [user, getHeaders]);

  // Poll & maintain active authorization session while dialog is open
  useEffect(() => {
    if (!user || authLoading) return;
    let active = true;

    const syncSession = async () => {
      if (!active) return;
      try {
        const headers = await getHeaders();
        const res = await fetch(`/api/dh360/authorization/sessions/current?_t=${Date.now()}`, {
          headers,
          cache: "no-store",
        });
        if (res.ok && active) {
          const data: AuthorizationSession = await res.json();
          setSession(data);
          // If session ended (e.g. face left frame -> lost or timeout), auto-restart unless operator explicitly stopped
          if (!userStoppedRef.current && (data.ended_at_iso || data.status === "lost" || data.status === "timeout")) {
            void startAuthorization();
          }
        } else if (res.status === 404 && active && !userStoppedRef.current) {
          // No session running yet, auto-start
          void startAuthorization();
        }
      } catch {
        // Network jitter
      }
    };

    void syncSession();
    const interval = setInterval(() => {
      void syncSession();
    }, 1000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [user, authLoading, getHeaders, startAuthorization]);

  const isSessionRunning = Boolean(
    session &&
      !session.ended_at_iso &&
      session.status !== "timeout" &&
      session.status !== "cancelled" &&
      session.status !== "lost"
  );

  async function stopAuthorization() {
    if (!user) return;
    setLoading(true);
    setError(null);
    userStoppedRef.current = true;
    try {
      const headers = await getHeaders();
      const res = await fetch("/api/dh360/authorization/sessions/current", {
        method: "DELETE",
        headers,
      });
      if (res.ok) {
        const data = await res.json().catch(() => null);
        if (data) {
          setSession(data);
        } else {
          setSession(null);
        }
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to stop session");
    } finally {
      setLoading(false);
    }
  }

  const renderStatus = () => {
    if (loading && !session) {
      return (
        <div className="flex items-center gap-3 rounded-2xl bg-sky-50 p-4 text-sm font-semibold text-sky-800 ring-1 ring-sky-200">
          <Loader2 className="size-5 shrink-0 animate-spin text-sky-600" />
          <span>{t("statusInitializing")}</span>
        </div>
      );
    }

    if (!session) {
      return (
        <div className="flex items-center gap-2 rounded-2xl bg-slate-50 p-4 text-sm text-slate-600 ring-1 ring-slate-200">
          <Clock className="size-5 shrink-0 text-slate-400" />
          <span>{t("statusIdle")}</span>
        </div>
      );
    }

    switch (session.status) {
      case "initializing":
        return (
          <div className="flex items-center gap-3 rounded-2xl bg-sky-50 p-4 text-sm font-semibold text-sky-800 ring-1 ring-sky-200">
            <Loader2 className="size-5 shrink-0 animate-spin text-sky-600" />
            <span>{t("statusInitializing")}</span>
          </div>
        );
      case "no_face":
        return (
          <div className="flex items-center gap-3 rounded-2xl bg-amber-50/70 p-4 text-sm font-semibold text-amber-800 ring-1 ring-amber-200">
            <UserX className="size-5 shrink-0 text-amber-600" />
            <span>{t("statusNoFace")}</span>
          </div>
        );
      case "processing_face":
      case "active":
      case "uncertain":
        return (
          <div className="flex items-center gap-3 rounded-2xl bg-purple-50 p-4 text-sm font-semibold text-purple-800 ring-1 ring-purple-200">
            <Loader2 className="size-5 shrink-0 animate-spin text-purple-600" />
            <span>{session.status === "uncertain" ? t("statusUncertain") : t("statusProcessing")}</span>
          </div>
        );
      case "unknown":
        return (
          <div className="flex items-center gap-3 rounded-2xl bg-rose-50 p-4 text-sm font-semibold text-rose-800 ring-1 ring-rose-200">
            <AlertCircle className="size-5 shrink-0 text-rose-600" />
            <span>{t("statusUnknown")}</span>
          </div>
        );
      case "not_authorized":
        return (
          <div className="flex items-center gap-3 rounded-2xl bg-amber-50 p-4 text-sm font-semibold text-amber-800 ring-1 ring-amber-200">
            <ShieldAlert className="size-5 shrink-0 text-amber-600" />
            <span>{t("statusNotAuthorized")}</span>
          </div>
        );
      case "camera_error":
        return (
          <div className="flex items-center gap-3 rounded-2xl bg-red-50 p-4 text-sm font-semibold text-red-800 ring-1 ring-red-200">
            <CameraOff className="size-5 shrink-0 text-red-600" />
            <span>{t("statusCameraError")}</span>
          </div>
        );
      case "processing_error":
        return (
          <div className="flex items-center gap-3 rounded-2xl bg-red-50 p-4 text-sm font-semibold text-red-800 ring-1 ring-red-200">
            <AlertTriangle className="size-5 shrink-0 text-red-600" />
            <span>{t("statusProcessingError")}</span>
          </div>
        );
      case "authorized":
        return (
          <div className="flex items-start gap-3 rounded-2xl bg-emerald-50 p-4 ring-1 ring-emerald-200">
            <CheckCircle2 className="size-6 shrink-0 text-emerald-600 mt-0.5" />
            <div>
              <p className="text-sm font-bold text-emerald-900">
                {t("statusAuthorized")}
              </p>
              <div className="mt-1 flex items-center gap-2">
                <UserCheck className="size-4 text-emerald-700" />
                <span className="text-sm font-semibold text-emerald-800">
                  {session.display_name || session.person_id}
                </span>
                <span className="text-xs text-emerald-600">
                  ({session.person_id})
                </span>
              </div>
            </div>
          </div>
        );
      case "lost":
      case "timeout":
      case "cancelled":
        return (
          <div className="flex items-center gap-3 rounded-2xl bg-slate-50 p-4 text-sm font-semibold text-slate-700 ring-1 ring-slate-200">
            <X className="size-5 shrink-0 text-slate-500" />
            <span>{t("statusEnded")}</span>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm">
      <div
        className="w-full max-w-lg overflow-hidden rounded-3xl border border-sky-200 bg-white shadow-2xl"
        role="dialog"
        aria-modal="true"
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-sky-100 bg-gradient-to-r from-sky-50 to-white px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-2xl bg-sky-500/10 text-sky-600 ring-1 ring-sky-500/20">
              <Shield className="size-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-sky-950">
                {t("authTitle")}
              </h2>
              <p className="text-xs text-sky-700/80">{t("testAuthorization")}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl p-1.5 text-sky-600 hover:bg-sky-100"
            aria-label="Close"
          >
            <X className="size-5" />
          </button>
        </div>

        {/* Content */}
        <div className="space-y-4 p-6">
          {error && (
            <div className="flex items-start gap-2.5 rounded-2xl bg-red-50 p-3.5 text-sm text-red-700 ring-1 ring-red-200">
              <AlertCircle className="size-4 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {/* Live Camera Viewport with Cybernetic Reticle */}
          <div className="relative aspect-video w-full overflow-hidden rounded-2xl bg-slate-950 border border-sky-300/60 shadow-inner flex items-center justify-center group">
            {previewUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={previewUrl}
                alt="Live authorization camera stream"
                className="h-full w-full object-cover"
              />
            ) : (
              <div className="text-center p-4">
                <Loader2 className="mx-auto size-8 animate-spin text-sky-400 mb-2" />
                <p className="text-xs font-medium text-sky-200">
                  Connecting to live camera feed...
                </p>
              </div>
            )}

            {/* Live Indicator Badge */}
            <div className="absolute top-3 left-3 flex items-center gap-1.5 rounded-xl bg-black/65 px-2.5 py-1 backdrop-blur-md pointer-events-auto border border-white/10 shadow-sm">
              <span className="size-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-[11px] font-semibold text-white tracking-wide">
                LIVE CAMERA
              </span>
            </div>
          </div>

          <p className="text-sm text-sky-800/90 leading-relaxed">
            {t("authDesc")}
          </p>

          {renderStatus()}

          {/* Action buttons */}
          <div className="flex justify-end gap-2.5 pt-4 border-t border-sky-100">
            <button
              type="button"
              onClick={onClose}
              className="rounded-xl px-4 py-2 text-sm font-semibold text-sky-800 ring-1 ring-sky-300 hover:bg-sky-50"
            >
              {t("cancel")}
            </button>

            {isSessionRunning ? (
              <button
                type="button"
                onClick={stopAuthorization}
                disabled={loading}
                className="flex items-center gap-2 rounded-xl bg-red-600 px-4 py-2 text-sm font-semibold text-white shadow-md shadow-red-300/60 hover:bg-red-700 disabled:opacity-60"
              >
                {loading ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Square className="size-4" />
                )}
                {t("stopAuth")}
              </button>
            ) : (
              <button
                type="button"
                onClick={startAuthorization}
                disabled={loading}
                className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-4 py-2 text-sm font-semibold text-white shadow-md shadow-sky-300/60 hover:from-sky-600 hover:to-sky-700 disabled:opacity-60"
              >
                {loading ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Play className="size-4" />
                )}
                {t("startAuth")}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export function AuthorizationDialog({
  open,
  onOpenChange,
}: AuthorizationDialogProps) {
  if (!open) return null;

  return <AuthorizationDialogContent onClose={() => onOpenChange(false)} />;
}
