"use client";

import { useCallback, useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import {
  AlertCircle,
  CheckCircle2,
  Clock,
  Loader2,
  Play,
  Shield,
  Square,
  UserCheck,
  X,
} from "lucide-react";
import { useAuth } from "@/features/auth/use-auth";

interface AuthorizationSession {
  authorization_id: string;
  status:
    | "active"
    | "authorized"
    | "unknown"
    | "uncertain"
    | "no_face"
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

  // Poll active authorization session
  const fetchCurrentSession = useCallback(async () => {
    if (!user || authLoading) return;
    try {
      const headers = await getHeaders();
      const res = await fetch("/api/dh360/authorization/sessions/current", {
        headers,
      });
      if (res.ok) {
        const data: AuthorizationSession = await res.json();
        setSession(data);
      } else if (res.status === 404) {
        setSession(null);
      } else if (res.status === 401 || res.status === 403) {
        setError(res.status === 403 ? "Admin role required." : "Authentication required.");
      }
    } catch {
      // Offline or network error
    }
  }, [user, authLoading, getHeaders]);

  useEffect(() => {
    if (!user || authLoading) return;
    const timeout = setTimeout(() => {
      void fetchCurrentSession();
    }, 0);
    const interval = setInterval(() => {
      void fetchCurrentSession();
    }, 1500);
    return () => {
      clearTimeout(timeout);
      clearInterval(interval);
    };
  }, [user, authLoading, fetchCurrentSession]);

  const isSessionRunning = Boolean(
    session &&
      !session.ended_at_iso &&
      session.status !== "timeout" &&
      session.status !== "cancelled" &&
      session.status !== "lost"
  );

  async function startAuthorization() {
    if (!user) {
      setError("Please sign in with an authorized account.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const headers = await getHeaders();
      const res = await fetch("/api/dh360/authorization/sessions", {
        method: "POST",
        headers,
      });
      if (res.status === 409) {
        // Active session already running; sync state immediately without throwing
        await fetchCurrentSession();
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
  }

  async function stopAuthorization() {
    if (!user) return;
    setLoading(true);
    setError(null);
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
      } else {
        await fetchCurrentSession();
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to stop session");
    } finally {
      setLoading(false);
    }
  }

  const renderStatus = () => {
    if (!session) {
      return (
        <div className="flex items-center gap-2 rounded-2xl bg-slate-50 p-4 text-sm text-slate-600 ring-1 ring-slate-200">
          <Clock className="size-5 shrink-0 text-slate-400" />
          <span>{t("statusIdle")}</span>
        </div>
      );
    }

    switch (session.status) {
      case "active":
        return (
          <div className="flex items-center gap-3 rounded-2xl bg-sky-50 p-4 text-sm font-semibold text-sky-800 ring-1 ring-sky-200">
            <Loader2 className="size-5 shrink-0 animate-spin text-sky-600" />
            <span>{t("statusActive")}</span>
          </div>
        );
      case "no_face":
        return (
          <div className="flex items-center gap-3 rounded-2xl bg-amber-50/70 p-4 text-sm font-semibold text-amber-800 ring-1 ring-amber-200">
            <Loader2 className="size-5 shrink-0 animate-spin text-amber-600" />
            <span>{t("statusNoFace")}</span>
          </div>
        );
      case "unknown":
        return (
          <div className="flex items-center gap-3 rounded-2xl bg-rose-50 p-4 text-sm font-semibold text-rose-800 ring-1 ring-rose-200">
            <AlertCircle className="size-5 shrink-0 text-rose-600" />
            <span>{t("statusUnknown")}</span>
          </div>
        );
      case "uncertain":
        return (
          <div className="flex items-center gap-3 rounded-2xl bg-purple-50 p-4 text-sm font-semibold text-purple-800 ring-1 ring-purple-200">
            <Loader2 className="size-5 shrink-0 animate-spin text-purple-600" />
            <span>{t("statusUncertain")}</span>
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
        return (
          <div className="flex items-center gap-3 rounded-2xl bg-amber-50 p-4 text-sm font-semibold text-amber-800 ring-1 ring-amber-200">
            <AlertCircle className="size-5 shrink-0 text-amber-600" />
            <span>{t("statusLost")}</span>
          </div>
        );
      case "timeout":
        return (
          <div className="flex items-center gap-3 rounded-2xl bg-slate-50 p-4 text-sm font-semibold text-slate-700 ring-1 ring-slate-200">
            <Clock className="size-5 shrink-0 text-slate-500" />
            <span>{t("statusTimeout")}</span>
          </div>
        );
      case "cancelled":
        return (
          <div className="flex items-center gap-3 rounded-2xl bg-slate-50 p-4 text-sm font-semibold text-slate-700 ring-1 ring-slate-200">
            <X className="size-5 shrink-0 text-slate-500" />
            <span>{t("statusCancelled")}</span>
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
