"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import {
  AlertCircle,
  Camera,
  CheckCircle2,
  ChevronRight,
  Loader2,
  X,
} from "lucide-react";
import { useAuth } from "@/features/auth/use-auth";
import type { PersonProfile } from "./types";

interface EnrollmentProgress {
  session_id: string;
  person_id: string;
  display_name: string;
  mode: string;
  state: string;
  samples_collected: number;
  target_samples: number;
  current_direction?: string | null;
  is_viable: boolean;
  is_complete: boolean;
  last_rejection_reason?: string | null;
}

interface EnrollmentDialogProps {
  person: PersonProfile | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

interface EnrollmentDialogContentProps {
  person: PersonProfile;
  onClose: () => void;
  onSuccess: () => void;
}

function EnrollmentDialogContent({
  person,
  onClose,
  onSuccess,
}: EnrollmentDialogContentProps) {
  const t = useTranslations("People");
  const { user, loading: authLoading } = useAuth();

  const [sessionActive, setSessionActive] = useState(false);
  const [progress, setProgress] = useState<EnrollmentProgress | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [stepping, setStepping] = useState(false);
  const [committing, setCommitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const prevUrlRef = useRef<string | null>(null);

  const getHeaders = useCallback(async (): Promise<HeadersInit> => {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (user) {
      try {
        const token = await user.getIdToken();
        headers["Authorization"] = `Bearer ${token}`;
      } catch {
        // Fallback
      }
    }
    return headers;
  }, [user]);

  // Fetch current camera preview frame
  const fetchPreviewFrame = useCallback(async () => {
    if (!user || authLoading) return;
    try {
      const headers = await getHeaders();
      const res = await fetch(
        `/api/dh360/persons/${person.person_id}/enrollment/sessions/preview`,
        { headers }
      );
      if (res.ok) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        if (prevUrlRef.current) {
          URL.revokeObjectURL(prevUrlRef.current);
        }
        prevUrlRef.current = url;
        setPreviewUrl(url);
      }
    } catch {
      // Stream frame unavailable or initializing
    }
  }, [user, authLoading, person.person_id, getHeaders]);

  // Polling loop for preview frames while session is active
  useEffect(() => {
    if (!sessionActive || !user || authLoading) return;
    const timeout = setTimeout(() => {
      void fetchPreviewFrame();
    }, 0);
    const interval = setInterval(() => {
      void fetchPreviewFrame();
    }, 750);
    return () => {
      clearTimeout(timeout);
      clearInterval(interval);
    };
  }, [sessionActive, user, authLoading, fetchPreviewFrame]);

  // Cleanup object URLs on unmount
  useEffect(() => {
    return () => {
      if (prevUrlRef.current) {
        URL.revokeObjectURL(prevUrlRef.current);
      }
    };
  }, []);

  async function startSession(mode: "replace" | "add_samples" = "replace") {
    setLoading(true);
    setError(null);
    try {
      const headers = await getHeaders();
      const res = await fetch(
        `/api/dh360/persons/${person.person_id}/enrollment/sessions`,
        {
          method: "POST",
          headers,
          body: JSON.stringify({ mode }),
        }
      );
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || errData.error || `HTTP ${res.status}`);
      }
      const data: EnrollmentProgress = await res.json();
      setProgress(data);
      setSessionActive(true);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : t("enrollFailed"));
    } finally {
      setLoading(false);
    }
  }

  async function stepSession() {
    setStepping(true);
    setError(null);
    try {
      const headers = await getHeaders();
      const res = await fetch(
        `/api/dh360/persons/${person.person_id}/enrollment/sessions/step`,
        {
          method: "POST",
          headers,
        }
      );
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || errData.error || `HTTP ${res.status}`);
      }
      const data: EnrollmentProgress = await res.json();
      setProgress(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : t("enrollFailed"));
    } finally {
      setStepping(false);
    }
  }

  async function commitSession() {
    setCommitting(true);
    setError(null);
    try {
      const headers = await getHeaders();
      const res = await fetch(
        `/api/dh360/persons/${person.person_id}/enrollment/sessions/commit`,
        {
          method: "POST",
          headers,
        }
      );
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || errData.error || `HTTP ${res.status}`);
      }
      setSuccess(true);
      setSessionActive(false);
      onSuccess();
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : t("enrollFailed"));
    } finally {
      setCommitting(false);
    }
  }

  async function cancelSession() {
    if (sessionActive) {
      try {
        const headers = await getHeaders();
        await fetch(
          `/api/dh360/persons/${person.person_id}/enrollment/sessions/cancel`,
          {
            method: "POST",
            headers,
          }
        );
      } catch {
        // Best effort cancellation
      }
    }
    setSessionActive(false);
    onClose();
  }

  const stageLabel = () => {
    const dir = progress?.current_direction?.toUpperCase();
    if (dir === "FRONT") return t("stageFront");
    if (dir === "RIGHT") return t("stageRight");
    if (dir === "LEFT") return t("stageLeft");
    return t("stageFront");
  };

  const collected = progress?.samples_collected ?? 0;
  const target = progress?.target_samples ?? 5;
  const progressPercent = Math.min(100, Math.round((collected / target) * 100));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm">
      <div
        className="w-full max-w-xl overflow-hidden rounded-3xl border border-sky-200 bg-white shadow-2xl"
        role="dialog"
        aria-modal="true"
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-sky-100 bg-gradient-to-r from-sky-50 to-white px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-2xl bg-sky-500/10 text-sky-600 ring-1 ring-sky-500/20">
              <Camera className="size-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-sky-950">
                {t("enrollmentTitle")}
              </h2>
              <p className="text-xs text-sky-700/80">{person.display_name}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={cancelSession}
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

          {success && (
            <div className="flex items-center gap-2.5 rounded-2xl bg-emerald-50 p-3.5 text-sm font-semibold text-emerald-800 ring-1 ring-emerald-200">
              <CheckCircle2 className="size-5 shrink-0 text-emerald-600" />
              <span>{t("enrollSuccess")}</span>
            </div>
          )}

          {!sessionActive && !success && (
            <div className="space-y-4 text-center py-6">
              <p className="text-sm text-sky-800/90 leading-relaxed max-w-md mx-auto">
                {t("enrollmentDesc")}
              </p>
              <div className="flex justify-center gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => startSession("replace")}
                  disabled={loading}
                  className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-sky-300/60 hover:from-sky-600 hover:to-sky-700 disabled:opacity-60"
                >
                  {loading && <Loader2 className="size-4 animate-spin" />}
                  {t("startEnrollment")}
                </button>
              </div>
            </div>
          )}

          {sessionActive && (
            <div className="space-y-4">
              {/* Camera Preview */}
              <div className="relative aspect-video w-full overflow-hidden rounded-2xl bg-slate-900 border border-sky-200 flex items-center justify-center">
                {previewUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={previewUrl}
                    alt="Camera preview"
                    className="h-full w-full object-cover"
                  />
                ) : (
                  <div className="text-center p-4">
                    <Loader2 className="mx-auto size-8 animate-spin text-sky-400 mb-2" />
                    <p className="text-xs text-sky-200">{t("noCamera")}</p>
                  </div>
                )}

                {/* Stage guidance overlay */}
                <div className="absolute top-3 left-3 flex items-center gap-2 rounded-xl bg-black/60 px-3 py-1.5 backdrop-blur-md">
                  <span className="size-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-xs font-semibold text-white">
                    {stageLabel()}
                  </span>
                </div>
              </div>

              {/* Progress metrics */}
              <div className="space-y-2">
                <div className="flex justify-between text-xs font-medium text-sky-900">
                  <span>{t("samplesCollected")}</span>
                  <span>
                    {collected} / {target}
                  </span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-sky-100">
                  <div
                    className="h-full bg-gradient-to-r from-sky-500 to-emerald-500 transition-all duration-300"
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
              </div>

              {progress?.is_viable && (
                <p className="text-xs font-semibold text-emerald-700 bg-emerald-50 rounded-xl px-3 py-1.5 ring-1 ring-emerald-200">
                  ✓ {t("readyToCommit")}
                </p>
              )}

              {/* Actions */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
                <button
                  type="button"
                  onClick={cancelSession}
                  className="rounded-xl px-4 py-2 text-sm font-semibold text-sky-800 ring-1 ring-sky-300 hover:bg-sky-50"
                >
                  {t("cancelEnrollment")}
                </button>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={stepSession}
                    disabled={stepping}
                    className="flex items-center gap-1.5 rounded-xl border border-sky-300 bg-white px-4 py-2 text-sm font-semibold text-sky-800 shadow-sm hover:bg-sky-50 disabled:opacity-60"
                  >
                    {stepping && <Loader2 className="size-4 animate-spin" />}
                    <span>{t("stepCapture")}</span>
                    <ChevronRight className="size-4" />
                  </button>

                  <button
                    type="button"
                    onClick={commitSession}
                    disabled={!progress?.is_viable || committing}
                    className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-emerald-600 px-4 py-2 text-sm font-semibold text-white shadow-md shadow-emerald-300/60 hover:from-emerald-600 hover:to-emerald-700 disabled:opacity-50"
                  >
                    {committing && <Loader2 className="size-4 animate-spin" />}
                    {t("commitEnrollment")}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function EnrollmentDialog({
  person,
  open,
  onOpenChange,
  onSuccess,
}: EnrollmentDialogProps) {
  if (!open || !person) return null;

  return (
    <EnrollmentDialogContent
      key={person.person_id}
      person={person}
      onClose={() => onOpenChange(false)}
      onSuccess={onSuccess}
    />
  );
}
