"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import {
  AlertCircle,
  Camera,
  CheckCircle2,
  Loader2,
  Pause,
  Play,
  RotateCcw,
  Shield,
  Sparkles,
  UserCheck,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/features/auth/use-auth";
import type { PersonProfile } from "./types";

interface EnrollmentProgress {
  session_id: string;
  enrollment_batch_id?: string;
  person_id: string;
  display_name: string;
  mode: string;
  state: string;
  samples_collected: number;
  target_samples: number;
  minimum_samples?: number;
  total_attempts?: number;
  current_direction?: string | null;
  last_accepted_direction?: string | null;
  direction_counts?: Record<string, number>;
  direction_targets?: Record<string, number>;
  face_detected?: boolean;
  is_viable: boolean;
  is_complete: boolean;
  last_rejection_reason?: string | null;
  last_rejection_details?: string | null;
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
  const [isPaused, setIsPaused] = useState(false);
  const [committing, setCommitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [lastFeedback, setLastFeedback] = useState<string | null>(null);

  const prevUrlRef = useRef<string | null>(null);
  const isSteppingRef = useRef(false);
  const stepTimerRef = useRef<NodeJS.Timeout | null>(null);
  const isPausedRef = useRef(false);
  const sessionActiveRef = useRef(false);
  const isCompleteRef = useRef(false);

  // Synchronize ref states
  useEffect(() => {
    isPausedRef.current = isPaused;
  }, [isPaused]);

  useEffect(() => {
    sessionActiveRef.current = sessionActive;
  }, [sessionActive]);

  useEffect(() => {
    isCompleteRef.current = !!progress?.is_complete || (progress?.samples_collected ?? 0) >= 30;
  }, [progress]);

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
    if (!user || authLoading || !sessionActiveRef.current) return;
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
      // Preview stream frame temporarily unavailable
    }
  }, [user, authLoading, person.person_id, getHeaders]);

  // Polling loop for live preview frames
  useEffect(() => {
    if (!sessionActive || !user || authLoading) return;
    const timeout = setTimeout(() => {
      void fetchPreviewFrame();
    }, 50);
    const interval = setInterval(() => {
      void fetchPreviewFrame();
    }, 600);
    return () => {
      clearTimeout(timeout);
      clearInterval(interval);
    };
  }, [sessionActive, user, authLoading, fetchPreviewFrame]);

  // Cleanup preview object URLs on unmount
  useEffect(() => {
    return () => {
      if (prevUrlRef.current) {
        URL.revokeObjectURL(prevUrlRef.current);
      }
      if (stepTimerRef.current) {
        clearTimeout(stepTimerRef.current);
      }
    };
  }, []);

  // Step capture function: evaluates 1 frame on backend
  const executeStep = useCallback(async () => {
    if (
      isSteppingRef.current ||
      isPausedRef.current ||
      !sessionActiveRef.current ||
      isCompleteRef.current
    ) {
      return;
    }

    isSteppingRef.current = true;
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
        if (res.status === 404 || res.status === 409) {
          // Session expired or cancelled on server
          setError(errData.detail || errData.error || t("enrollFailed"));
          setSessionActive(false);
          return;
        }
        // Non-terminal error, schedule retry
        scheduleNextStep(500);
        return;
      }

      const data: EnrollmentProgress = await res.json();
      setProgress(data);

      const isFinished = data.is_complete || data.samples_collected >= (data.target_samples || 30);
      if (isFinished) {
        setLastFeedback(t("allCompletedReady"));
        return;
      }

      // Determine real-time feedback and dynamic delay
      let nextDelayMs = 250;
      if (!data.face_detected) {
        setLastFeedback(t("waitingForFace"));
        nextDelayMs = 500; // Throttle when waiting for face
      } else if (data.last_rejection_reason === "wrong_direction") {
        if (data.current_direction === "right") {
          setLastFeedback(t("turnRightHint"));
        } else if (data.current_direction === "left") {
          setLastFeedback(t("turnLeftHint"));
        } else {
          setLastFeedback(t("lookFrontHint"));
        }
        nextDelayMs = 350;
      } else if (data.last_rejection_reason === "duplicate_sample") {
        setLastFeedback(t("duplicateHint"));
        nextDelayMs = 250;
      } else if (data.last_rejection_reason === "quality_rejected") {
        setLastFeedback(t("lightingWarning"));
        nextDelayMs = 350;
      } else if (data.last_rejection_reason === "multiple_faces") {
        setLastFeedback(t("multipleFacesWarning"));
        nextDelayMs = 500;
      } else {
        setLastFeedback(t("holdSteady"));
        nextDelayMs = 220;
      }

      scheduleNextStep(nextDelayMs);
    } catch {
      scheduleNextStep(600);
    } finally {
      isSteppingRef.current = false;
    }
  }, [getHeaders, person.person_id, t]);

  const scheduleNextStep = useCallback(
    (delayMs: number) => {
      if (stepTimerRef.current) {
        clearTimeout(stepTimerRef.current);
      }
      stepTimerRef.current = setTimeout(() => {
        void executeStep();
      }, delayMs);
    },
    [executeStep]
  );

  // Start or resume enrollment session
  const startSession = useCallback(
    async (mode: "replace" | "add_samples" = "replace") => {
      if (person.consent_status !== "granted") {
        setError("Biometric enrollment requires explicit granted consent.");
        return;
      }
      setLoading(true);
      setError(null);
      setIsPaused(false);
      try {
        const headers = await getHeaders();

        // 1. Check if there is already an active session for this person
        let activeData: EnrollmentProgress | null = null;
        try {
          const checkRes = await fetch(
            `/api/dh360/persons/${person.person_id}/enrollment/sessions`,
            { headers }
          );
          if (checkRes.ok) {
            activeData = await checkRes.json();
          }
        } catch {
          // No active session
        }

        // 2. If no active session for this person, start a new one
        if (!activeData) {
          // If a stale session exists for another entity, force clear it
          try {
            await fetch(`/api/dh360/persons/enrollment/sessions/current`, {
              method: "DELETE",
              headers,
            });
          } catch {
            // Ignore if none active
          }

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
          activeData = await res.json();
        }

        if (activeData) {
          setProgress(activeData);
          setSessionActive(true);
          sessionActiveRef.current = true;
          // Trigger first preview and start auto-capture loop
          void fetchPreviewFrame();
          scheduleNextStep(300);
        }
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : t("enrollFailed"));
      } finally {
        setLoading(false);
      }
    },
    [getHeaders, person.person_id, fetchPreviewFrame, scheduleNextStep, t]
  );

  // Auto-start session immediately when modal opens
  useEffect(() => {
    if (!user || authLoading || sessionActive) return;
    void startSession("replace");
  }, [user, authLoading, sessionActive, startSession]);

  // Handle Pause / Resume toggle
  const togglePause = useCallback(() => {
    setIsPaused((prev) => {
      const next = !prev;
      if (!next) {
        // Resuming: schedule step immediately
        scheduleNextStep(100);
      }
      return next;
    });
  }, [scheduleNextStep]);

  // Restart session from 0
  const restartSession = useCallback(async () => {
    if (stepTimerRef.current) {
      clearTimeout(stepTimerRef.current);
    }
    setError(null);
    setProgress(null);
    setIsPaused(false);
    try {
      const headers = await getHeaders();
      await fetch(`/api/dh360/persons/${person.person_id}/enrollment/sessions/cancel`, {
        method: "POST",
        headers,
      }).catch(() => {});
    } catch {
      // Ignore
    }
    void startSession("replace");
  }, [getHeaders, person.person_id, startSession]);

  // Cancel session
  const cancelSession = useCallback(async () => {
    if (stepTimerRef.current) {
      clearTimeout(stepTimerRef.current);
    }
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
      // Ignore
    }
    setSessionActive(false);
    sessionActiveRef.current = false;
    onClose();
  }, [getHeaders, person.person_id, onClose]);

  // Commit session explicitly
  const commitSession = useCallback(async () => {
    if (stepTimerRef.current) {
      clearTimeout(stepTimerRef.current);
    }
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
      setTimeout(() => {
        onSuccess();
        onClose();
      }, 1600);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : t("enrollFailed"));
    } finally {
      setCommitting(false);
    }
  }, [getHeaders, person.person_id, onSuccess, onClose, t]);

  // Directional metrics
  const target = progress?.target_samples ?? 30;
  const collected = progress?.samples_collected ?? 0;
  const progressPercent = Math.min(100, Math.round((collected / target) * 100));

  const dirCounts = progress?.direction_counts || {};
  const frontCount = dirCounts["front"] ?? 0;
  const rightCount = dirCounts["right"] ?? 0;
  const leftCount = dirCounts["left"] ?? 0;

  const currentDir = progress?.current_direction || "front";
  const isFrontDone = frontCount >= 10;
  const isRightDone = rightCount >= 10;
  const isLeftDone = leftCount >= 10;
  const isReadyToSave = progress?.is_complete || collected >= target || (progress?.is_viable && collected >= 25);

  const getStageHeader = () => {
    if (isReadyToSave) {
      return {
        stageNum: "✓",
        title: t("readyToCommit"),
        desc: t("allCompletedReady"),
        badgeColor: "bg-emerald-500",
      };
    }
    if (!isFrontDone) {
      return {
        stageNum: "1 / 3",
        title: t("stageFront"),
        desc: t("lookFrontHint"),
        badgeColor: "bg-sky-500",
      };
    }
    if (!isRightDone) {
      return {
        stageNum: "2 / 3",
        title: t("stageRight"),
        desc: t("turnRightHint"),
        badgeColor: "bg-blue-600",
      };
    }
    return {
      stageNum: "3 / 3",
      title: t("stageLeft"),
      desc: t("turnLeftHint"),
      badgeColor: "bg-indigo-600",
    };
  };

  const stageInfo = getStageHeader();

  if (person.consent_status !== "granted") {
    return (
      <div
        role="dialog"
        aria-modal="true"
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm"
      >
        <div className="relative w-full max-w-md rounded-2xl bg-white shadow-2xl border border-amber-200 overflow-hidden flex flex-col p-6 space-y-4">
          <div className="flex items-center gap-3 text-amber-600">
            <div className="flex size-10 items-center justify-center rounded-xl bg-amber-100 text-amber-700">
              <Shield className="size-6" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900">
                Consent Required for Enrollment
              </h3>
              <p className="text-xs text-slate-500">
                {person.display_name} • Status: <span className="font-semibold capitalize">{person.consent_status}</span>
              </p>
            </div>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Biometric facial enrollment and identification require explicit granted consent under data protection policies.
            Please open the person&apos;s profile and grant consent before proceeding with biometric enrollment.
          </p>
          <div className="flex justify-end pt-2">
            <Button variant="outline" size="sm" onClick={onClose} className="text-xs">
              Close
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="enrollment-dialog-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm"
    >
      <div className="relative w-full max-w-xl rounded-2xl bg-white shadow-2xl border border-sky-100 overflow-hidden flex flex-col max-h-[92vh]">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-sky-100 px-6 py-4 bg-gradient-to-r from-sky-50/70 via-sky-50/30 to-white">
          <div className="flex items-center gap-2.5">
            <div className="flex size-9 items-center justify-center rounded-xl bg-gradient-to-tr from-sky-500 to-sky-600 text-white shadow-sm shadow-sky-300">
              <Camera className="size-5" />
            </div>
            <div>
              <h2
                id="enrollment-dialog-title"
                className="text-base font-bold text-sky-950"
              >
                {t("enrollmentTitle")}
              </h2>
              <p className="text-xs text-sky-700/80">
                {person.display_name} • ID: {person.person_id}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={cancelSession}
            aria-label={t("cancelEnrollment")}
            className="rounded-lg p-1.5 text-sky-600 hover:bg-sky-100 hover:text-sky-900 transition-colors"
          >
            <X className="size-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 space-y-4 overflow-y-auto">
          {error && (
            <div className="flex items-center gap-2 rounded-xl bg-red-50 p-3 text-xs font-medium text-red-700 ring-1 ring-red-200">
              <AlertCircle className="size-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {success && (
            <div className="flex flex-col items-center justify-center p-8 text-center space-y-3 bg-emerald-50 rounded-2xl border border-emerald-200">
              <div className="size-14 rounded-full bg-emerald-500 text-white flex items-center justify-center shadow-lg shadow-emerald-300 animate-bounce">
                <CheckCircle2 className="size-8" />
              </div>
              <h3 className="text-base font-bold text-emerald-950">
                {t("enrollSuccess")}
              </h3>
              <p className="text-xs text-emerald-700">
                30 facial template angles securely enrolled into edge biometric gallery.
              </p>
            </div>
          )}

          {!success && (
            <div className="space-y-4">
              {/* Active Stage & Guidance Banner */}
              <div className="rounded-2xl border border-sky-200/80 bg-gradient-to-r from-sky-50/80 via-white to-sky-50/50 p-3.5 shadow-sm">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2.5">
                    <span
                      className={`inline-flex items-center justify-center rounded-lg px-2 py-0.5 text-[11px] font-bold text-white shadow-sm ${stageInfo.badgeColor}`}
                    >
                      {stageInfo.stageNum}
                    </span>
                    <h3 className="text-sm font-bold text-sky-950">
                      {stageInfo.title}
                    </h3>
                  </div>
                  {isPaused ? (
                    <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2.5 py-0.5 text-[11px] font-bold text-amber-800 ring-1 ring-amber-300">
                      <Pause className="size-3" />
                      {t("paused")}
                    </span>
                  ) : isReadyToSave ? (
                    <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 px-2.5 py-0.5 text-[11px] font-bold text-emerald-800 ring-1 ring-emerald-300 animate-pulse">
                      <Sparkles className="size-3" />
                      {t("targetReached")}
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-sky-100 px-2.5 py-0.5 text-[11px] font-semibold text-sky-800">
                      <span className="size-2 rounded-full bg-sky-500 animate-ping" />
                      {t("autoCapturing")}
                    </span>
                  )}
                </div>
                <p className="mt-1 text-xs text-sky-800">
                  {stageInfo.desc}
                </p>
              </div>

              {/* Live Camera Preview with Head-Up Display (HUD) */}
              <div className="relative aspect-video w-full overflow-hidden rounded-2xl bg-slate-950 border border-sky-300/60 shadow-inner flex items-center justify-center group">
                {previewUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={previewUrl}
                    alt="Robot camera stream"
                    className="h-full w-full object-cover"
                  />
                ) : (
                  <div className="text-center p-4">
                    <Loader2 className="mx-auto size-8 animate-spin text-sky-400 mb-2" />
                    <p className="text-xs font-medium text-sky-200">
                      {loading ? t("autoCapturing") : t("noCamera")}
                    </p>
                  </div>
                )}

                {/* Top Overlay: Pose badge & pause toggle */}
                <div className="absolute top-3 inset-x-3 flex items-center justify-between pointer-events-none">
                  <div className="flex items-center gap-1.5 rounded-xl bg-black/65 px-3 py-1.5 backdrop-blur-md pointer-events-auto border border-white/10 shadow-sm">
                    <span
                      className={`size-2.5 rounded-full ${
                        isReadyToSave
                          ? "bg-emerald-400"
                          : isPaused
                          ? "bg-amber-400"
                          : progress?.face_detected
                          ? "bg-emerald-400 animate-pulse"
                          : "bg-amber-400 animate-ping"
                      }`}
                    />
                    <span className="text-xs font-semibold text-white tracking-wide">
                      {stageInfo.title}
                    </span>
                  </div>

                  <div className="flex items-center gap-1.5 pointer-events-auto">
                    <button
                      type="button"
                      onClick={togglePause}
                      disabled={isReadyToSave}
                      title={isPaused ? t("resume") : t("pause")}
                      className="rounded-xl bg-black/65 px-2.5 py-1.5 text-xs font-semibold text-white backdrop-blur-md border border-white/10 hover:bg-black/80 transition-colors disabled:opacity-40"
                    >
                      {isPaused ? (
                        <span className="flex items-center gap-1">
                          <Play className="size-3 text-emerald-400" />
                          {t("resume")}
                        </span>
                      ) : (
                        <span className="flex items-center gap-1">
                          <Pause className="size-3 text-amber-300" />
                          {t("pause")}
                        </span>
                      )}
                    </button>
                  </div>
                </div>

                {/* Bottom Overlay: Real-time feedback status badge */}
                <div className="absolute bottom-3 inset-x-3 flex justify-center pointer-events-none">
                  <div className="flex items-center gap-2 rounded-xl bg-black/75 px-3.5 py-1.5 backdrop-blur-md border border-white/15 shadow-md">
                    {progress?.face_detected ? (
                      <UserCheck className="size-3.5 text-emerald-400 shrink-0" />
                    ) : (
                      <span className="size-2 rounded-full bg-amber-400 animate-ping shrink-0" />
                    )}
                    <span className="text-xs font-medium text-slate-100 max-w-sm truncate text-center">
                      {isPaused
                        ? t("paused")
                        : lastFeedback || (progress?.face_detected ? t("holdSteady") : t("waitingForFace"))}
                    </span>
                  </div>
                </div>
              </div>

              {/* 3-Direction Progress Matrix */}
              <div className="grid grid-cols-3 gap-2.5">
                {/* Front (10) */}
                <div
                  className={`rounded-xl border p-2.5 transition-all ${
                    isFrontDone
                      ? "border-emerald-300 bg-emerald-50/70"
                      : currentDir === "front" && !isReadyToSave
                      ? "border-sky-400 bg-sky-50 shadow-sm ring-1 ring-sky-300"
                      : "border-slate-200 bg-slate-50/60"
                  }`}
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-700">
                      {t("frontStage")}
                    </span>
                    {isFrontDone ? (
                      <CheckCircle2 className="size-3.5 text-emerald-600" />
                    ) : (
                      <span className="text-[10px] text-slate-500 font-mono">
                        {frontCount}/10
                      </span>
                    )}
                  </div>
                  <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
                    <div
                      className={`h-full transition-all duration-300 ${
                        isFrontDone ? "bg-emerald-500" : "bg-sky-500"
                      }`}
                      style={{
                        width: `${Math.min(100, (frontCount / 10) * 100)}%`,
                      }}
                    />
                  </div>
                </div>

                {/* Right (10) */}
                <div
                  className={`rounded-xl border p-2.5 transition-all ${
                    isRightDone
                      ? "border-emerald-300 bg-emerald-50/70"
                      : currentDir === "right" && !isReadyToSave
                      ? "border-blue-400 bg-blue-50 shadow-sm ring-1 ring-blue-300"
                      : "border-slate-200 bg-slate-50/60"
                  }`}
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-700">
                      {t("rightStage")}
                    </span>
                    {isRightDone ? (
                      <CheckCircle2 className="size-3.5 text-emerald-600" />
                    ) : (
                      <span className="text-[10px] text-slate-500 font-mono">
                        {rightCount}/10
                      </span>
                    )}
                  </div>
                  <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
                    <div
                      className={`h-full transition-all duration-300 ${
                        isRightDone ? "bg-emerald-500" : "bg-blue-600"
                      }`}
                      style={{
                        width: `${Math.min(100, (rightCount / 10) * 100)}%`,
                      }}
                    />
                  </div>
                </div>

                {/* Left (10) */}
                <div
                  className={`rounded-xl border p-2.5 transition-all ${
                    isLeftDone
                      ? "border-emerald-300 bg-emerald-50/70"
                      : currentDir === "left" && !isReadyToSave
                      ? "border-indigo-400 bg-indigo-50 shadow-sm ring-1 ring-indigo-300"
                      : "border-slate-200 bg-slate-50/60"
                  }`}
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-700">
                      {t("leftStage")}
                    </span>
                    {isLeftDone ? (
                      <CheckCircle2 className="size-3.5 text-emerald-600" />
                    ) : (
                      <span className="text-[10px] text-slate-500 font-mono">
                        {leftCount}/10
                      </span>
                    )}
                  </div>
                  <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
                    <div
                      className={`h-full transition-all duration-300 ${
                        isLeftDone ? "bg-emerald-500" : "bg-indigo-600"
                      }`}
                      style={{
                        width: `${Math.min(100, (leftCount / 10) * 100)}%`,
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Total Progress Bar */}
              <div className="space-y-1.5 pt-1">
                <div className="flex justify-between text-xs font-semibold text-sky-950">
                  <span>{t("samplesCollected")}</span>
                  <span className="font-mono">
                    {collected} / {target} ({progressPercent}%)
                  </span>
                </div>
                <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100 border border-slate-200">
                  <div
                    className="h-full bg-gradient-to-r from-sky-500 via-blue-500 to-emerald-500 transition-all duration-300"
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
              </div>

              {/* Bottom Actions Bar */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-100">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={cancelSession}
                    className="rounded-xl px-3.5 py-2 text-xs font-semibold text-slate-700 ring-1 ring-slate-300 hover:bg-slate-100 transition-colors"
                  >
                    {t("cancelEnrollment")}
                  </button>
                  <button
                    type="button"
                    onClick={restartSession}
                    title={t("restart")}
                    className="flex items-center gap-1 rounded-xl px-3 py-2 text-xs font-semibold text-slate-600 ring-1 ring-slate-300 hover:bg-slate-100 transition-colors"
                  >
                    <RotateCcw className="size-3.5" />
                    <span>{t("restart")}</span>
                  </button>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={commitSession}
                    disabled={!isReadyToSave || committing}
                    className={`flex items-center gap-2 rounded-xl px-5 py-2.5 text-sm font-bold text-white shadow-md transition-all ${
                      isReadyToSave
                        ? "bg-gradient-to-r from-emerald-500 to-emerald-600 shadow-emerald-300/80 hover:from-emerald-600 hover:to-emerald-700 ring-2 ring-emerald-400 ring-offset-2 animate-pulse"
                        : "bg-slate-400 opacity-50 cursor-not-allowed"
                    }`}
                  >
                    {committing ? (
                      <>
                        <Loader2 className="size-4 animate-spin" />
                        <span>{t("saving")}</span>
                      </>
                    ) : (
                      <>
                        <UserCheck className="size-4" />
                        <span>{t("commitEnrollment")}</span>
                      </>
                    )}
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
