"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import {
  applyActionCode,
  checkActionCode,
  confirmPasswordReset,
  verifyPasswordResetCode,
} from "firebase/auth";
import {
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  Eye,
  EyeOff,
  KeyRound,
  Loader2,
  MailCheck,
  ShieldAlert,
  ShieldCheck,
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Link, useRouter } from "@/i18n/navigation";
import { auth } from "@/lib/firebase";

function evaluatePasswordStrength(password: string): {
  score: number;
  label: string;
  color: string;
  checks: { length: boolean; lower: boolean; upper: boolean; numberOrSpecial: boolean };
} {
  const checks = {
    length: password.length >= 8,
    lower: /[a-z]/.test(password),
    upper: /[A-Z]/.test(password),
    numberOrSpecial: /[0-9!@#$%^&*()_+\-=[\]{};':"\\|,.<>/?]/.test(password),
  };

  const validCount = Object.values(checks).filter(Boolean).length;

  if (validCount <= 1) {
    return { score: 25, label: "Weak", color: "bg-red-500", checks };
  }
  if (validCount <= 2) {
    return { score: 50, label: "Fair", color: "bg-amber-500", checks };
  }
  if (validCount <= 3) {
    return { score: 75, label: "Good", color: "bg-sky-500", checks };
  }
  return { score: 100, label: "Strong", color: "bg-emerald-500", checks };
}

type ActionStatus =
  | "INITIALIZING"
  | "VERIFYING"
  | "RESET_READY"
  | "SUBMITTING"
  | "SUCCESS"
  | "ERROR_EXPIRED"
  | "ERROR_INVALID"
  | "ERROR_NO_CODE"
  | "ERROR_GENERIC"
  | "VERIFY_EMAIL_SUCCESS";

export function AuthActionContent() {
  const t = useTranslations("AuthAction");
  const tl = useTranslations("Login");
  const router = useRouter();
  const searchParams = useSearchParams();

  const [mode, setMode] = useState<string | null>(null);
  const [oobCode, setOobCode] = useState<string | null>(null);
  const [targetEmail, setTargetEmail] = useState<string | null>(null);

  const [status, setStatus] = useState<ActionStatus>("INITIALIZING");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showNew, setShowNew] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);

  const strength = evaluatePasswordStrength(newPassword);

  // 1. Extract params on mount and immediately sanitize the browser URL for security
  useEffect(() => {
    const rawMode = searchParams.get("mode") || "";
    const rawCode = searchParams.get("oobCode") || "";

    setMode(rawMode);

    if (!rawCode) {
      setStatus("ERROR_NO_CODE");
      return;
    }

    setOobCode(rawCode);

    // SECURITY: Strip sensitive one-time oobCode credential from the address bar & history immediately
    try {
      if (typeof window !== "undefined" && window.history?.replaceState) {
        window.history.replaceState(null, "", window.location.pathname);
      }
    } catch {
      // Safe fallback
    }

    // Process action based on mode
    if (rawMode === "resetPassword") {
      setStatus("VERIFYING");
      verifyPasswordResetCode(auth, rawCode)
        .then((email) => {
          setTargetEmail(email);
          setStatus("RESET_READY");
        })
        .catch((err) => {
          const code = (err as { code?: string })?.code;
          if (code === "auth/expired-action-code") {
            setStatus("ERROR_EXPIRED");
          } else if (code === "auth/invalid-action-code") {
            setStatus("ERROR_INVALID");
          } else {
            setStatus("ERROR_GENERIC");
            setErrorMessage(err instanceof Error ? err.message : t("invalidMessage"));
          }
        });
    } else if (rawMode === "verifyEmail") {
      setStatus("VERIFYING");
      applyActionCode(auth, rawCode)
        .then(() => {
          setStatus("VERIFY_EMAIL_SUCCESS");
        })
        .catch((err) => {
          const code = (err as { code?: string })?.code;
          if (code === "auth/expired-action-code") {
            setStatus("ERROR_EXPIRED");
          } else {
            setStatus("ERROR_INVALID");
          }
        });
    } else if (rawMode === "recoverEmail") {
      setStatus("VERIFYING");
      checkActionCode(auth, rawCode)
        .then(() => {
          return applyActionCode(auth, rawCode);
        })
        .then(() => {
          setStatus("VERIFY_EMAIL_SUCCESS");
        })
        .catch(() => {
          setStatus("ERROR_INVALID");
        });
    } else {
      setStatus("ERROR_INVALID");
    }
  }, [searchParams, t]);

  // 2. Submit password reset
  async function handleResetSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!oobCode) return;

    setErrorMessage(null);

    if (newPassword.length < 8 || strength.score < 50) {
      setErrorMessage(t("weakPassword"));
      return;
    }
    if (newPassword !== confirmPassword) {
      setErrorMessage(t("passwordMismatch"));
      return;
    }

    setStatus("SUBMITTING");
    try {
      await confirmPasswordReset(auth, oobCode, newPassword);
      setStatus("SUCCESS");

      // Auto redirect to login after short delay
      setTimeout(() => {
        router.replace("/login");
      }, 2000);
    } catch (err: unknown) {
      const code = (err as { code?: string })?.code;
      if (code === "auth/expired-action-code") {
        setStatus("ERROR_EXPIRED");
      } else if (code === "auth/invalid-action-code") {
        setStatus("ERROR_INVALID");
      } else if (code === "auth/weak-password") {
        setStatus("RESET_READY");
        setErrorMessage(t("weakPassword"));
      } else {
        setStatus("RESET_READY");
        setErrorMessage(err instanceof Error ? err.message : t("invalidMessage"));
      }
    }
  }

  // --- Rendering States ---

  // State A: Verifying / Initializing
  if (status === "INITIALIZING" || status === "VERIFYING") {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-center">
        <Loader2 className="size-10 animate-spin text-sky-600 mb-4" />
        <p className="text-sm font-medium text-sky-950">{t("verifying")}</p>
        <p className="mt-1 text-xs text-slate-500">
          Communicating with KBCON security service...
        </p>
      </div>
    );
  }

  // State B: Success
  if (status === "SUCCESS") {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="space-y-4 py-4 text-center"
      >
        <div className="mx-auto flex size-14 items-center justify-center rounded-2xl bg-emerald-100 text-emerald-600 shadow-sm ring-1 ring-emerald-300">
          <CheckCircle2 className="size-8" />
        </div>
        <h2 className="text-lg font-bold text-sky-950">{t("successTitle")}</h2>
        <p className="text-xs text-slate-600 leading-relaxed">
          {t("successMessage")}
        </p>
        <div className="pt-2">
          <button
            type="button"
            onClick={() => router.replace("/login")}
            className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-4 py-2.5 text-xs font-semibold text-white shadow-md shadow-sky-200 hover:from-sky-600 hover:to-sky-700 transition"
          >
            {t("goToLogin")}
          </button>
        </div>
      </motion.div>
    );
  }

  // State C: Verify Email Success
  if (status === "VERIFY_EMAIL_SUCCESS") {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="space-y-4 py-4 text-center"
      >
        <div className="mx-auto flex size-14 items-center justify-center rounded-2xl bg-emerald-100 text-emerald-600 shadow-sm ring-1 ring-emerald-300">
          <MailCheck className="size-8" />
        </div>
        <h2 className="text-lg font-bold text-sky-950">Email Verified</h2>
        <p className="text-xs text-slate-600 leading-relaxed">
          Your email address has been successfully verified for KBCON.
        </p>
        <div className="pt-2">
          <button
            type="button"
            onClick={() => router.replace("/login")}
            className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-sky-600 px-4 py-2.5 text-xs font-semibold text-white shadow-md hover:bg-sky-700 transition"
          >
            {t("goToLogin")}
          </button>
        </div>
      </motion.div>
    );
  }

  // State D: Expired Link
  if (status === "ERROR_EXPIRED") {
    return (
      <div className="space-y-4 py-4 text-center">
        <div className="mx-auto flex size-14 items-center justify-center rounded-2xl bg-amber-100 text-amber-600 shadow-sm ring-1 ring-amber-300">
          <ShieldAlert className="size-8" />
        </div>
        <h2 className="text-lg font-bold text-sky-950">{t("expiredTitle")}</h2>
        <p className="text-xs text-slate-600 leading-relaxed">
          {t("expiredMessage")}
        </p>
        <div className="pt-2 space-y-2">
          <Link
            href="/forgot-password"
            className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-sky-600 px-4 py-2.5 text-xs font-semibold text-white shadow-md hover:bg-sky-700 transition"
          >
            {t("requestNewLink")}
          </Link>
          <Link
            href="/login"
            className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-600 hover:text-sky-800 transition pt-2"
          >
            <ArrowLeft className="size-3.5" />
            <span>{tl("signIn")}</span>
          </Link>
        </div>
      </div>
    );
  }

  // State E: Invalid Code or Missing Code
  if (status === "ERROR_INVALID" || status === "ERROR_NO_CODE" || status === "ERROR_GENERIC") {
    return (
      <div className="space-y-4 py-4 text-center">
        <div className="mx-auto flex size-14 items-center justify-center rounded-2xl bg-red-100 text-red-600 shadow-sm ring-1 ring-red-300">
          <AlertCircle className="size-8" />
        </div>
        <h2 className="text-lg font-bold text-sky-950">
          {status === "ERROR_NO_CODE" ? t("noCodeTitle") : t("invalidTitle")}
        </h2>
        <p className="text-xs text-slate-600 leading-relaxed">
          {errorMessage || (status === "ERROR_NO_CODE" ? t("noCodeMessage") : t("invalidMessage"))}
        </p>
        <div className="pt-2 space-y-2">
          <Link
            href="/forgot-password"
            className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-sky-600 px-4 py-2.5 text-xs font-semibold text-white shadow-md hover:bg-sky-700 transition"
          >
            {t("requestNewLink")}
          </Link>
          <Link
            href="/login"
            className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-600 hover:text-sky-800 transition pt-2"
          >
            <ArrowLeft className="size-3.5" />
            <span>{tl("signIn")}</span>
          </Link>
        </div>
      </div>
    );
  }

  // State F: Reset Ready (Form)
  return (
    <form onSubmit={handleResetSubmit} className="space-y-4">
      {targetEmail && (
        <div className="rounded-xl bg-sky-50 p-3 text-xs border border-sky-100">
          <span className="text-slate-500">{t("accountLabel")}: </span>
          <span className="font-semibold text-sky-950">{targetEmail}</span>
        </div>
      )}

      {errorMessage && (
        <div className="flex items-center gap-2 rounded-xl bg-red-50 p-3 text-xs font-medium text-red-700 ring-1 ring-red-200">
          <AlertCircle className="size-4 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* New Password */}
      <div className="space-y-1.5">
        <Label htmlFor="new-pass" className="text-xs font-semibold text-sky-900">
          {t("newPassword")}
        </Label>
        <div className="relative">
          <Input
            id="new-pass"
            type={showNew ? "text" : "password"}
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            disabled={status === "SUBMITTING"}
            className="pr-10 text-sm"
            required
            autoComplete="new-password"
          />
          <button
            type="button"
            onClick={() => setShowNew(!showNew)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
            tabIndex={-1}
          >
            {showNew ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
          </button>
        </div>

        {/* Strength Meter */}
        {newPassword && (
          <div className="mt-2 space-y-1.5 rounded-xl bg-slate-50 p-2.5 border border-slate-200">
            <div className="flex items-center justify-between text-[11px] font-semibold text-slate-700">
              <span>Password Strength</span>
              <span>{strength.label}</span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
              <div
                className={`h-full transition-all duration-300 ${strength.color}`}
                style={{ width: `${strength.score}%` }}
              />
            </div>
            <div className="grid grid-cols-2 gap-1 pt-1 text-[10px] text-slate-600">
              <span className={strength.checks.length ? "text-emerald-600 font-medium" : ""}>
                {strength.checks.length ? "✓" : "○"} 8+ characters
              </span>
              <span className={strength.checks.lower ? "text-emerald-600 font-medium" : ""}>
                {strength.checks.lower ? "✓" : "○"} Lowercase letter
              </span>
              <span className={strength.checks.upper ? "text-emerald-600 font-medium" : ""}>
                {strength.checks.upper ? "✓" : "○"} Uppercase letter
              </span>
              <span className={strength.checks.numberOrSpecial ? "text-emerald-600 font-medium" : ""}>
                {strength.checks.numberOrSpecial ? "✓" : "○"} Number or symbol
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Confirm Password */}
      <div className="space-y-1.5">
        <Label htmlFor="confirm-pass" className="text-xs font-semibold text-sky-900">
          {t("confirmPassword")}
        </Label>
        <div className="relative">
          <Input
            id="confirm-pass"
            type={showConfirm ? "text" : "password"}
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            disabled={status === "SUBMITTING"}
            className="pr-10 text-sm"
            required
            autoComplete="new-password"
          />
          <button
            type="button"
            onClick={() => setShowConfirm(!showConfirm)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
            tabIndex={-1}
          >
            {showConfirm ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
          </button>
        </div>
      </div>

      <p className="text-[11px] text-slate-500 leading-tight">
        {t("securityNotice")}
      </p>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={status === "SUBMITTING"}
        className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-4 py-2.5 text-xs font-semibold text-white shadow-md shadow-sky-200 hover:from-sky-600 hover:to-sky-700 disabled:opacity-50 transition"
      >
        {status === "SUBMITTING" ? (
          <>
            <Loader2 className="size-4 animate-spin" />
            <span>{t("resetting")}</span>
          </>
        ) : (
          <>
            <KeyRound className="size-4" />
            <span>{t("resetButton")}</span>
          </>
        )}
      </button>

      <div className="text-center pt-2">
        <Link
          href="/login"
          className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-600 hover:text-sky-800 transition"
        >
          <ArrowLeft className="size-3.5" />
          <span>{tl("signIn")}</span>
        </Link>
      </div>
    </form>
  );
}
