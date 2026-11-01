"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import {
  EmailAuthProvider,
  reauthenticateWithCredential,
  updatePassword,
} from "firebase/auth";
import {
  AlertCircle,
  CheckCircle2,
  Eye,
  EyeOff,
  KeyRound,
  Loader2,
  ShieldAlert,
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "./use-auth";

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

export function ForcePasswordChangeModal({
  onPasswordChanged,
}: {
  onPasswordChanged: () => void;
}) {
  const t = useTranslations("Settings");
  const tl = useTranslations("Login");
  const { user, logout } = useAuth();

  const [oldPassword, setOldPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const [showOld, setShowOld] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const strength = evaluatePasswordStrength(newPassword);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!user || !user.email) return;

    setError(null);

    // Validation checks
    if (!oldPassword) {
      setError(t("oldPasswordIncorrect"));
      return;
    }
    if (newPassword.length < 8 || strength.score < 50) {
      setError(t("weakPassword"));
      return;
    }
    if (newPassword === oldPassword) {
      setError(t("samePasswordError"));
      return;
    }
    if (newPassword !== confirmPassword) {
      setError(t("passwordMismatch"));
      return;
    }

    setSubmitting(true);
    try {
      // 1. Reauthenticate with old password
      const credential = EmailAuthProvider.credential(user.email, oldPassword);
      await reauthenticateWithCredential(user, credential);

      // 2. Update Firebase Authentication password
      await updatePassword(user, newPassword);

      // Successfully updated password in Firebase Auth!
      // Record resolution in sessionStorage so the active session is never trapped
      if (typeof window !== "undefined") {
        sessionStorage.setItem(`mustChangePasswordResolved_${user.uid}`, "true");
      }

      // 3. Inform server to clear temporary bootstrap status
      try {
        const token = await user.getIdToken();
        const res = await fetch("/api/auth/clear-temporary-password", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        });

        if (!res.ok) {
          const errJson = await res.json().catch(() => ({}));
          console.warn("Backend bootstrap status clear note:", errJson.message || res.statusText);
        }
      } catch (postErr) {
        console.warn("Backend bootstrap status network note:", postErr);
      }

      // 4. Force refresh ID token so claims update in browser
      await user.getIdToken(true);

      setSuccess(true);
      setTimeout(() => {
        onPasswordChanged();
      }, 1200);
    } catch (err: unknown) {
      const code = (err as { code?: string })?.code;
      if (code === "auth/invalid-credential" || code === "auth/wrong-password") {
        setError(t("oldPasswordIncorrect"));
      } else if (code === "auth/weak-password") {
        setError(t("weakPassword"));
      } else {
        setError(err instanceof Error ? err.message : t("saveFailed"));
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/80 backdrop-blur-md"
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="w-full max-w-lg rounded-3xl bg-white p-6 sm:p-8 shadow-2xl border border-sky-100"
      >
        <div className="flex items-center gap-3 border-b border-sky-100 pb-4">
          <div className="flex size-11 items-center justify-center rounded-2xl bg-amber-500 text-white shadow-md shadow-amber-300">
            <ShieldAlert className="size-6" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-sky-950">
              Mandatory Password Change
            </h2>
            <p className="text-xs text-sky-800">
              Security initialization for bootstrap administrator
            </p>
          </div>
        </div>

        <p className="mt-3 text-xs leading-relaxed text-slate-600 bg-amber-50/70 p-3 rounded-xl border border-amber-200">
          This account was initialized with a one-time temporary password. You
          must establish your own secure password before normal dashboard access
          can be granted.
        </p>

        {error && (
          <div className="mt-3 flex items-center gap-2 rounded-xl bg-red-50 p-3 text-xs font-medium text-red-700 ring-1 ring-red-200">
            <AlertCircle className="size-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {success && (
          <div className="mt-3 flex items-center gap-2 rounded-xl bg-emerald-50 p-3 text-xs font-medium text-emerald-700 ring-1 ring-emerald-200">
            <CheckCircle2 className="size-4 shrink-0" />
            <span>{t("passwordChangedSuccess")}</span>
          </div>
        )}

        {!success && (
          <form onSubmit={handleSubmit} className="mt-4 space-y-4">
            {/* Old Password */}
            <div className="space-y-1.5">
              <Label htmlFor="old-pass" className="text-xs font-semibold text-sky-900">
                {t("oldPassword")}
              </Label>
              <div className="relative">
                <Input
                  id="old-pass"
                  type={showOld ? "text" : "password"}
                  value={oldPassword}
                  onChange={(e) => setOldPassword(e.target.value)}
                  className="h-10 pr-10 border-sky-200 bg-white"
                  autoComplete="current-password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowOld(!showOld)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                >
                  {showOld ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                </button>
              </div>
            </div>

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
                  className="h-10 pr-10 border-sky-200 bg-white"
                  autoComplete="new-password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowNew(!showNew)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                >
                  {showNew ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                </button>
              </div>

              {/* Strength Bar */}
              {newPassword && (
                <div className="space-y-1 pt-1">
                  <div className="flex justify-between text-[11px] font-semibold text-slate-600">
                    <span>Password Strength:</span>
                    <span>{strength.label}</span>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
                    <div
                      className={`h-full transition-all duration-300 ${strength.color}`}
                      style={{ width: `${strength.score}%` }}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Confirm Password */}
            <div className="space-y-1.5">
              <Label htmlFor="conf-pass" className="text-xs font-semibold text-sky-900">
                {t("confirmNewPassword")}
              </Label>
              <div className="relative">
                <Input
                  id="conf-pass"
                  type={showConfirm ? "text" : "password"}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="h-10 pr-10 border-sky-200 bg-white"
                  autoComplete="new-password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowConfirm(!showConfirm)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                >
                  {showConfirm ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-slate-100">
              <button
                type="button"
                onClick={() => void logout()}
                className="text-xs font-semibold text-slate-600 hover:text-slate-900 hover:underline"
              >
                Sign out
              </button>
              <button
                type="submit"
                disabled={submitting || strength.score < 50}
                className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-emerald-600 px-5 py-2.5 text-xs font-bold text-white shadow-md shadow-emerald-300 hover:from-emerald-600 hover:to-emerald-700 disabled:opacity-50"
              >
                {submitting && <Loader2 className="size-3.5 animate-spin" />}
                <span>{t("changePasswordBtn")}</span>
              </button>
            </div>
          </form>
        )}
      </motion.div>
    </div>
  );
}
