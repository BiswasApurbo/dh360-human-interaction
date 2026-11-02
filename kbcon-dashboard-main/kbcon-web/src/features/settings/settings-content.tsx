"use client";

import { useEffect, useState } from "react";
import { useLocale, useTranslations } from "next-intl";
import { motion } from "motion/react";
import {
  EmailAuthProvider,
  reauthenticateWithCredential,
  sendPasswordResetEmail,
  updatePassword,
  updateProfile,
} from "firebase/auth";
import {
  AlertCircle,
  Check,
  CheckCircle2,
  Eye,
  EyeOff,
  KeyRound,
  Languages,
  Loader2,
  UserRound,
} from "lucide-react";
import { usePathname, useRouter } from "@/i18n/navigation";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/features/auth/use-auth";

const cardClass =
  "rounded-3xl border border-sky-200 bg-gradient-to-br from-white via-sky-50 to-sky-100 p-5 shadow-lg shadow-sky-200/60 sm:p-6";

type Msg = { ok: boolean; text: string } | null;

function Banner({ msg }: { msg: Msg }) {
  if (!msg) return null;
  return (
    <div
      className={`flex items-center gap-2 rounded-xl px-3 py-2 text-sm ring-1 ${
        msg.ok
          ? "bg-emerald-50 text-emerald-700 ring-emerald-200"
          : "bg-red-50 text-red-600 ring-red-200"
      }`}
    >
      {msg.ok ? (
        <CheckCircle2 className="size-4 shrink-0 text-emerald-600" />
      ) : (
        <AlertCircle className="size-4 shrink-0 text-red-600" />
      )}
      <span>{msg.text}</span>
    </div>
  );
}

function evaluatePasswordStrength(password: string): {
  score: number;
  label: string;
  color: string;
} {
  const checks = {
    length: password.length >= 8,
    lower: /[a-z]/.test(password),
    upper: /[A-Z]/.test(password),
    numberOrSpecial: /[0-9!@#$%^&*()_+\-=[\]{};':"\\|,.<>/?]/.test(password),
  };
  const count = Object.values(checks).filter(Boolean).length;
  if (count <= 1) return { score: 25, label: "Weak", color: "bg-red-500" };
  if (count <= 2) return { score: 50, label: "Fair", color: "bg-amber-500" };
  if (count <= 3) return { score: 75, label: "Good", color: "bg-sky-500" };
  return { score: 100, label: "Strong", color: "bg-emerald-500" };
}

export function SettingsContent() {
  const t = useTranslations("Settings");
  const locale = useLocale();
  const router = useRouter();
  const pathname = usePathname();
  const { user } = useAuth();

  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);
  const [profileMsg, setProfileMsg] = useState<Msg>(null);

  // Email reset state
  const [sending, setSending] = useState(false);
  const [resetMsg, setResetMsg] = useState<Msg>(null);

  // Direct password change state
  const [oldPassword, setOldPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showOld, setShowOld] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [changingPass, setChangingPass] = useState(false);
  const [directPassMsg, setDirectPassMsg] = useState<Msg>(null);

  const strength = evaluatePasswordStrength(newPassword);

  useEffect(() => {
    if (user) {
      const id = setTimeout(() => setName(user.displayName ?? ""), 0);
      return () => clearTimeout(id);
    }
  }, [user]);

  async function saveName() {
    if (!user) return;
    const trimmed = name.trim();
    if (trimmed.length < 2) {
      setProfileMsg({ ok: false, text: t("nameShort") });
      return;
    }
    setSaving(true);
    setProfileMsg(null);
    try {
      await updateProfile(user, { displayName: trimmed });
      setProfileMsg({ ok: true, text: t("saved") });
    } catch {
      setProfileMsg({ ok: false, text: t("saveFailed") });
    } finally {
      setSaving(false);
    }
  }

  async function handleDirectPasswordChange(e: React.FormEvent) {
    e.preventDefault();
    if (!user || !user.email) return;

    setDirectPassMsg(null);

    if (!oldPassword) {
      setDirectPassMsg({ ok: false, text: t("oldPasswordIncorrect") });
      return;
    }
    if (newPassword.length < 8 || strength.score < 50) {
      setDirectPassMsg({ ok: false, text: t("weakPassword") });
      return;
    }
    if (newPassword === oldPassword) {
      setDirectPassMsg({ ok: false, text: t("samePasswordError") });
      return;
    }
    if (newPassword !== confirmPassword) {
      setDirectPassMsg({ ok: false, text: t("passwordMismatch") });
      return;
    }

    setChangingPass(true);
    try {
      // 1. Reauthenticate with old password
      const credential = EmailAuthProvider.credential(user.email, oldPassword);
      await reauthenticateWithCredential(user, credential);

      // 2. Update Firebase Authentication password
      await updatePassword(user, newPassword);

      // 3. Clear fields
      setOldPassword("");
      setNewPassword("");
      setConfirmPassword("");

      setDirectPassMsg({ ok: true, text: t("passwordChangedSuccess") });
    } catch (err: unknown) {
      const code = (err as { code?: string })?.code;
      if (code === "auth/invalid-credential" || code === "auth/wrong-password") {
        setDirectPassMsg({ ok: false, text: t("oldPasswordIncorrect") });
      } else if (code === "auth/weak-password") {
        setDirectPassMsg({ ok: false, text: t("weakPassword") });
      } else {
        setDirectPassMsg({ ok: false, text: t("saveFailed") });
      }
    } finally {
      setChangingPass(false);
    }
  }

  async function sendReset() {
    if (!user?.email) return;
    setSending(true);
    setResetMsg(null);
    try {
      const { auth } = await import("@/lib/firebase");
      const actionCodeSettings = {
        url: "https://kbcon-dashboard-gamma.vercel.app/auth/action",
        handleCodeInApp: false,
      };
      await sendPasswordResetEmail(auth, user.email, actionCodeSettings);
      setResetMsg({ ok: true, text: t("resetSent") });
    } catch {
      setResetMsg({ ok: false, text: t("resetFailed") });
    } finally {
      setSending(false);
    }
  }

  const languages = [
    { code: "en", label: "English" },
    { code: "bn", label: "বাংলা" },
  ] as const;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-bold text-sky-900 sm:text-2xl">
          {t("title")}
        </h1>
        <p className="text-sm text-sky-800/80">{t("subtitle")}</p>
      </div>

      {/* Language */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className={cardClass}
      >
        <div className="mb-4 flex items-center gap-3">
          <span className="flex size-10 items-center justify-center rounded-xl bg-white text-sky-700 ring-1 ring-sky-300">
            <Languages className="size-5" />
          </span>
          <div>
            <h2 className="text-base font-bold text-sky-900">
              {t("languageTitle")}
            </h2>
            <p className="text-xs text-sky-700">{t("languageDesc")}</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-3">
          {languages.map((l) => {
            const active = locale === l.code;
            return (
              <motion.button
                key={l.code}
                type="button"
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.96 }}
                onClick={() => router.replace(pathname, { locale: l.code })}
                className={`flex h-11 items-center gap-2 rounded-xl px-5 text-sm font-semibold ring-1 transition-colors ${
                  active
                    ? "bg-gradient-to-r from-sky-500 to-sky-600 text-white shadow-md shadow-sky-300/60 ring-sky-500"
                    : "bg-white/80 text-sky-800 ring-sky-300 hover:bg-sky-50"
                }`}
              >
                {active && <Check className="size-4" />}
                {l.label}
              </motion.button>
            );
          })}
        </div>
      </motion.section>

      {/* Profile */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
        className={cardClass}
      >
        <div className="mb-4 flex items-center gap-3">
          <span className="flex size-10 items-center justify-center rounded-xl bg-white text-sky-700 ring-1 ring-sky-300">
            <UserRound className="size-5" />
          </span>
          <div>
            <h2 className="text-base font-bold text-sky-900">
              {t("profileTitle")}
            </h2>
            <p className="text-xs text-sky-700">{t("profileDesc")}</p>
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="s-name" className="text-sky-900">
              {t("name")}
            </Label>
            <Input
              id="s-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="h-10 border-sky-200 bg-white"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="s-email" className="text-sky-900">
              {t("email")}
            </Label>
            <Input
              id="s-email"
              value={user?.email ?? ""}
              readOnly
              disabled
              className="h-10 border-sky-200 bg-sky-50"
            />
          </div>
        </div>

        <div className="mt-4 space-y-3">
          <Banner msg={profileMsg} />
          <div className="flex justify-end">
            <motion.button
              type="button"
              onClick={saveName}
              disabled={saving}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className="flex h-10 items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-5 text-sm font-semibold text-white shadow-md shadow-sky-300/60 disabled:opacity-70"
            >
              {saving && <Loader2 className="size-4 animate-spin" />}
              {saving ? t("saving") : t("save")}
            </motion.button>
          </div>
        </div>
      </motion.section>

      {/* Password Management */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.2 }}
        className={cardClass}
      >
        <div className="mb-4 flex items-center gap-3">
          <span className="flex size-10 items-center justify-center rounded-xl bg-white text-sky-700 ring-1 ring-sky-300">
            <KeyRound className="size-5" />
          </span>
          <div>
            <h2 className="text-base font-bold text-sky-900">
              {t("passwordTitle")}
            </h2>
            <p className="text-xs text-sky-700">{t("passwordDesc")}</p>
          </div>
        </div>

        {/* Direct Password Change Form */}
        <form onSubmit={handleDirectPasswordChange} className="space-y-4">
          <h3 className="text-xs font-bold uppercase tracking-wider text-sky-900">
            {t("changePasswordDirect")}
          </h3>

          <div className="grid gap-3.5 sm:grid-cols-3">
            {/* Old Password */}
            <div className="space-y-1.5">
              <Label htmlFor="old-p" className="text-xs font-semibold text-sky-900">
                {t("oldPassword")}
              </Label>
              <div className="relative">
                <Input
                  id="old-p"
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
              <Label htmlFor="new-p" className="text-xs font-semibold text-sky-900">
                {t("newPassword")}
              </Label>
              <div className="relative">
                <Input
                  id="new-p"
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
              {newPassword && (
                <div className="space-y-1 pt-1">
                  <div className="flex justify-between text-[10px] font-semibold text-slate-600">
                    <span>Strength:</span>
                    <span>{strength.label}</span>
                  </div>
                  <div className="h-1 w-full overflow-hidden rounded-full bg-slate-200">
                    <div
                      className={`h-full transition-all duration-300 ${strength.color}`}
                      style={{ width: `${strength.score}%` }}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Confirm New Password */}
            <div className="space-y-1.5">
              <Label htmlFor="conf-p" className="text-xs font-semibold text-sky-900">
                {t("confirmNewPassword")}
              </Label>
              <div className="relative">
                <Input
                  id="conf-p"
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
          </div>

          <Banner msg={directPassMsg} />

          <div className="flex justify-end pt-1">
            <motion.button
              type="submit"
              disabled={changingPass || strength.score < 50}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="flex h-10 items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-5 text-sm font-semibold text-white shadow-md shadow-sky-300/60 disabled:opacity-50"
            >
              {changingPass && <Loader2 className="size-4 animate-spin" />}
              {changingPass ? t("changingPassword") : t("changePasswordBtn")}
            </motion.button>
          </div>
        </form>

        {/* Existing Send Reset Link Section */}
        <div className="mt-6 border-t border-sky-200/60 pt-5">
          <p className="text-xs font-semibold text-sky-900 mb-2">
            {t("orEmailReset")}
          </p>
          <div className="space-y-3">
            <Banner msg={resetMsg} />
            <motion.button
              type="button"
              onClick={sendReset}
              disabled={sending}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className="flex h-10 items-center gap-2 rounded-xl bg-white/80 px-5 text-sm font-semibold text-sky-800 ring-1 ring-sky-300 hover:bg-sky-50 disabled:opacity-70"
            >
              {sending && <Loader2 className="size-4 animate-spin" />}
              {sending ? t("sending") : t("sendReset")}
            </motion.button>
          </div>
        </div>
      </motion.section>
    </div>
  );
}
