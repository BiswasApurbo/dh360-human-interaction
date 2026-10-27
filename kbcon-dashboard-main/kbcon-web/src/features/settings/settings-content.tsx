"use client";

import { useEffect, useState } from "react";
import { useLocale, useTranslations } from "next-intl";
import { motion } from "motion/react";
import { sendPasswordResetEmail, updateProfile } from "firebase/auth";
import { Check, KeyRound, Languages, Loader2, UserRound } from "lucide-react";
import { usePathname, useRouter } from "@/i18n/navigation";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/features/auth/use-auth";
import { RetentionCard } from "./retention-card";

const cardClass =
  "rounded-3xl border border-sky-200 bg-gradient-to-br from-white via-sky-50 to-sky-100 p-5 shadow-lg shadow-sky-200/60 sm:p-6";

type Msg = { ok: boolean; text: string } | null;

function Banner({ msg }: { msg: Msg }) {
  if (!msg) return null;
  return (
    <p
      className={`rounded-lg px-3 py-2 text-sm ring-1 ${
        msg.ok
          ? "bg-emerald-50 text-emerald-700 ring-emerald-200"
          : "bg-red-50 text-red-600 ring-red-200"
      }`}
    >
      {msg.text}
    </p>
  );
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
  const [sending, setSending] = useState(false);
  const [resetMsg, setResetMsg] = useState<Msg>(null);

  useEffect(() => {
    if (user) setName(user.displayName ?? "");
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

  async function sendReset() {
    if (!user?.email) return;
    setSending(true);
    setResetMsg(null);
    try {
      await sendPasswordResetEmail(
        // auth ইনস্ট্যান্স user থেকেই পাওয়া যায়
        (await import("@/lib/firebase")).auth,
        user.email,
      );
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

      {/* ভাষা */}
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

      {/* প্রোফাইল */}
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

      {/* পাসওয়ার্ড */}
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
      </motion.section>
      <RetentionCard />
    </div>
  );
}
