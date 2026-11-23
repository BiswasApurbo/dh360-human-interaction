"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import { Loader2, ShieldCheck, Sparkles } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { auth } from "@/lib/firebase";
import { useRole } from "@/features/auth/use-role";
import type { RobotRow } from "./use-robots";

export function RobotConfigForm({
  robot,
  onClose,
}: {
  robot: RobotRow;
  onClose: () => void;
}) {
  const t = useTranslations("Robots");
  const { role } = useRole();
  const isSuper = role === "superAdmin";
  const c = robot.config;

  // Authoritative robot mode / type ("healthcare" | "education")
  const [mode, setMode] = useState<"healthcare" | "education">(
    robot.mode === "healthcare" || robot.type?.toLowerCase() === "health"
      ? "healthcare"
      : "education"
  );

  // Authoritative robot default language ("bn" | "en")
  const [lang, setLang] = useState<"bn" | "en">(
    robot.language === "en" || c.defaultLanguage === "en" ? "en" : "bn"
  );

  const [interval, setIntervalSec] = useState(
    String(Number(c.telemetryIntervalSec) || 30)
  );
  const [fbBn, setFbBn] = useState(
    typeof c.fallbackBn === "string"
      ? c.fallbackBn
      : "দুঃখিত, আমি এই প্রশ্নের উত্তর জানি না।"
  );
  const [fbEn, setFbEn] = useState(
    typeof c.fallbackEn === "string"
      ? c.fallbackEn
      : "Sorry, I do not know the answer to that."
  );
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  async function save() {
    const n = Number(interval);
    if (!Number.isInteger(n) || n < 10 || n > 300) {
      setMsg({ ok: false, text: t("intervalRange") });
      return;
    }
    if (!fbBn.trim() || !fbEn.trim()) {
      setMsg({ ok: false, text: t("fallbackRequired") });
      return;
    }
    setSaving(true);
    setMsg(null);
    try {
      const token = await auth.currentUser?.getIdToken();
      const res = await fetch("/api/admin/robots", {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          robotId: robot.id,
          // Synchronize mode and type
          mode,
          type: mode === "healthcare" ? "Health" : "Education",
          language: lang,
          config: {
            defaultLanguage: lang,
            telemetryIntervalSec: n,
            fallbackBn: fbBn.trim(),
            fallbackEn: fbEn.trim(),
          },
        }),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.message || data.error || t("configFailed"));
      }

      setMsg({ ok: true, text: t("configSaved") });
    } catch (err: unknown) {
      setMsg({
        ok: false,
        text: err instanceof Error ? err.message : t("configFailed"),
      });
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-4 rounded-2xl border border-sky-200 bg-white/90 p-4 shadow-sm sm:p-5 backdrop-blur-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-sky-100 pb-3">
        <div>
          <h2 className="text-base font-bold text-sky-950 flex items-center gap-2">
            <Sparkles className="size-4 text-sky-600" />
            {t("editRobotConfig")}
          </h2>
          <p className="text-xs text-sky-700 font-mono">
            {robot.name} • {robot.id}
          </p>
        </div>
        {isSuper && (
          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2.5 py-0.5 text-[11px] font-semibold text-emerald-700 border border-emerald-200">
            <ShieldCheck className="size-3" />
            Superadmin Access
          </span>
        )}
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {/* Dynamic Robot Type Selector */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center">
            <Label htmlFor="cfg-type" className="text-sky-950 font-semibold text-xs">
              {t("robotType")}
            </Label>
            {!isSuper && (
              <span className="text-[10px] text-amber-700">Read-only</span>
            )}
          </div>
          <select
            id="cfg-type"
            value={mode}
            disabled={!isSuper}
            onChange={(e) => setMode(e.target.value as "healthcare" | "education")}
            className="h-10 w-full rounded-xl border border-sky-200 bg-white px-3 text-sm font-medium text-sky-950 outline-none focus-visible:ring-2 focus-visible:ring-sky-400 disabled:opacity-60 disabled:bg-slate-50 transition-all cursor-pointer"
          >
            <option value="healthcare">{t("typeHealth")}</option>
            <option value="education">{t("typeEducation")}</option>
          </select>
          <p className="text-[11px] text-sky-700/80">
            {mode === "healthcare"
              ? "Health mode activates medical triage context and healthcare assistant suggestions."
              : "Education mode activates classroom context, quiz modules, and educational assistant suggestions."}
          </p>
        </div>

        {/* Dynamic Robot Language Selector */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center">
            <Label htmlFor="cfg-lang" className="text-sky-950 font-semibold text-xs">
              {t("robotLanguage")}
            </Label>
            {!isSuper && (
              <span className="text-[10px] text-amber-700">Read-only</span>
            )}
          </div>
          <select
            id="cfg-lang"
            value={lang}
            disabled={!isSuper}
            onChange={(e) => setLang(e.target.value as "bn" | "en")}
            className="h-10 w-full rounded-xl border border-sky-200 bg-white px-3 text-sm font-medium text-sky-950 outline-none focus-visible:ring-2 focus-visible:ring-sky-400 disabled:opacity-60 disabled:bg-slate-50 transition-all cursor-pointer"
          >
            <option value="bn">{t("langBnFull")}</option>
            <option value="en">{t("langEnFull")}</option>
          </select>
          <p className="text-[11px] text-sky-700/80">
            Controls the robot&apos;s default conversational speech and greeting language.
          </p>
        </div>

        {/* Telemetry Interval */}
        <div className="space-y-1.5 sm:col-span-2">
          <Label htmlFor="cfg-int" className="text-sky-950 font-semibold text-xs">
            {t("telemetryInterval")}
          </Label>
          <Input
            id="cfg-int"
            type="number"
            min={10}
            max={300}
            value={interval}
            onChange={(e) => setIntervalSec(e.target.value)}
            className="h-10 border-sky-200 bg-white rounded-xl"
          />
          <p className="text-[11px] text-sky-700">
            {t("telemetryIntervalHint")}
          </p>
        </div>

        {/* Fallback Responses */}
        <div className="space-y-1.5 sm:col-span-2">
          <Label htmlFor="cfg-bn" className="text-sky-950 font-semibold text-xs">
            {t("fallbackBn")}
          </Label>
          <Input
            id="cfg-bn"
            value={fbBn}
            onChange={(e) => setFbBn(e.target.value)}
            className="h-10 border-sky-200 bg-white rounded-xl"
          />
        </div>

        <div className="space-y-1.5 sm:col-span-2">
          <Label htmlFor="cfg-en" className="text-sky-950 font-semibold text-xs">
            {t("fallbackEn")}
          </Label>
          <Input
            id="cfg-en"
            value={fbEn}
            onChange={(e) => setFbEn(e.target.value)}
            className="h-10 border-sky-200 bg-white rounded-xl"
          />
        </div>
      </div>

      {msg && (
        <p
          className={`rounded-xl px-3.5 py-2.5 text-xs font-semibold ring-1 ${
            msg.ok
              ? "bg-emerald-50 text-emerald-800 ring-emerald-200"
              : "bg-red-50 text-red-700 ring-red-200"
          }`}
        >
          {msg.text}
        </p>
      )}

      <div className="flex flex-wrap justify-end gap-2 pt-2 border-t border-sky-100">
        <button
          type="button"
          onClick={onClose}
          className="h-9 rounded-xl px-4 text-xs font-semibold text-sky-800 ring-1 ring-sky-300 hover:bg-sky-50 transition-colors"
        >
          {t("cancel")}
        </button>
        <motion.button
          type="button"
          onClick={save}
          disabled={saving}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="flex h-9 items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-4 text-xs font-bold text-white shadow-md shadow-sky-300/60 hover:from-sky-600 hover:to-sky-700 disabled:opacity-70 transition-all"
        >
          {saving && <Loader2 className="size-3.5 animate-spin" />}
          {saving ? t("savingConfig") : t("saveConfig")}
        </motion.button>
      </div>
    </div>
  );
}
