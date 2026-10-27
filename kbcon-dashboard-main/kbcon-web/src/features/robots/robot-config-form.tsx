"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import { httpsCallable } from "firebase/functions";
import { Loader2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { functions } from "@/lib/firebase";
import type { RobotRow } from "./use-robots";

export function RobotConfigForm({
  robot,
  onClose,
}: {
  robot: RobotRow;
  onClose: () => void;
}) {
  const t = useTranslations("Robots");
  const c = robot.config;

  const [lang, setLang] = useState<"bn" | "en">(
    c.defaultLanguage === "en" ? "en" : "bn",
  );
  const [interval, setIntervalSec] = useState(
    String(Number(c.telemetryIntervalSec) || 30),
  );
  const [fbBn, setFbBn] = useState(
    typeof c.fallbackBn === "string"
      ? c.fallbackBn
      : "দুঃখিত, আমি এই প্রশ্নের উত্তর জানি না।",
  );
  const [fbEn, setFbEn] = useState(
    typeof c.fallbackEn === "string"
      ? c.fallbackEn
      : "Sorry, I do not know the answer to that.",
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
      const fn = httpsCallable(functions, "updateRobotConfig");
      await fn({
        robotId: robot.id,
        config: {
          defaultLanguage: lang,
          telemetryIntervalSec: n,
          fallbackBn: fbBn.trim(),
          fallbackEn: fbEn.trim(),
        },
      });
      setMsg({ ok: true, text: t("configSaved") });
    } catch {
      setMsg({ ok: false, text: t("configFailed") });
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-4 rounded-2xl border border-sky-200 bg-white/80 p-4 shadow-sm sm:p-5">
      <h2 className="text-base font-bold text-sky-900">
        {t("configTitle")}:{" "}
        <span className="font-mono text-sm">{robot.id}</span> {robot.name}
      </h2>

      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-1.5">
          <Label htmlFor="cfg-lang" className="text-sky-900">
            {t("defaultLanguage")}
          </Label>
          <select
            id="cfg-lang"
            value={lang}
            onChange={(e) => setLang(e.target.value as "bn" | "en")}
            className="h-10 w-full rounded-md border border-sky-200 bg-white px-3 text-sm text-sky-900 outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
          >
            <option value="bn">{t("langBn")}</option>
            <option value="en">{t("langEn")}</option>
          </select>
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="cfg-int" className="text-sky-900">
            {t("telemetryInterval")}
          </Label>
          <Input
            id="cfg-int"
            type="number"
            min={10}
            max={300}
            value={interval}
            onChange={(e) => setIntervalSec(e.target.value)}
            className="h-10 border-sky-200 bg-white"
          />
          <p className="text-[11px] text-sky-700">
            {t("telemetryIntervalHint")}
          </p>
        </div>

        <div className="space-y-1.5 sm:col-span-2">
          <Label htmlFor="cfg-bn" className="text-sky-900">
            {t("fallbackBn")}
          </Label>
          <Input
            id="cfg-bn"
            value={fbBn}
            onChange={(e) => setFbBn(e.target.value)}
            className="h-10 border-sky-200 bg-white"
          />
        </div>

        <div className="space-y-1.5 sm:col-span-2">
          <Label htmlFor="cfg-en" className="text-sky-900">
            {t("fallbackEn")}
          </Label>
          <Input
            id="cfg-en"
            value={fbEn}
            onChange={(e) => setFbEn(e.target.value)}
            className="h-10 border-sky-200 bg-white"
          />
        </div>
      </div>

      {msg && (
        <p
          className={`rounded-lg px-3 py-2 text-sm ring-1 ${
            msg.ok
              ? "bg-emerald-50 text-emerald-700 ring-emerald-200"
              : "bg-red-50 text-red-600 ring-red-200"
          }`}
        >
          {msg.text}
        </p>
      )}

      <div className="flex flex-wrap justify-end gap-2">
        <button
          type="button"
          onClick={onClose}
          className="h-10 rounded-xl px-4 text-sm font-medium text-sky-800 ring-1 ring-sky-300 hover:bg-sky-50"
        >
          {t("cancel")}
        </button>
        <motion.button
          type="button"
          onClick={save}
          disabled={saving}
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
          className="flex h-10 items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-4 text-sm font-semibold text-white shadow-md shadow-sky-300/60 disabled:opacity-70"
        >
          {saving && <Loader2 className="size-4 animate-spin" />}
          {saving ? t("savingConfig") : t("saveConfig")}
        </motion.button>
      </div>
    </div>
  );
}
