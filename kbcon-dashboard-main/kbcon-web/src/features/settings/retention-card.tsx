"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import { doc, getDoc } from "firebase/firestore";
import { httpsCallable } from "firebase/functions";
import { Database, Loader2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { db, functions } from "@/lib/firebase";
import { useAuth } from "@/features/auth/use-auth";

const cardClass =
  "rounded-3xl border border-sky-200 bg-gradient-to-br from-white via-sky-50 to-sky-100 p-5 shadow-lg shadow-sky-200/60 sm:p-6";

export function RetentionCard() {
  const t = useTranslations("Settings");
  const { user } = useAuth();
  const [days, setDays] = useState("30");
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    if (!user) return;
    getDoc(doc(db, "users", user.uid))
      .then((snap) => {
        const d = Number(snap.data()?.telemetryRetentionDays);
        if (Number.isInteger(d) && d >= 1 && d <= 90) setDays(String(d));
      })
      .catch(() => {});
  }, [user]);

  async function save() {
    const n = Number(days);
    if (!Number.isInteger(n) || n < 1 || n > 90) {
      setMsg({ ok: false, text: t("retentionRange") });
      return;
    }
    setSaving(true);
    setMsg(null);
    try {
      const fn = httpsCallable(functions, "setTelemetryRetention");
      await fn({ days: n });
      setMsg({ ok: true, text: t("retentionSaved") });
    } catch {
      setMsg({ ok: false, text: t("retentionFailed") });
    } finally {
      setSaving(false);
    }
  }

  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.3 }}
      className={cardClass}
    >
      <div className="mb-4 flex items-center gap-3">
        <span className="flex size-10 items-center justify-center rounded-xl bg-white text-sky-700 ring-1 ring-sky-300">
          <Database className="size-5" />
        </span>
        <div>
          <h2 className="text-base font-bold text-sky-900">
            {t("retentionTitle")}
          </h2>
          <p className="text-xs text-sky-700">{t("retentionDesc")}</p>
        </div>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <div className="space-y-1.5">
          <Label htmlFor="ret-days" className="text-sky-900">
            {t("retentionDays")}
          </Label>
          <Input
            id="ret-days"
            type="number"
            min={1}
            max={90}
            value={days}
            onChange={(e) => setDays(e.target.value)}
            className="h-10 w-28 border-sky-200 bg-white"
          />
        </div>
        <motion.button
          type="button"
          onClick={save}
          disabled={saving}
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
          className="flex h-10 items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-5 text-sm font-semibold text-white shadow-md shadow-sky-300/60 disabled:opacity-70"
        >
          {saving && <Loader2 className="size-4 animate-spin" />}
          {saving ? t("saving") : t("save")}
        </motion.button>
      </div>

      {msg && (
        <p
          className={`mt-3 rounded-lg px-3 py-2 text-sm ring-1 ${
            msg.ok
              ? "bg-emerald-50 text-emerald-700 ring-emerald-200"
              : "bg-red-50 text-red-600 ring-red-200"
          }`}
        >
          {msg.text}
        </p>
      )}
    </motion.section>
  );
}
