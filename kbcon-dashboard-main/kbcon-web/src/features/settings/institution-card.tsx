"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import { Building2, GraduationCap, Hospital, Loader2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/features/auth/use-auth";

const cardClass =
  "rounded-3xl border border-sky-200 bg-gradient-to-br from-white via-sky-50 to-sky-100 p-5 shadow-lg shadow-sky-200/60 sm:p-6";

export function InstitutionCard() {
  const t = useTranslations("Settings");
  const { user } = useAuth();
  const [instType, setInstType] = useState<"university" | "clinic">("university");
  const [instName, setInstName] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    let active = true;
    async function loadSettings() {
      if (!user) return;
      try {
        const token = await user.getIdToken();
        const res = await fetch("/api/dh360/settings/institution", {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
          const data = await res.json();
          if (active) {
            if (data.institution_type === "clinic" || data.institution_type === "hospital") {
              setInstType("clinic");
            } else {
              setInstType("university");
            }
            if (typeof data.institution_name === "string") {
              setInstName(data.institution_name);
            }
          }
        }
      } catch {
        // Fallback to default
      } finally {
        if (active) setLoading(false);
      }
    }

    loadSettings();
    return () => {
      active = false;
    };
  }, [user]);

  async function handleSave() {
    if (!user) return;
    setSaving(true);
    setMsg(null);
    try {
      const token = await user.getIdToken();
      const res = await fetch("/api/dh360/settings/institution", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          institution_type: instType,
          institution_name: instName.trim(),
        }),
      });

      if (res.ok) {
        setMsg({ ok: true, text: t("institutionSaved") });
      } else {
        setMsg({ ok: false, text: t("institutionSaveFailed") });
      }
    } catch {
      setMsg({ ok: false, text: t("institutionSaveFailed") });
    } finally {
      setSaving(false);
    }
  }

  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.35 }}
      className={cardClass}
    >
      <div className="mb-4 flex items-center gap-3">
        <span className="flex size-10 items-center justify-center rounded-xl bg-white text-sky-700 ring-1 ring-sky-300">
          <Building2 className="size-5" />
        </span>
        <div>
          <h2 className="text-base font-bold text-sky-900">
            {t("institutionTitle")}
          </h2>
          <p className="text-xs text-sky-700">{t("institutionDesc")}</p>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center gap-2 py-4 text-xs text-sky-700">
          <Loader2 className="size-4 animate-spin" />
          <span>Loading settings...</span>
        </div>
      ) : (
        <div className="space-y-4">
          <div>
            <Label className="mb-2 block text-xs font-semibold text-sky-900">
              {t("institutionType")}
            </Label>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              <button
                type="button"
                onClick={() => setInstType("university")}
                className={`flex items-center gap-3 rounded-2xl border p-3.5 text-left text-sm font-medium transition-all ${
                  instType === "university"
                    ? "border-sky-600 bg-sky-600/10 text-sky-950 ring-2 ring-sky-600"
                    : "border-sky-200 bg-white/70 text-slate-700 hover:border-sky-300 hover:bg-white"
                }`}
              >
                <div
                  className={`flex size-8 shrink-0 items-center justify-center rounded-xl ${
                    instType === "university"
                      ? "bg-sky-600 text-white"
                      : "bg-sky-100 text-sky-700"
                  }`}
                >
                  <GraduationCap className="size-4" />
                </div>
                <div>
                  <div className="font-semibold">{t("typeUniversity")}</div>
                  <div className="text-xs text-slate-500">
                    Akib, Tanvir, Tuhin, Admissions, Labs
                  </div>
                </div>
              </button>

              <button
                type="button"
                onClick={() => setInstType("clinic")}
                className={`flex items-center gap-3 rounded-2xl border p-3.5 text-left text-sm font-medium transition-all ${
                  instType === "clinic"
                    ? "border-sky-600 bg-sky-600/10 text-sky-950 ring-2 ring-sky-600"
                    : "border-sky-200 bg-white/70 text-slate-700 hover:border-sky-300 hover:bg-white"
                }`}
              >
                <div
                  className={`flex size-8 shrink-0 items-center justify-center rounded-xl ${
                    instType === "clinic"
                      ? "bg-sky-600 text-white"
                      : "bg-sky-100 text-sky-700"
                  }`}
                >
                  <Hospital className="size-4" />
                </div>
                <div>
                  <div className="font-semibold">{t("typeClinic")}</div>
                  <div className="text-xs text-slate-500">
                    OPD, Visiting Hours, Emergency, Pharmacy
                  </div>
                </div>
              </button>
            </div>
          </div>

          <div>
            <Label htmlFor="inst-name" className="mb-1 block text-xs font-semibold text-sky-900">
              {t("institutionName")}
            </Label>
            <Input
              id="inst-name"
              type="text"
              value={instName}
              onChange={(e) => setInstName(e.target.value)}
              placeholder={t("institutionNamePlaceholder")}
              className="border-sky-200 bg-white/90 text-sm focus-visible:ring-sky-500"
            />
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

          <div className="pt-1">
            <button
              type="button"
              onClick={handleSave}
              disabled={saving}
              className="inline-flex items-center gap-2 rounded-xl bg-sky-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-sky-700 focus:outline-none focus:ring-2 focus:ring-sky-500 disabled:opacity-50"
            >
              {saving && <Loader2 className="size-4 animate-spin" />}
              <span>{t("saveInstitution")}</span>
            </button>
          </div>
        </div>
      )}
    </motion.section>
  );
}
