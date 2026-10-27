"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { AnimatePresence, motion } from "motion/react";
import { httpsCallable } from "firebase/functions";
import { Loader2, Plus, ShieldAlert } from "lucide-react";
import { functions } from "@/lib/firebase";
import { useRole } from "@/features/auth/use-role";
import { useAdmins } from "./use-admins";
import { CreateAdminForm } from "./create-admin-form";

export function AdminsPageContent() {
  const t = useTranslations("Admins");
  const { role, loading: roleLoading } = useRole();
  const isSuper = role === "superAdmin";
  const { admins, loading, error } = useAdmins(isSuper);
  const [showForm, setShowForm] = useState(false);
  const [busyUid, setBusyUid] = useState<string | null>(null);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(
    null,
  );

  async function toggle(uid: string, active: boolean) {
    setBusyUid(uid);
    setMessage(null);
    try {
      const fn = httpsCallable(functions, "setAdminActive");
      await fn({ uid, active });
      setMessage({ ok: true, text: t("updated") });
    } catch {
      setMessage({ ok: false, text: t("updateFailed") });
    } finally {
      setBusyUid(null);
    }
  }

  if (roleLoading) {
    return (
      <div className="flex justify-center py-16">
        <Loader2 className="size-7 animate-spin text-sky-500" />
      </div>
    );
  }

  if (!isSuper) {
    return (
      <div className="flex items-center gap-2 rounded-2xl bg-red-50 p-4 text-sm text-red-700 ring-1 ring-red-200">
        <ShieldAlert className="size-5 shrink-0" />
        {t("notAllowed")}
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-sky-900 sm:text-2xl">
            {t("title")}
          </h1>
          <p className="text-sm text-sky-800/80">{t("subtitle")}</p>
        </div>
        {!showForm && (
          <motion.button
            type="button"
            onClick={() => {
              setShowForm(true);
              setMessage(null);
            }}
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            className="flex h-10 items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-4 text-sm font-semibold text-white shadow-md shadow-sky-300/60"
          >
            <Plus className="size-4" />
            {t("newAdmin")}
          </motion.button>
        )}
      </div>

      <AnimatePresence>
        {showForm && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
          >
            <CreateAdminForm
              onCancel={() => setShowForm(false)}
              onDone={() => {
                setShowForm(false);
                setMessage({ ok: true, text: t("created") });
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {message && (
        <p
          className={`rounded-lg px-3 py-2 text-sm ring-1 ${
            message.ok
              ? "bg-emerald-50 text-emerald-700 ring-emerald-200"
              : "bg-red-50 text-red-600 ring-red-200"
          }`}
        >
          {message.text}
        </p>
      )}

      <div className="overflow-x-auto rounded-2xl border border-sky-200 bg-white/80 shadow-sm">
        <table className="w-full min-w-[520px] text-left text-sm">
          <thead className="bg-sky-100 text-sky-900">
            <tr>
              <th className="px-4 py-3 font-semibold">{t("name")}</th>
              <th className="px-4 py-3 font-semibold">{t("email")}</th>
              <th className="px-4 py-3 font-semibold">{t("status")}</th>
              <th className="px-4 py-3 text-right font-semibold">
                {t("actions")}
              </th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={4} className="px-4 py-8 text-center">
                  <Loader2 className="mx-auto size-6 animate-spin text-sky-500" />
                </td>
              </tr>
            )}
            {!loading && error && (
              <tr>
                <td colSpan={4} className="px-4 py-8 text-center text-red-600">
                  {t("loadFailed")}
                </td>
              </tr>
            )}
            {!loading && !error && admins.length === 0 && (
              <tr>
                <td colSpan={4} className="px-4 py-8 text-center text-sky-700">
                  {t("empty")}
                </td>
              </tr>
            )}
            {admins.map((a) => (
              <motion.tr
                key={a.uid}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="border-t border-sky-100"
              >
                <td className="px-4 py-3 font-medium text-sky-900">{a.name}</td>
                <td className="px-4 py-3 text-sky-800">{a.email}</td>
                <td className="px-4 py-3">
                  <span
                    className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
                      a.active
                        ? "bg-emerald-100 text-emerald-700"
                        : "bg-slate-200 text-slate-600"
                    }`}
                  >
                    {a.active ? t("active") : t("disabled")}
                  </span>
                </td>
                <td className="px-4 py-3 text-right">
                  <button
                    type="button"
                    disabled={busyUid === a.uid}
                    onClick={() => toggle(a.uid, !a.active)}
                    className="inline-flex h-8 items-center gap-1.5 rounded-lg px-3 text-xs font-semibold text-sky-800 ring-1 ring-sky-300 hover:bg-sky-50 disabled:opacity-60"
                  >
                    {busyUid === a.uid && (
                      <Loader2 className="size-3 animate-spin" />
                    )}
                    {a.active ? t("disable") : t("enable")}
                  </button>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
