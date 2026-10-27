"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { AnimatePresence, motion } from "motion/react";
import { httpsCallable } from "firebase/functions";
import { Check, Copy, KeyRound, Loader2, Plus } from "lucide-react";
import { functions } from "@/lib/firebase";
import { useRole } from "@/features/auth/use-role";
import { useAdmins } from "@/features/admins/use-admins";
import { useRobots } from "./use-robots";
import { CreateRobotForm } from "./create-robot-form";
import { effectiveStatus } from "@/lib/robot-status";
import { useNow } from "@/hooks/use-now";
import { RobotConfigForm } from "./robot-config-form";
import { Link } from "@/i18n/navigation";

const statusStyle = {
  online: "bg-emerald-100 text-emerald-700",
  offline: "bg-slate-200 text-slate-600",
  connecting: "bg-amber-100 text-amber-700",
  error: "bg-red-100 text-red-700",
} as const;

export function RobotsPageContent() {
  const t = useTranslations("Robots");
  const { role, loading: roleLoading } = useRole();
  const isSuper = role === "superAdmin";
  const { robots, loading, error } = useRobots(role);
  const { admins } = useAdmins(isSuper);
  const now = useNow();
  const [showForm, setShowForm] = useState(false);
  const [secret, setSecret] = useState<{ id: string; key: string } | null>(
    null,
  );

  const [configId, setConfigId] = useState<string | null>(null);
  const configRobot = robots.find((r) => r.id === configId) ?? null;
  const [copied, setCopied] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(
    null,
  );

  async function assign(robotId: string, adminId: string) {
    setBusyId(robotId);
    setMessage(null);
    try {
      const fn = httpsCallable(functions, "assignRobot");
      await fn({ robotId, adminId: adminId === "" ? null : adminId });
      setMessage({ ok: true, text: t("assigned") });
    } catch {
      setMessage({ ok: false, text: t("assignFailed") });
    } finally {
      setBusyId(null);
    }
  }

  async function copyKey() {
    if (!secret) return;
    try {
      await navigator.clipboard.writeText(secret.key);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // কপি ব্যর্থ হলে ব্যবহারকারী হাতে কপি করবেন
    }
  }

  if (roleLoading) {
    return (
      <div className="flex justify-center py-16">
        <Loader2 className="size-7 animate-spin text-sky-500" />
      </div>
    );
  }

  const colSpan = isSuper ? 8 : 7;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-sky-900 sm:text-2xl">
            {t("title")}
          </h1>
          <p className="text-sm text-sky-800/80">
            {isSuper ? t("subtitle") : t("mySubtitle")}
          </p>
        </div>
        {isSuper && !showForm && (
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
            {t("newRobot")}
          </motion.button>
        )}
      </div>

      <AnimatePresence>
        {isSuper && showForm && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
          >
            <CreateRobotForm
              onCancel={() => setShowForm(false)}
              onDone={(key, id) => {
                setShowForm(false);
                setSecret({ id, key });
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {secret && (
        <div className="space-y-3 rounded-2xl border border-amber-300 bg-amber-50 p-4">
          <div className="flex items-center gap-2 text-amber-800">
            <KeyRound className="size-5" />
            <h2 className="text-sm font-bold">
              {t("secretTitle")} ({secret.id})
            </h2>
          </div>
          <p className="text-xs text-amber-800">{t("secretWarn")}</p>
          <code className="block break-all rounded-lg bg-white px-3 py-2 text-sm text-slate-800 ring-1 ring-amber-200">
            {secret.key}
          </code>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={copyKey}
              className="inline-flex h-9 items-center gap-2 rounded-lg bg-white px-3 text-xs font-semibold text-amber-800 ring-1 ring-amber-300 hover:bg-amber-100"
            >
              {copied ? (
                <Check className="size-4" />
              ) : (
                <Copy className="size-4" />
              )}
              {copied ? t("copied") : t("copy")}
            </button>
            <button
              type="button"
              onClick={() => setSecret(null)}
              className="inline-flex h-9 items-center rounded-lg px-3 text-xs font-semibold text-amber-800 ring-1 ring-amber-300 hover:bg-amber-100"
            >
              {t("done")}
            </button>
          </div>
        </div>
      )}

      {configRobot && (
        <RobotConfigForm
          key={configRobot.id}
          robot={configRobot}
          onClose={() => setConfigId(null)}
        />
      )}

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
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead className="bg-sky-100 text-sky-900">
            <tr>
              <th className="px-4 py-3 font-semibold">{t("robotId")}</th>
              <th className="px-4 py-3 font-semibold">{t("name")}</th>
              <th className="px-4 py-3 font-semibold">{t("mode")}</th>
              <th className="px-4 py-3 font-semibold">{t("status")}</th>
              {isSuper ? (
                <th className="px-4 py-3 font-semibold">{t("assignedTo")}</th>
              ) : null}
              <th className="px-4 py-3 font-semibold">{t("type")}</th>
              <th className="px-4 py-3 font-semibold">{t("firmware")}</th>
              <th className="px-4 py-3 font-semibold" />
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={colSpan} className="px-4 py-8 text-center">
                  <Loader2 className="mx-auto size-6 animate-spin text-sky-500" />
                </td>
              </tr>
            )}
            {!loading && error && (
              <tr>
                <td
                  colSpan={colSpan}
                  className="px-4 py-8 text-center text-red-600"
                >
                  {t("loadFailed")}
                </td>
              </tr>
            )}
            {!loading && !error && robots.length === 0 && (
              <tr>
                <td
                  colSpan={colSpan}
                  className="px-4 py-8 text-center text-sky-700"
                >
                  {t("empty")}
                </td>
              </tr>
            )}
            {robots.map((r) => (
              <motion.tr
                key={r.id}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="border-t border-sky-100"
              >
                <td className="px-4 py-3 font-mono text-xs font-semibold text-sky-900">
                  <Link
                    href={`/robots/${r.id}`}
                    className="text-sky-700 underline-offset-2 hover:underline"
                  >
                    {r.id}
                  </Link>
                </td>
                <td className="px-4 py-3 font-medium text-sky-900">{r.name}</td>
                <td className="px-4 py-3 text-sky-800">{t(r.mode)}</td>
                <td className="px-4 py-3">
                  <span
                    className={`rounded-full px-2.5 py-1 text-xs font-semibold ${statusStyle[effectiveStatus(r, now)]}`}
                  >
                    {t(effectiveStatus(r, now))}
                  </span>
                </td>
                {isSuper && (
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <select
                        value={r.assignedAdminId ?? ""}
                        disabled={busyId === r.id}
                        onChange={(e) => assign(r.id, e.target.value)}
                        className="h-8 max-w-[180px] rounded-lg border border-sky-200 bg-white px-2 text-xs text-sky-900 outline-none focus-visible:ring-2 focus-visible:ring-sky-400 disabled:opacity-60"
                      >
                        <option value="">{t("unassigned")}</option>
                        {admins
                          .filter(
                            (a) => a.active || a.uid === r.assignedAdminId,
                          )
                          .map((a) => (
                            <option key={a.uid} value={a.uid}>
                              {a.name}
                            </option>
                          ))}
                      </select>
                      {busyId === r.id && (
                        <Loader2 className="size-4 animate-spin text-sky-500" />
                      )}
                    </div>
                  </td>
                )}
                <td className="px-4 py-3 text-sky-800">{r.type}</td>
                <td className="px-4 py-3 font-mono text-xs text-sky-800">
                  {r.firmwareVersion ?? "—"}
                </td>
                <td className="px-4 py-3 font-mono text-xs text-sky-800"></td>
                <td className="px-4 py-3 text-right">
                  <button
                    type="button"
                    onClick={() => setConfigId(r.id)}
                    className="inline-flex h-8 items-center rounded-lg px-3 text-xs font-semibold text-sky-800 ring-1 ring-sky-300 hover:bg-sky-50"
                  >
                    {t("configure")}
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
