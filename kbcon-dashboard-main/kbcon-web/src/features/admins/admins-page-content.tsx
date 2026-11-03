"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { AnimatePresence, motion } from "motion/react";
import {
  AlertCircle,
  CheckCircle2,
  KeyRound,
  Loader2,
  LogOut,
  Plus,
  RotateCw,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Trash2,
} from "lucide-react";
import { useAuth } from "@/features/auth/use-auth";
import { useRole } from "@/features/auth/use-role";
import { useAdmins, type AdminRow } from "./use-admins";
import { CreateAdminForm } from "./create-admin-form";

export function AdminsPageContent() {
  const t = useTranslations("Admins");
  const { user } = useAuth();
  const { role, loading: roleLoading } = useRole();
  const isSuper = role === "superAdmin";
  const { admins, loading, error, refetch } = useAdmins(isSuper);

  const [showForm, setShowForm] = useState(false);
  const [busyUid, setBusyUid] = useState<string | null>(null);
  const [deleteConfirmUser, setDeleteConfirmUser] = useState<AdminRow | null>(null);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);

  // Toggle active status
  async function toggleActive(uid: string, nextActive: boolean) {
    if (!user) return;
    setBusyUid(uid);
    setMessage(null);
    try {
      const token = await user.getIdToken();
      const res = await fetch(`/api/admin/users/${uid}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ active: nextActive }),
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.message || t("updateFailed"));
      }
      setMessage({ ok: true, text: t("updated") });
      await refetch();
    } catch (err: unknown) {
      setMessage({ ok: false, text: err instanceof Error ? err.message : t("updateFailed") });
    } finally {
      setBusyUid(null);
    }
  }

  // Trigger password reset email
  async function triggerPasswordReset(uid: string, email: string) {
    if (!user) return;
    setBusyUid(uid);
    setMessage(null);
    try {
      const token = await user.getIdToken();
      const res = await fetch(`/api/admin/users/${uid}/reset-password`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.message || "Failed to trigger password reset.");
      }
      setMessage({ ok: true, text: `Password reset initiated for ${email}.` });
    } catch (err: unknown) {
      setMessage({ ok: false, text: err instanceof Error ? err.message : "Failed to send reset email." });
    } finally {
      setBusyUid(null);
    }
  }

  // Revoke active sessions
  async function revokeSessions(uid: string) {
    if (!user) return;
    setBusyUid(uid);
    setMessage(null);
    try {
      const token = await user.getIdToken();
      const res = await fetch(`/api/admin/users/${uid}/revoke-sessions`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.message || "Failed to revoke sessions.");
      }
      setMessage({ ok: true, text: `All active sessions revoked for user.` });
    } catch (err: unknown) {
      setMessage({ ok: false, text: err instanceof Error ? err.message : "Failed to revoke sessions." });
    } finally {
      setBusyUid(null);
    }
  }

  // Delete user
  async function confirmDelete() {
    if (!user || !deleteConfirmUser) return;
    const uid = deleteConfirmUser.uid;
    setBusyUid(uid);
    setMessage(null);
    try {
      const token = await user.getIdToken();
      const res = await fetch(`/api/admin/users/${uid}`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.message || "Failed to delete administrator.");
      }
      setMessage({ ok: true, text: `Account ${deleteConfirmUser.email} deleted.` });
      setDeleteConfirmUser(null);
      await refetch();
    } catch (err: unknown) {
      setMessage({ ok: false, text: err instanceof Error ? err.message : "Failed to delete administrator." });
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
                void refetch();
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {message && (
        <div
          className={`flex items-center gap-2 rounded-xl px-3 py-2.5 text-sm ring-1 ${
            message.ok
              ? "bg-emerald-50 text-emerald-700 ring-emerald-200"
              : "bg-red-50 text-red-600 ring-red-200"
          }`}
        >
          {message.ok ? (
            <CheckCircle2 className="size-4 shrink-0 text-emerald-600" />
          ) : (
            <AlertCircle className="size-4 shrink-0 text-red-600" />
          )}
          <span>{message.text}</span>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deleteConfirmUser && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm"
        >
          <div className="w-full max-w-md rounded-2xl bg-white p-6 shadow-2xl border border-red-100">
            <div className="flex items-center gap-3 text-red-600 mb-3">
              <ShieldAlert className="size-6 shrink-0" />
              <h3 className="text-base font-bold text-slate-900">
                Confirm Account Removal
              </h3>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Are you sure you want to permanently remove administrator{" "}
              <strong className="text-slate-900">{deleteConfirmUser.name}</strong> (
              {deleteConfirmUser.email})? Their Firebase credentials and dashboard
              access will be permanently revoked.
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setDeleteConfirmUser(null)}
                className="rounded-xl px-4 py-2 text-xs font-semibold text-slate-700 ring-1 ring-slate-300 hover:bg-slate-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={confirmDelete}
                disabled={busyUid === deleteConfirmUser.uid}
                className="flex items-center gap-1.5 rounded-xl bg-red-600 px-4 py-2 text-xs font-bold text-white shadow-md hover:bg-red-700 disabled:opacity-50"
              >
                {busyUid === deleteConfirmUser.uid && (
                  <Loader2 className="size-3.5 animate-spin" />
                )}
                <span>Remove Account</span>
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="overflow-x-auto rounded-2xl border border-sky-200 bg-white/90 shadow-sm">
        <table className="w-full min-w-[680px] text-left text-sm">
          <thead className="bg-sky-100/70 text-sky-900">
            <tr>
              <th className="px-4 py-3 font-semibold">{t("name")}</th>
              <th className="px-4 py-3 font-semibold">{t("email")}</th>
              <th className="px-4 py-3 font-semibold">Role</th>
              <th className="px-4 py-3 font-semibold">Assigned Robots</th>
              <th className="px-4 py-3 font-semibold">{t("status")}</th>
              <th className="px-4 py-3 font-semibold">Permissions</th>
              <th className="px-4 py-3 text-right font-semibold">
                {t("actions")}
              </th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center">
                  <Loader2 className="mx-auto size-6 animate-spin text-sky-500" />
                </td>
              </tr>
            )}
            {!loading && error && (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center">
                  <div className="mx-auto flex max-w-md flex-col items-center justify-center gap-2">
                    <div className="flex items-center gap-2 text-sm font-semibold text-red-600">
                      <AlertCircle className="size-5 shrink-0" />
                      <span>{t("loadFailed")}</span>
                    </div>
                    <p className="text-xs text-red-500 font-mono">{error}</p>
                    <button
                      type="button"
                      onClick={() => void refetch()}
                      className="mt-2 inline-flex items-center gap-1.5 rounded-xl bg-sky-500 px-3.5 py-1.5 text-xs font-semibold text-white shadow hover:bg-sky-600 transition"
                    >
                      <RotateCw className="size-3.5" />
                      <span>Retry</span>
                    </button>
                  </div>
                </td>
              </tr>
            )}
            {!loading && !error && admins.length === 0 && (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-sky-700">
                  {t("empty")}
                </td>
              </tr>
            )}
            {admins.map((a) => {
              const isAccountSuper = a.role === "superAdmin";
              return (
                <motion.tr
                  key={a.uid}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="border-t border-sky-100 hover:bg-sky-50/40 transition-colors"
                >
                  <td className="px-4 py-3 font-medium text-sky-950">
                    {a.name}
                  </td>
                  <td className="px-4 py-3 text-sky-800 font-mono text-xs">
                    {a.email}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-bold ${
                        isAccountSuper
                          ? "bg-purple-100 text-purple-800 ring-1 ring-purple-300"
                          : "bg-sky-100 text-sky-800"
                      }`}
                    >
                      {isAccountSuper ? (
                        <ShieldCheck className="size-3 text-purple-600" />
                      ) : (
                        <Shield className="size-3 text-sky-600" />
                      )}
                      <span>{isAccountSuper ? "Super Admin" : "Admin"}</span>
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    {isAccountSuper ? (
                      <span className="text-xs text-purple-700 font-semibold">
                        All Robots
                      </span>
                    ) : a.assignedRobots && a.assignedRobots.length > 0 ? (
                      <div className="flex flex-wrap gap-1 max-w-xs">
                        {a.assignedRobots.map((rId) => (
                          <span
                            key={rId}
                            className="rounded bg-sky-100 px-2 py-0.5 font-mono text-[11px] font-semibold text-sky-900 border border-sky-300"
                          >
                            {rId}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="text-xs text-slate-400 italic">None</span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                        a.active
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-slate-200 text-slate-600"
                      }`}
                    >
                      {a.active ? t("active") : t("disabled")}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    {isAccountSuper ? (
                      <span className="text-xs text-purple-700 font-semibold">
                        All Permitted
                      </span>
                    ) : (
                      <div className="flex flex-wrap gap-1 max-w-xs">
                        {Object.entries(a.permissions || {})
                          .filter(([, v]) => v)
                          .map(([k]) => (
                            <span
                              key={k}
                              className="rounded bg-sky-50 px-1.5 py-0.5 text-[10px] font-medium text-sky-800 border border-sky-200"
                            >
                              {k}
                            </span>
                          ))}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex items-center justify-end gap-1.5">
                      {!isAccountSuper && (
                        <>
                          <button
                            type="button"
                            disabled={busyUid === a.uid}
                            onClick={() => toggleActive(a.uid, !a.active)}
                            title={a.active ? t("disable") : t("enable")}
                            className="inline-flex h-8 items-center gap-1 rounded-lg px-2 text-xs font-semibold text-sky-800 ring-1 ring-sky-300 hover:bg-sky-50 disabled:opacity-60"
                          >
                            {busyUid === a.uid && (
                              <Loader2 className="size-3 animate-spin" />
                            )}
                            {a.active ? t("disable") : t("enable")}
                          </button>

                          <button
                            type="button"
                            disabled={busyUid === a.uid}
                            onClick={() => triggerPasswordReset(a.uid, a.email)}
                            title="Send Password Reset Email"
                            className="inline-flex h-8 items-center gap-1 rounded-lg px-2 text-xs font-semibold text-slate-700 ring-1 ring-slate-300 hover:bg-slate-50 disabled:opacity-60"
                          >
                            <KeyRound className="size-3.5" />
                          </button>

                          <button
                            type="button"
                            disabled={busyUid === a.uid}
                            onClick={() => revokeSessions(a.uid)}
                            title="Revoke Active Sessions"
                            className="inline-flex h-8 items-center gap-1 rounded-lg px-2 text-xs font-semibold text-amber-800 ring-1 ring-amber-300 hover:bg-amber-50 disabled:opacity-60"
                          >
                            <LogOut className="size-3.5" />
                          </button>

                          <button
                            type="button"
                            disabled={busyUid === a.uid}
                            onClick={() => setDeleteConfirmUser(a)}
                            title="Delete Admin Account"
                            className="inline-flex h-8 items-center gap-1 rounded-lg px-2 text-xs font-semibold text-red-700 ring-1 ring-red-300 hover:bg-red-50 disabled:opacity-60"
                          >
                            <Trash2 className="size-3.5" />
                          </button>
                        </>
                      )}
                      {isAccountSuper && (
                        <span className="text-xs text-slate-400 italic pr-2">
                          Protected
                        </span>
                      )}
                    </div>
                  </td>
                </motion.tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
