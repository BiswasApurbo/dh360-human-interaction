"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { AlertTriangle, Cpu, Loader2, Trash2, X } from "lucide-react";
import { auth } from "@/lib/firebase";
import type { RobotRow } from "./use-robots";

interface DeleteRobotModalProps {
  robot: RobotRow | null;
  isOpen: boolean;
  onClose: () => void;
  onDeleted: () => void;
}

export function DeleteRobotModal({
  robot,
  isOpen,
  onClose,
  onDeleted,
}: DeleteRobotModalProps) {
  const t = useTranslations("Robots");
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen || !robot) return null;

  async function handleDelete() {
    if (!robot) return;
    setIsDeleting(true);
    setError(null);

    try {
      const token = await auth.currentUser?.getIdToken();
      if (!token) {
        setError("Authentication token required. Please sign in again.");
        setIsDeleting(false);
        return;
      }

      const res = await fetch("/api/admin/robots", {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ robotId: robot.id }),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.message || data.error || t("deleteRobotFailed"));
      }

      onDeleted();
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : t("deleteRobotFailed"));
    } finally {
      setIsDeleting(false);
    }
  }

  const robotTypeLabel =
    robot.mode === "healthcare" || robot.type?.toLowerCase() === "health"
      ? t("typeHealth")
      : t("typeEducation");

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-200"
    >
      <div className="relative w-full max-w-md rounded-2xl bg-white shadow-2xl border border-red-200 overflow-hidden flex flex-col p-6 space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-xl bg-red-100 text-red-700">
              <AlertTriangle className="size-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900">
                {t("deleteConfirmTitle")}
              </h3>
              <p className="text-xs text-slate-500 font-mono">
                {robot.name} • {robot.id}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={isDeleting}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700 transition-colors"
          >
            <X className="size-5" />
          </button>
        </div>

        {/* Warning text */}
        <p className="text-xs text-slate-600 leading-relaxed">
          {t("deleteConfirmDesc", { name: robot.name })}
        </p>

        {/* Confirmation Details Card */}
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-3.5 space-y-2 text-xs">
          <div className="flex justify-between items-center py-1 border-b border-slate-200/60">
            <span className="text-slate-500 font-medium">{t("name")}:</span>
            <span className="font-semibold text-slate-900">{robot.name}</span>
          </div>
          <div className="flex justify-between items-center py-1 border-b border-slate-200/60">
            <span className="text-slate-500 font-medium">{t("robotId")}:</span>
            <span className="font-mono font-semibold text-slate-900">{robot.id}</span>
          </div>
          <div className="flex justify-between items-center py-1 border-b border-slate-200/60">
            <span className="text-slate-500 font-medium">{t("robotType")}:</span>
            <span className="font-semibold text-slate-900">{robotTypeLabel}</span>
          </div>
          <div className="flex justify-between items-center py-1">
            <span className="text-slate-500 font-medium">Associated Device:</span>
            {robot.deviceId ? (
              <span className="inline-flex items-center gap-1 font-mono text-[11px] text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                <Cpu className="size-3 text-emerald-600" />
                {robot.deviceId}
              </span>
            ) : (
              <span className="text-slate-400 font-normal">None (Unpaired)</span>
            )}
          </div>
        </div>

        {/* Error notice */}
        {error && (
          <div className="rounded-xl bg-red-50 p-3 text-xs text-red-700 ring-1 ring-red-200">
            {error}
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex items-center justify-end gap-2 pt-2">
          <button
            type="button"
            onClick={onClose}
            disabled={isDeleting}
            className="rounded-xl px-4 py-2 text-xs font-semibold text-slate-700 ring-1 ring-slate-300 hover:bg-slate-100 transition-colors disabled:opacity-50"
          >
            {t("cancel")}
          </button>
          <button
            type="button"
            onClick={handleDelete}
            disabled={isDeleting}
            className="inline-flex items-center gap-1.5 rounded-xl bg-red-600 px-4 py-2 text-xs font-bold text-white shadow-md shadow-red-200 hover:bg-red-700 transition-colors disabled:opacity-50"
          >
            {isDeleting ? (
              <>
                <Loader2 className="size-3.5 animate-spin" />
                <span>{t("deletingRobot")}</span>
              </>
            ) : (
              <>
                <Trash2 className="size-3.5" />
                <span>{t("deleteRobot")}</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
