"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import { doc, onSnapshot } from "firebase/firestore";
import { ArrowLeft, CheckCircle2, Loader2, RefreshCw, ShieldAlert } from "lucide-react";
import { auth, db } from "@/lib/firebase";
import { Link } from "@/i18n/navigation";
import { useAuth } from "@/features/auth/use-auth";
import { useRole } from "@/features/auth/use-role";
import { useNow } from "@/hooks/use-now";
import { effectiveStatus } from "@/lib/robot-status";
import { useRobots } from "./use-robots";
import { RobotConfigForm } from "./robot-config-form";

const statusStyle = {
  online: "bg-emerald-100 text-emerald-700",
  offline: "bg-slate-200 text-slate-600",
  connecting: "bg-amber-100 text-amber-700",
  error: "bg-red-100 text-red-700",
} as const;

const cardClass =
  "rounded-3xl border border-sky-200 bg-gradient-to-br from-white via-sky-50 to-sky-100 p-5 shadow-lg shadow-sky-200/60";

type Tel = {
  battery: number;
  temperature: number;
  activity: string;
  errorCode: string | null;
} | null;

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <p className="text-xs text-sky-700">{label}</p>
      <div className="mt-0.5 text-sm font-semibold text-sky-900">
        {children}
      </div>
    </div>
  );
}

export function RobotDetail({ robotId }: { robotId: string }) {
  const t = useTranslations("Robots");
  const { user } = useAuth();
  const { role, loading: roleLoading } = useRole();
  const { robots, loading } = useRobots(role);
  const now = useNow();
  const [tel, setTel] = useState<Tel>(null);
  const [kbName, setKbName] = useState<string | null>(null);
  const [reconnecting, setReconnecting] = useState(false);
  const [reconnectResult, setReconnectResult] = useState<{
    ok: boolean;
    text: string;
  } | null>(null);

  const robot = robots.find((r) => r.id === robotId) ?? null;
  const isSuper = role === "superAdmin";
  const canReconnect = isSuper || (user && robot?.assignedAdminId === user.uid);

  async function handleReconnect() {
    setReconnecting(true);
    setReconnectResult(null);
    try {
      const token = await auth.currentUser?.getIdToken();
      const res = await fetch(`/api/admin/robots/${robotId}/reconnect`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.success) {
        setReconnectResult({
          ok: true,
          text:
            data.message ||
            `Robot ${robotId} successfully reconnected and verified online.`,
        });
      } else {
        setReconnectResult({
          ok: false,
          text:
            data.message ||
            `Reconnection probe failed for ${robotId}. Device is unreachable.`,
        });
      }
    } catch (err: unknown) {
      setReconnectResult({
        ok: false,
        text: err instanceof Error ? err.message : "Failed to reconnect to robot.",
      });
    } finally {
      setReconnecting(false);
    }
  }

  // সর্বশেষ Telemetry ও Knowledge Base-এর নাম
  useEffect(() => {
    if (!robot) return;
    const unsub = onSnapshot(
      doc(db, "robots", robotId),
      (snap) => {
        const x = snap.data()?.telemetry as Record<string, unknown> | undefined;
        setTel(
          x
            ? {
                battery: Number(x.battery ?? 0),
                temperature: Number(x.temperature ?? 0),
                activity: String(x.activity ?? "idle"),
                errorCode: typeof x.errorCode === "string" ? x.errorCode : null,
              }
            : null,
        );
      },
      () => setTel(null),
    );
    return () => unsub();
  }, [robot, robotId]);

  const kbId = (robot?.knowledgeBaseId ?? null) as string | null;
  useEffect(() => {
    if (!kbId) return;
    const unsub = onSnapshot(
      doc(db, "knowledgeBases", kbId),
      (snap) =>
        setKbName(snap.exists() ? String(snap.data()?.name ?? "") : null),
      () => setKbName(null),
    );
    return () => unsub();
  }, [kbId]);

  if (roleLoading || loading) {
    return (
      <div className="flex justify-center py-16">
        <Loader2 className="size-7 animate-spin text-sky-500" />
      </div>
    );
  }

  const back = (
    <Link
      href="/robots"
      className="inline-flex h-9 items-center gap-1.5 rounded-lg px-3 text-xs font-semibold text-sky-800 ring-1 ring-sky-300 hover:bg-sky-50"
    >
      <ArrowLeft className="size-4" />
      {t("backToList")}
    </Link>
  );

  if (!robot) {
    return (
      <div className="space-y-4">
        {back}
        <div className="flex items-center gap-2 rounded-2xl bg-red-50 p-4 text-sm text-red-700 ring-1 ring-red-200">
          <ShieldAlert className="size-5 shrink-0" />
          {t("notFound")}
        </div>
      </div>
    );
  }

  const status = effectiveStatus(robot, now);
  const seen = robot.lastSeen
    ? new Date(robot.lastSeen).toLocaleString()
    : t("never");

  return (
    <div className="space-y-5">
      {back}

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-xl font-bold text-sky-900 sm:text-2xl">
            {robot.name}
          </h1>
          <span className="font-mono text-sm text-sky-700">{robot.id}</span>
          <span
            className={`rounded-full px-2.5 py-1 text-xs font-semibold ${statusStyle[status]}`}
          >
            {t(status)}
          </span>
        </div>

        {canReconnect && (
          <button
            type="button"
            disabled={reconnecting}
            onClick={handleReconnect}
            className="inline-flex h-9 items-center gap-2 rounded-xl border border-sky-300 bg-white px-3.5 text-xs font-semibold text-sky-800 shadow-sm hover:bg-sky-50 disabled:opacity-60"
          >
            {reconnecting ? (
              <>
                <Loader2 className="size-3.5 animate-spin text-sky-600" />
                <span>Reconnecting...</span>
              </>
            ) : (
              <>
                <RefreshCw className="size-3.5 text-sky-600" />
                <span>Reconnect</span>
              </>
            )}
          </button>
        )}
      </div>

      {reconnectResult && (
        <p
          className={`rounded-xl px-4 py-2.5 text-xs font-medium ring-1 flex items-center gap-2 ${
            reconnectResult.ok
              ? "bg-emerald-50 text-emerald-800 ring-emerald-200"
              : "bg-red-50 text-red-700 ring-red-200"
          }`}
        >
          {reconnectResult.ok ? (
            <CheckCircle2 className="size-4 shrink-0 text-emerald-600" />
          ) : (
            <ShieldAlert className="size-4 shrink-0 text-red-500" />
          )}
          <span>{reconnectResult.text}</span>
        </p>
      )}

      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className={cardClass}
      >
        <h2 className="mb-4 text-base font-bold text-sky-900">{t("info")}</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Field label={t("type")}>{robot.type}</Field>
          <Field label={t("mode")}>{t(robot.mode)}</Field>
          <Field label={t("assignedTo")}>
            {robot.assignedAdminName ?? t("unassigned")}
          </Field>
          <Field label={t("firmware")}>{robot.firmwareVersion ?? "—"}</Field>
          <Field label={t("lastSeen")}>{seen}</Field>
          <Field label={t("knowledgeBase")}>
            {kbId ? (kbName ?? "…") : t("notLinked")}
          </Field>
        </div>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
        className={cardClass}
      >
        <h2 className="mb-4 text-base font-bold text-sky-900">
          {t("latestTelemetry")}
        </h2>
        {tel ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Field label={t("battery")}>{tel.battery}%</Field>
            <Field label={t("temperature")}>{tel.temperature}°C</Field>
            <Field label={t("activity")}>
              {t(`act_${tel.activity}` as "act_idle")}
            </Field>
            <Field label={t("errorCode")}>
              {tel.errorCode ? (
                <span className="font-mono text-red-700">{tel.errorCode}</span>
              ) : (
                t("none")
              )}
            </Field>
          </div>
        ) : (
          <p className="text-sm text-sky-700">{t("noTelemetry")}</p>
        )}
      </motion.section>

      <RobotConfigForm key={robot.id} robot={robot} onClose={() => {}} />
    </div>
  );
}
