"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import { Activity, Clock, ShieldCheck, User, Users } from "lucide-react";
import { useAuth } from "@/features/auth/use-auth";

type LatestInteraction = {
  session_id: string;
  display_name: string | null;
  started_at_iso: string;
  duration_seconds: number;
  final_state: string;
  identity_was_known: boolean;
};

type SessionAnalytics = {
  total_sessions: number;
  identified_sessions_count: number;
  anonymous_sessions_count: number;
};

export function RecentActivity() {
  const t = useTranslations("Dashboard");
  const { user } = useAuth();

  const [latest, setLatest] = useState<LatestInteraction | null>(null);
  const [analytics, setAnalytics] = useState<SessionAnalytics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadActivity() {
      try {
        const headers: Record<string, string> = {};
        if (user) {
          const token = await user.getIdToken();
          headers["Authorization"] = `Bearer ${token}`;
        }

        const [latestRes, analyticsRes] = await Promise.all([
          fetch("/api/dh360/interaction/summary/latest", { headers }).catch(() => null),
          fetch("/api/dh360/sessions/analytics", { headers }).catch(() => null),
        ]);

        if (latestRes && latestRes.ok) {
          const data = await latestRes.json();
          if (data && data.session_id) {
            setLatest(data);
          }
        }

        if (analyticsRes && analyticsRes.ok) {
          const aData = await analyticsRes.json();
          setAnalytics(aData);
        }
      } catch {
        // Fallback
      } finally {
        setLoading(false);
      }
    }

    void loadActivity();
  }, [user]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.6 }}
      className="rounded-3xl border border-sky-200 bg-gradient-to-br from-white via-sky-50 to-sky-100 p-5 shadow-lg shadow-sky-200/60"
    >
      <div className="mb-4 flex items-center justify-between gap-2">
        <div>
          <h2 className="text-base font-bold text-sky-900">
            {t("activityTitle")}
          </h2>
          <p className="text-xs text-sky-700">Real-time edge interaction events and sessions</p>
        </div>
      </div>

      {latest ? (
        <div className="space-y-3">
          {/* Latest Session Card */}
          <div className="flex items-center gap-3 rounded-2xl bg-white/80 p-3.5 ring-1 ring-sky-200 shadow-sm">
            <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-sky-100 text-sky-700 ring-1 ring-sky-200">
              {latest.identity_was_known ? (
                <ShieldCheck className="size-5 text-emerald-600" />
              ) : (
                <User className="size-5 text-sky-600" />
              )}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="truncate text-sm font-bold text-sky-950">
                  {latest.identity_was_known && latest.display_name
                    ? `Verified session with ${latest.display_name}`
                    : "Anonymous visitor interaction"}
                </span>
                <span className="rounded-lg bg-sky-100 px-2 py-0.5 text-[10px] font-extrabold text-sky-800">
                  {latest.final_state}
                </span>
              </div>
              <p className="text-xs text-sky-700 font-mono truncate">
                Session: {latest.session_id}
              </p>
            </div>
            <div className="text-right shrink-0">
              <span className="inline-flex items-center gap-1 text-xs font-semibold text-sky-800">
                <Clock className="size-3 text-sky-500" />
                {latest.duration_seconds.toFixed(1)}s
              </span>
            </div>
          </div>

          {/* Aggregate Session Metrics if available */}
          {analytics && analytics.total_sessions > 0 && (
            <div className="grid grid-cols-2 gap-3 pt-1 sm:grid-cols-3">
              <div className="rounded-2xl bg-white/60 p-3 ring-1 ring-sky-100">
                <span className="text-[11px] font-semibold text-sky-700">Total Sessions</span>
                <p className="text-base font-bold text-sky-950">{analytics.total_sessions}</p>
              </div>
              <div className="rounded-2xl bg-white/60 p-3 ring-1 ring-sky-100">
                <span className="text-[11px] font-semibold text-emerald-700">Identified</span>
                <p className="text-base font-bold text-emerald-950">{analytics.identified_sessions_count}</p>
              </div>
              <div className="rounded-2xl bg-white/60 p-3 ring-1 ring-sky-100 col-span-2 sm:col-span-1">
                <span className="text-[11px] font-semibold text-slate-700">Anonymous</span>
                <p className="text-base font-bold text-slate-900">{analytics.anonymous_sessions_count}</p>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-sky-200 bg-white/60 py-8 text-center">
          <Activity className="size-8 text-sky-400" />
          <p className="mt-2 text-xs font-semibold text-sky-900">
            No recent interaction sessions recorded
          </p>
          <p className="mt-0.5 text-[11px] text-sky-700 max-w-sm">
            Live interaction sessions will be shown here as users approach the robot.
          </p>
        </div>
      )}
    </motion.div>
  );
}
