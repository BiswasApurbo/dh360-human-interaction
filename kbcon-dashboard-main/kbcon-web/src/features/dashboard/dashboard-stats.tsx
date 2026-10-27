"use client";

import { useTranslations } from "next-intl";
import { Loader2 } from "lucide-react";
import { StatCard } from "@/components/shared/stat-card";
import { useRole } from "@/features/auth/use-role";
import { useRobots } from "@/features/robots/use-robots";
import { useNow } from "@/hooks/use-now";
import { effectiveStatus } from "@/lib/robot-status";

export function DashboardStats() {
  const t = useTranslations("Dashboard");
  const { role, loading: roleLoading } = useRole();
  const { robots, loading } = useRobots(role);
  const now = useNow();

  if (roleLoading || loading) {
    return (
      <div className="flex justify-center py-10">
        <Loader2 className="size-6 animate-spin text-sky-500" />
      </div>
    );
  }

  const statuses = robots.map((r) => effectiveStatus(r, now));
  const total = robots.length;
  const online = statuses.filter((s) => s === "online").length;
  const offline = statuses.filter((s) => s === "offline").length;
  const errors = statuses.filter((s) => s === "error").length;

  const stats = [
    { key: "totalRobots", value: total, icon: "bot", tone: "sky" },
    { key: "online", value: online, icon: "online", tone: "green" },
    { key: "offline", value: offline, icon: "offline", tone: "slate" },
    { key: "errors", value: errors, icon: "error", tone: "red" },
  ] as const;

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {stats.map((s, i) => (
        <StatCard
          key={s.key}
          index={i}
          title={t(s.key)}
          value={s.value}
          icon={s.icon}
          tone={s.tone}
        />
      ))}
    </div>
  );
}
