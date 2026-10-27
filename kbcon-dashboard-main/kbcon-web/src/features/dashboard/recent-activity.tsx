"use client";

import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import { BookOpen, TriangleAlert, Wifi, WifiOff } from "lucide-react";
import { Badge } from "@/components/ui/badge";

const events = [
  { type: "activityOnline", robot: "R-01", icon: Wifi, unit: "min", n: 4 },
  { type: "activityKb", robot: "R-03", icon: BookOpen, unit: "min", n: 25 },
  {
    type: "activityError",
    robot: "R-07",
    icon: TriangleAlert,
    unit: "h",
    n: 1,
  },
  { type: "activityOffline", robot: "R-05", icon: WifiOff, unit: "h", n: 3 },
  { type: "activityOnline", robot: "R-02", icon: Wifi, unit: "h", n: 5 },
] as const;

export function RecentActivity() {
  const t = useTranslations("Dashboard");

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.6 }}
      className="rounded-3xl border border-sky-200 bg-gradient-to-br from-white via-sky-50 to-sky-100 p-5 shadow-lg shadow-sky-200/60"
    >
      <div className="mb-4 flex items-center justify-between gap-2">
        <h2 className="text-base font-bold text-sky-900">
          {t("activityTitle")}
        </h2>
        <Badge variant="outline" className="border-amber-500 text-amber-600">
          {t("demo")}
        </Badge>
      </div>

      <ul className="space-y-2">
        {events.map((e, i) => {
          const Icon = e.icon;
          return (
            <motion.li
              key={i}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: 0.7 + i * 0.08 }}
              className="flex items-center gap-3 rounded-2xl bg-white/70 px-3 py-2.5 ring-1 ring-sky-100"
            >
              <span className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-sky-100 text-sky-700 ring-1 ring-sky-200">
                <Icon className="size-4" />
              </span>
              <span className="min-w-0 flex-1 truncate text-sm font-medium text-sky-900">
                {t(e.type, { robot: e.robot })}
              </span>
              <span className="shrink-0 text-xs text-sky-700">
                {e.unit === "min"
                  ? t("minutesAgo", { n: e.n })
                  : t("hoursAgo", { n: e.n })}
              </span>
            </motion.li>
          );
        })}
      </ul>
    </motion.div>
  );
}
