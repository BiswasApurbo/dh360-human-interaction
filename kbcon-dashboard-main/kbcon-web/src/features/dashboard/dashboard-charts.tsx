"use client";

import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { ModeSplit } from "./mode-split";

const weekData = [
  { day: "day0", online: 7, offline: 5 },
  { day: "day1", online: 8, offline: 4 },
  { day: "day2", online: 9, offline: 3 },
  { day: "day3", online: 6, offline: 6 },
  { day: "day4", online: 8, offline: 4 },
  { day: "day5", online: 9, offline: 2 },
  { day: "day6", online: 8, offline: 3 },
] as const;

export function DashboardCharts() {
  const t = useTranslations("Dashboard");

  const week = weekData.map((d) => ({
    day: t(d.day),
    online: d.online,
    offline: d.offline,
  }));

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="rounded-3xl border border-sky-200 bg-gradient-to-br from-white via-sky-50 to-sky-100 p-5 shadow-lg shadow-sky-200/60 lg:col-span-2"
      >
        <div className="mb-4 flex flex-wrap items-start justify-between gap-2">
          <div>
            <h2 className="text-base font-bold text-sky-900">
              {t("weekTitle")}
            </h2>
            <p className="text-xs text-sky-700">{t("weekSubtitle")}</p>
          </div>
          <Badge variant="outline" className="border-amber-500 text-amber-600">
            {t("demo")}
          </Badge>
        </div>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={week}
              margin={{ top: 5, right: 8, left: -20, bottom: 0 }}
            >
              <defs>
                <linearGradient id="gOnline" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#0ea5e9" stopOpacity={0.5} />
                  <stop offset="100%" stopColor="#0ea5e9" stopOpacity={0.05} />
                </linearGradient>
                <linearGradient id="gOffline" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#94a3b8" stopOpacity={0.45} />
                  <stop offset="100%" stopColor="#94a3b8" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#bae6fd" />
              <XAxis dataKey="day" tick={{ fill: "#0369a1", fontSize: 12 }} />
              <YAxis
                allowDecimals={false}
                tick={{ fill: "#0369a1", fontSize: 12 }}
              />
              <Tooltip />
              <Area
                type="monotone"
                dataKey="online"
                name={t("online")}
                stroke="#0284c7"
                strokeWidth={2.5}
                fill="url(#gOnline)"
              />
              <Area
                type="monotone"
                dataKey="offline"
                name={t("offline")}
                stroke="#64748b"
                strokeWidth={2.5}
                fill="url(#gOffline)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      <ModeSplit />
    </div>
  );
}
