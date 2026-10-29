"use client";

import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import { Activity, Radio } from "lucide-react";
import { ModeSplit } from "./mode-split";

export function DashboardCharts() {
  const t = useTranslations("Dashboard");

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="rounded-3xl border border-sky-200 bg-gradient-to-br from-white via-sky-50 to-sky-100 p-5 shadow-lg shadow-sky-200/60 lg:col-span-2 flex flex-col justify-between"
      >
        <div className="mb-4">
          <h2 className="text-base font-bold text-sky-900">
            {t("weekTitle")}
          </h2>
          <p className="text-xs text-sky-700">{t("weekSubtitle")}</p>
        </div>

        {/* Real Data State: No fake historical series */}
        <div className="flex h-64 w-full flex-col items-center justify-center rounded-2xl border border-dashed border-sky-200 bg-white/60 p-6 text-center">
          <div className="flex size-12 items-center justify-center rounded-2xl bg-sky-100 text-sky-600 shadow-sm">
            <Radio className="size-6 animate-pulse" />
          </div>
          <h3 className="mt-3 text-sm font-bold text-sky-950">
            No Historical Telemetry Snapshots
          </h3>
          <p className="mt-1 max-w-sm text-xs text-sky-800/80 leading-relaxed">
            Connectivity trends are calculated from live edge robot heartbeats. Historical trends will populate automatically once active robots stream continuous operational telemetry.
          </p>
        </div>
      </motion.div>

      <ModeSplit />
    </div>
  );
}
