"use client";

import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { useRole } from "@/features/auth/use-role";
import { useRobots } from "@/features/robots/use-robots";

export function ModeSplit() {
  const t = useTranslations("Dashboard");
  const { role } = useRole();
  const { robots } = useRobots(role);

  const modes = [
    {
      name: t("education"),
      value: robots.filter((r) => r.mode === "education").length,
      color: "#0ea5e9",
    },
    {
      name: t("healthcare"),
      value: robots.filter((r) => r.mode === "healthcare").length,
      color: "#7dd3fc",
    },
  ];
  const total = modes.reduce((sum, m) => sum + m.value, 0);

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
      className="rounded-3xl border border-sky-200 bg-gradient-to-br from-white via-sky-50 to-sky-100 p-5 shadow-lg shadow-sky-200/60"
    >
      <div className="mb-4">
        <h2 className="text-base font-bold text-sky-900">{t("modeTitle")}</h2>
        <p className="text-xs text-sky-700">{t("modeSubtitle")}</p>
      </div>
      <div className="relative h-52 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={
                total === 0
                  ? [{ name: "-", value: 1, color: "#e0f2fe" }]
                  : modes
              }
              dataKey="value"
              nameKey="name"
              innerRadius={55}
              outerRadius={80}
              paddingAngle={3}
              stroke="none"
            >
              {(total === 0
                ? [{ name: "-", value: 1, color: "#e0f2fe" }]
                : modes
              ).map((m) => (
                <Cell key={m.name} fill={m.color} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <span className="text-3xl font-extrabold text-sky-900">{total}</span>
        </div>
      </div>
      <div className="mt-3 flex flex-wrap justify-center gap-4 text-xs text-sky-800">
        {modes.map((m) => (
          <span key={m.name} className="inline-flex items-center gap-1.5">
            <span
              className="size-2.5 rounded-full"
              style={{ backgroundColor: m.color }}
            />
            {m.name}: {m.value}
          </span>
        ))}
      </div>
    </motion.div>
  );
}
