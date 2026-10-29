"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { GraduationCap, Hospital, Settings2 } from "lucide-react";
import { useAuth } from "@/features/auth/use-auth";
import { useRole } from "@/features/auth/use-role";
import { useRobots } from "@/features/robots/use-robots";

export function ModeSplit() {
  const t = useTranslations("Dashboard");
  const { user } = useAuth();
  const { role } = useRole();
  const { robots } = useRobots(role);

  const [institution, setInstitution] = useState<{
    type: string;
    name: string;
  } | null>(null);

  useEffect(() => {
    async function loadInstitution() {
      try {
        const headers: Record<string, string> = {};
        if (user) {
          const token = await user.getIdToken();
          headers["Authorization"] = `Bearer ${token}`;
        }
        const res = await fetch("/api/dh360/settings/institution", { headers });
        if (res.ok) {
          const data = await res.json();
          setInstitution({
            type: data.institution_type ?? "university",
            name: data.institution_name ?? "",
          });
        }
      } catch {
        // Ignored fallback
      }
    }
    void loadInstitution();
  }, [user]);

  const educationCount = robots.filter((r) => r.mode === "education").length;
  const healthcareCount = robots.filter((r) => r.mode === "healthcare").length;
  const total = robots.length;

  const modes = [
    {
      name: t("education"),
      value: educationCount,
      color: "#0ea5e9",
    },
    {
      name: t("healthcare"),
      value: healthcareCount,
      color: "#10b981",
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
      className="rounded-3xl border border-sky-200 bg-gradient-to-br from-white via-sky-50 to-sky-100 p-5 shadow-lg shadow-sky-200/60 flex flex-col justify-between"
    >
      <div className="mb-2">
        <h2 className="text-base font-bold text-sky-900">{t("modeTitle")}</h2>
        <p className="text-xs text-sky-700">{t("modeSubtitle")}</p>
      </div>

      {total > 0 ? (
        <>
          <div className="relative h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={modes}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={50}
                  outerRadius={75}
                  paddingAngle={4}
                  stroke="none"
                >
                  {modes.map((m) => (
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
          <div className="mt-2 flex flex-wrap justify-center gap-4 text-xs text-sky-800">
            {modes.map((m) => (
              <span key={m.name} className="inline-flex items-center gap-1.5 font-medium">
                <span
                  className="size-2.5 rounded-full"
                  style={{ backgroundColor: m.color }}
                />
                {m.name}: {m.value}
              </span>
            ))}
          </div>
        </>
      ) : (
        <div className="my-auto flex flex-col items-center justify-center rounded-2xl border border-dashed border-sky-200 bg-white/60 p-4 text-center">
          <Settings2 className="size-8 text-sky-400" />
          <p className="mt-2 text-xs font-semibold text-sky-900">
            No robots registered in fleet yet
          </p>
          {institution && (
            <div className="mt-2 inline-flex items-center gap-1.5 rounded-xl bg-sky-100 px-3 py-1 text-[11px] font-bold text-sky-800">
              {institution.type === "clinic" ? (
                <Hospital className="size-3.5 text-emerald-600" />
              ) : (
                <GraduationCap className="size-3.5 text-blue-600" />
              )}
              <span>
                Active Config: {institution.type === "clinic" ? "Healthcare / Clinic" : "University / Academic"}
                {institution.name ? ` (${institution.name})` : ""}
              </span>
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
}
