"use client";

import { useEffect, useMemo, useState } from "react";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Battery, Loader2, Thermometer } from "lucide-react";
import { useRole } from "@/features/auth/use-role";
import { useTelemetryHistory, useTelemetryRobots } from "./use-telemetry";
import { offlineAfterMs } from "@/lib/robot-status";

type Status = "online" | "offline" | "error";

const statusStyle: Record<Status, string> = {
  online: "bg-emerald-100 text-emerald-700",
  offline: "bg-slate-200 text-slate-600",
  error: "bg-red-100 text-red-700",
};

const cardClass =
  "rounded-3xl border border-sky-200 bg-gradient-to-br from-white via-sky-50 to-sky-100 p-5 shadow-lg shadow-sky-200/60";

export function TelemetryContent({ locale }: { locale: string }) {
  const t = useTranslations("Telemetry");
  const { role, loading: roleLoading } = useRole();
  const { robots, loading, error } = useTelemetryRobots(role);

  const [statusFilter, setStatusFilter] = useState<"all" | Status>("all");
  const [range, setRange] = useState<24 | 168 | "custom">(24);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [pickedId, setPickedId] = useState<string | null>(null);

  // প্রতি ১৫ সেকেন্ডে সময় হালনাগাদ, যাতে চুপ থাকা Robot Offline দেখায়
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 15000);
    return () => clearInterval(id);
  }, []);

  const rows = useMemo(
    () =>
      robots.map((r) => {
        const seen = r.lastSeen ? Date.parse(r.lastSeen) : 0;
        const stale = !seen || now - seen > offlineAfterMs(r.config);
        const status: Status = stale
          ? "offline"
          : r.errorCode
            ? "error"
            : "online";
        return { ...r, status, seen };
      }),
    [robots, now],
  );

  const filtered = rows.filter(
    (r) => statusFilter === "all" || r.status === statusFilter,
  );
  const selectedId =
    pickedId && rows.some((r) => r.id === pickedId)
      ? pickedId
      : (rows[0]?.id ?? null);
  const selected = rows.find((r) => r.id === selectedId) ?? null;

  // কাস্টম তারিখ ঠিক না থাকলে ২৪ ঘণ্টার হিসাবে ফিরে যায়
  const { fromIso, toIso, dateError } = useMemo(() => {
    const end = now;
    if (range === "custom") {
      const f = dateFrom ? new Date(dateFrom + "T00:00:00").getTime() : NaN;
      const tt = dateTo ? new Date(dateTo + "T23:59:59").getTime() : NaN;
      if (Number.isFinite(f) && Number.isFinite(tt)) {
        if (f > tt) {
          return {
            fromIso: new Date(end - 24 * 3600000).toISOString(),
            toIso: new Date(end).toISOString(),
            dateError: true,
          };
        }
        return {
          fromIso: new Date(f).toISOString(),
          toIso: new Date(tt).toISOString(),
          dateError: false,
        };
      }
      return {
        fromIso: new Date(end - 24 * 3600000).toISOString(),
        toIso: new Date(end).toISOString(),
        dateError: false,
      };
    }
    const hours: number = typeof range === "number" ? range : 24;
    return {
      fromIso: new Date(end - hours * 3600000).toISOString(),
      toIso: new Date(end).toISOString(),
      dateError: false,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [range, dateFrom, dateTo, selectedId, now]);

  const { points, loading: histLoading } = useTelemetryHistory(
    selectedId,
    fromIso,
    toIso,
  );

  const series = points.map((p) => {
    const time = new Date(p.timestamp);
    return {
      label:
        range === 24 || (range === "custom" && dateFrom === dateTo)
          ? time.toLocaleTimeString(locale, {
              hour: "2-digit",
              minute: "2-digit",
            })
          : time.toLocaleString(locale, {
              day: "numeric",
              month: "short",
              hour: "2-digit",
              minute: "2-digit",
            }),
      battery: p.battery,
      temp: p.temperature,
    };
  });

  function ago(seen: number) {
    if (!seen) return t("never");
    const min = Math.max(0, Math.floor((now - seen) / 60000));
    if (min < 60) return t("minAgo", { n: min });
    const hours = Math.floor(min / 60);
    if (hours < 24) return t("hoursAgo", { n: hours });
    return t("daysAgo", { n: Math.floor(hours / 24) });
  }

  const filters: ("all" | Status)[] = ["all", "online", "offline", "error"];

  if (roleLoading || loading) {
    return (
      <div className="flex justify-center py-16">
        <Loader2 className="size-7 animate-spin text-sky-500" />
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-bold text-sky-900 sm:text-2xl">
          {t("title")}
        </h1>
        <p className="text-sm text-sky-800/80">{t("subtitle")}</p>
      </div>

      {error && (
        <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 ring-1 ring-red-200">
          {t("loadFailed")}
        </p>
      )}

      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className={cardClass}
      >
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-base font-bold text-sky-900">
            {t("robotsTitle")}
          </h2>
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="mr-1 text-xs text-sky-700">
              {t("statusFilter")}:
            </span>
            {filters.map((f) => (
              <button
                key={f}
                type="button"
                onClick={() => setStatusFilter(f)}
                className={`h-8 rounded-lg px-3 text-xs font-semibold ring-1 transition-colors ${
                  statusFilter === f
                    ? "bg-sky-500 text-white ring-sky-500"
                    : "bg-white/80 text-sky-800 ring-sky-300 hover:bg-sky-50"
                }`}
              >
                {t(f)}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto rounded-2xl border border-sky-200 bg-white/80">
          <table className="w-full min-w-[720px] text-left text-sm">
            <thead className="bg-sky-100 text-sky-900">
              <tr>
                <th className="px-4 py-3 font-semibold">{t("robot")}</th>
                <th className="px-4 py-3 font-semibold">{t("status")}</th>
                <th className="px-4 py-3 font-semibold">{t("battery")}</th>
                <th className="px-4 py-3 font-semibold">{t("temperature")}</th>
                <th className="px-4 py-3 font-semibold">{t("activity")}</th>
                <th className="px-4 py-3 font-semibold">{t("errorCode")}</th>
                <th className="px-4 py-3 font-semibold">{t("lastComm")}</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 && (
                <tr>
                  <td
                    colSpan={7}
                    className="px-4 py-8 text-center text-sky-700"
                  >
                    {t("empty")}
                  </td>
                </tr>
              )}
              {filtered.map((r) => (
                <tr
                  key={r.id}
                  onClick={() => setPickedId(r.id)}
                  className={`cursor-pointer border-t border-sky-100 transition-colors hover:bg-sky-50 ${
                    r.id === selectedId ? "bg-sky-100/70" : ""
                  }`}
                >
                  <td className="px-4 py-3">
                    <span className="font-mono text-xs font-semibold text-sky-900">
                      {r.id}
                    </span>{" "}
                    <span className="text-sky-800">{r.name}</span>
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`rounded-full px-2.5 py-1 text-xs font-semibold ${statusStyle[r.status]}`}
                    >
                      {t(r.status)}
                    </span>
                  </td>
                  {r.hasData ? (
                    <>
                      <td className="px-4 py-3 text-sky-900">{r.battery}%</td>
                      <td className="px-4 py-3 text-sky-900">
                        {r.temperature}°C
                      </td>
                      <td className="px-4 py-3 text-sky-800">
                        {t(`act_${r.activity}`)}
                      </td>
                      <td className="px-4 py-3">
                        {r.errorCode ? (
                          <span className="font-mono text-xs font-semibold text-red-700">
                            {r.errorCode}
                          </span>
                        ) : (
                          <span className="text-sky-600">{t("none")}</span>
                        )}
                      </td>
                    </>
                  ) : (
                    <td colSpan={4} className="px-4 py-3 text-sky-600">
                      {t("noData")}
                    </td>
                  )}
                  <td className="px-4 py-3 text-sky-800">{ago(r.seen)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.section>

      {selected && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-sky-900">
              {t("selected")}:{" "}
              <span className="font-semibold">
                {selected.id} {selected.name}
              </span>
            </p>
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="mr-1 text-xs text-sky-700">{t("range")}:</span>
              {([24, 168] as const).map((h) => (
                <button
                  key={h}
                  type="button"
                  onClick={() => setRange(h)}
                  className={`h-8 rounded-lg px-3 text-xs font-semibold ring-1 transition-colors ${
                    range === h
                      ? "bg-sky-500 text-white ring-sky-500"
                      : "bg-white/80 text-sky-800 ring-sky-300 hover:bg-sky-50"
                  }`}
                >
                  {h === 24 ? t("range24h") : t("range7d")}
                </button>
              ))}
              <button
                type="button"
                onClick={() => setRange("custom")}
                className={`h-8 rounded-lg px-3 text-xs font-semibold ring-1 transition-colors ${
                  range === "custom"
                    ? "bg-sky-500 text-white ring-sky-500"
                    : "bg-white/80 text-sky-800 ring-sky-300 hover:bg-sky-50"
                }`}
              >
                {t("rangeCustom")}
              </button>
            </div>
          </div>

          {range === "custom" && (
            <div className="flex flex-wrap items-end gap-3">
              <label className="space-y-1 text-xs text-sky-800">
                <span>{t("dateFrom")}</span>
                <input
                  type="date"
                  value={dateFrom}
                  max={dateTo || undefined}
                  onChange={(e) => setDateFrom(e.target.value)}
                  className="block h-9 rounded-md border border-sky-200 bg-white px-2 text-sm text-sky-900 outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
                />
              </label>
              <label className="space-y-1 text-xs text-sky-800">
                <span>{t("dateTo")}</span>
                <input
                  type="date"
                  value={dateTo}
                  min={dateFrom || undefined}
                  onChange={(e) => setDateTo(e.target.value)}
                  className="block h-9 rounded-md border border-sky-200 bg-white px-2 text-sm text-sky-900 outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
                />
              </label>
              {dateError && (
                <p className="text-xs text-red-600">{t("dateInvalid")}</p>
              )}
            </div>
          )}

          <div className="grid gap-4 lg:grid-cols-2">
            <motion.section
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.1 }}
              className={cardClass}
            >
              <h2 className="mb-3 flex items-center gap-2 text-base font-bold text-sky-900">
                <Battery className="size-5 text-sky-600" />
                {t("batteryChart")}
              </h2>
              <div className="h-60 w-full">
                {histLoading ? (
                  <div className="flex h-full items-center justify-center">
                    <Loader2 className="size-6 animate-spin text-sky-500" />
                  </div>
                ) : series.length === 0 ? (
                  <p className="flex h-full items-center justify-center text-sm text-sky-700">
                    {t("noHistory")}
                  </p>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={series}
                      margin={{ top: 5, right: 8, left: -20, bottom: 0 }}
                    >
                      <defs>
                        <linearGradient id="gBat" x1="0" y1="0" x2="0" y2="1">
                          <stop
                            offset="0%"
                            stopColor="#0ea5e9"
                            stopOpacity={0.5}
                          />
                          <stop
                            offset="100%"
                            stopColor="#0ea5e9"
                            stopOpacity={0.05}
                          />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#bae6fd" />
                      <XAxis
                        dataKey="label"
                        tick={{ fill: "#0369a1", fontSize: 11 }}
                        interval="preserveStartEnd"
                      />
                      <YAxis
                        domain={[0, 100]}
                        tick={{ fill: "#0369a1", fontSize: 11 }}
                      />
                      <Tooltip />
                      <Area
                        type="monotone"
                        dataKey="battery"
                        name={t("battery")}
                        stroke="#0284c7"
                        strokeWidth={2.5}
                        fill="url(#gBat)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                )}
              </div>
            </motion.section>

            <motion.section
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.2 }}
              className={cardClass}
            >
              <h2 className="mb-3 flex items-center gap-2 text-base font-bold text-sky-900">
                <Thermometer className="size-5 text-sky-600" />
                {t("tempChart")}
              </h2>
              <div className="h-60 w-full">
                {histLoading ? (
                  <div className="flex h-full items-center justify-center">
                    <Loader2 className="size-6 animate-spin text-sky-500" />
                  </div>
                ) : series.length === 0 ? (
                  <p className="flex h-full items-center justify-center text-sm text-sky-700">
                    {t("noHistory")}
                  </p>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={series}
                      margin={{ top: 5, right: 8, left: -20, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#bae6fd" />
                      <XAxis
                        dataKey="label"
                        tick={{ fill: "#0369a1", fontSize: 11 }}
                        interval="preserveStartEnd"
                      />
                      <YAxis tick={{ fill: "#0369a1", fontSize: 11 }} />
                      <Tooltip />
                      <Line
                        type="monotone"
                        dataKey="temp"
                        name={t("temperature")}
                        stroke="#f97316"
                        strokeWidth={2.5}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </motion.section>
          </div>
        </>
      )}
    </div>
  );
}
