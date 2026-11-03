"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Eye, EyeOff, Loader2, ShieldCheck, Bot } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/features/auth/use-auth";
import { useRobots } from "@/features/robots/use-robots";

export function CreateAdminForm({
  onDone,
  onCancel,
}: {
  onDone: () => void;
  onCancel: () => void;
}) {
  const t = useTranslations("Admins");
  const tl = useTranslations("Login");
  const { user } = useAuth();
  const { robots: allRobots, loading: robotsLoading } = useRobots("superAdmin");
  const [selectedRobots, setSelectedRobots] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showPassword, setShowPassword] = useState(false);

  // Module permissions state
  const [permissions, setPermissions] = useState({
    dashboard: true,
    robots: true,
    assistant: true,
    people: true,
    knowledgeBase: true,
    telemetry: true,
    settings: true,
  });

  const schema = z.object({
    name: z.string().trim().min(2, t("nameShort")),
    email: z.string().trim().email(tl("invalidEmail")),
    password: z.string().min(6, tl("shortPassword")),
  });
  type Values = z.infer<typeof schema>;

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<Values>({ resolver: zodResolver(schema) });

  function togglePerm(key: keyof typeof permissions) {
    setPermissions((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  function toggleRobot(id: string) {
    setSelectedRobots((prev) =>
      prev.includes(id) ? prev.filter((r) => r !== id) : [...prev, id]
    );
  }

  async function onSubmit(values: Values) {
    if (!user) return;
    setError(null);
    try {
      const token = await user.getIdToken(true);
      const res = await fetch("/api/admin/users", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          ...values,
          permissions,
          assignedRobots: selectedRobots,
        }),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        if (res.status === 409) {
          setError(t("emailExists"));
          return;
        }
        const message =
          typeof errJson.detail === "string"
            ? errJson.detail
            : Array.isArray(errJson.detail)
              ? errJson.detail.map((d: { msg?: string }) => d.msg || "Validation error").join("; ")
              : (errJson.message || t("createFailed"));
        throw new Error(message);
      }

      onDone();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : t("createFailed"));
    }
  }

  const moduleItems: { key: keyof typeof permissions; label: string }[] = [
    { key: "dashboard", label: "Dashboard Overview" },
    { key: "robots", label: "Robots Management" },
    { key: "assistant", label: "AI Assistant" },
    { key: "people", label: "People & Enrollment" },
    { key: "knowledgeBase", label: "Knowledge Base" },
    { key: "telemetry", label: "Telemetry & History" },
    { key: "settings", label: "System Settings" },
  ];

  return (
    <form
      onSubmit={handleSubmit(onSubmit)}
      className="space-y-5 rounded-2xl border border-sky-200 bg-white/95 p-5 shadow-sm sm:p-6"
    >
      <div className="flex items-center gap-2 border-b border-sky-100 pb-3">
        <ShieldCheck className="size-5 text-sky-600" />
        <h3 className="text-sm font-bold text-sky-950">
          Provision New Administrator Account
        </h3>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <div className="space-y-1.5">
          <Label htmlFor="name" className="text-xs font-semibold text-sky-900">
            {t("name")}
          </Label>
          <Input
            id="name"
            placeholder="Full Name"
            className="h-10 border-sky-200 bg-white"
            {...register("name")}
          />
          {errors.name && (
            <p className="text-xs text-red-600">{errors.name.message}</p>
          )}
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="admin-email" className="text-xs font-semibold text-sky-900">
            {t("email")}
          </Label>
          <Input
            id="admin-email"
            type="email"
            autoComplete="off"
            placeholder="admin@example.com"
            className="h-10 border-sky-200 bg-white"
            {...register("email")}
          />
          {errors.email && (
            <p className="text-xs text-red-600">{errors.email.message}</p>
          )}
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="admin-password" className="text-xs font-semibold text-sky-900">
            {t("password")}
          </Label>
          <div className="relative">
            <Input
              id="admin-password"
              type={showPassword ? "text" : "password"}
              autoComplete="new-password"
              placeholder="Initial Password"
              className="h-10 pr-10 border-sky-200 bg-white"
              {...register("password")}
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
            >
              {showPassword ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
            </button>
          </div>
          {errors.password && (
            <p className="text-xs text-red-600">{errors.password.message}</p>
          )}
        </div>
      </div>

      {/* Assigned Robots Selection */}
      <div className="rounded-xl border border-sky-100 bg-sky-50/50 p-4">
        <div className="flex items-center gap-2 mb-2">
          <Bot className="size-4 text-sky-700" />
          <h4 className="text-xs font-bold text-sky-900">
            Assigned Robots (Multi-Select)
          </h4>
        </div>
        {robotsLoading ? (
          <div className="flex items-center gap-2 py-2 text-xs text-sky-700">
            <Loader2 className="size-3.5 animate-spin" />
            <span>Loading robots...</span>
          </div>
        ) : allRobots.length === 0 ? (
          <p className="text-xs text-sky-700/80 italic">No robots currently registered.</p>
        ) : (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {allRobots.map((robot) => (
              <label
                key={robot.id}
                className="flex items-center gap-2 rounded-lg bg-white p-2 border border-sky-100 text-xs font-medium text-slate-700 cursor-pointer hover:bg-sky-50 transition-colors"
              >
                <input
                  type="checkbox"
                  checked={selectedRobots.includes(robot.id)}
                  onChange={() => toggleRobot(robot.id)}
                  className="rounded border-sky-300 text-sky-600 focus:ring-sky-500"
                />
                <span className="font-semibold text-sky-950">{robot.name}</span>
                <span className="text-[10px] font-mono text-sky-700">({robot.id})</span>
              </label>
            ))}
          </div>
        )}
      </div>

      {/* Granular Module Permissions */}
      <div className="rounded-xl border border-sky-100 bg-sky-50/50 p-4">
        <h4 className="text-xs font-bold text-sky-900 mb-2">
          Assigned Module Permissions
        </h4>
        <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
          {moduleItems.map(({ key, label }) => (
            <label
              key={key}
              className="flex items-center gap-2 rounded-lg bg-white p-2 border border-sky-100 text-xs font-medium text-slate-700 cursor-pointer hover:bg-sky-50 transition-colors"
            >
              <input
                type="checkbox"
                checked={permissions[key]}
                onChange={() => togglePerm(key)}
                className="rounded border-sky-300 text-sky-600 focus:ring-sky-500"
              />
              <span>{label}</span>
            </label>
          ))}
        </div>
      </div>

      {error && (
        <motion.p
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 ring-1 ring-red-200"
        >
          {error}
        </motion.p>
      )}

      <div className="flex flex-wrap justify-end gap-2 pt-2 border-t border-sky-100">
        <button
          type="button"
          onClick={onCancel}
          className="h-10 rounded-xl px-4 text-sm font-medium text-sky-800 ring-1 ring-sky-300 hover:bg-sky-50"
        >
          {t("cancel")}
        </button>
        <motion.button
          type="submit"
          disabled={isSubmitting}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="flex h-10 items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-5 text-sm font-semibold text-white shadow-md shadow-sky-300/60 disabled:opacity-70"
        >
          {isSubmitting && <Loader2 className="size-4 animate-spin" />}
          {isSubmitting ? t("creating") : t("create")}
        </motion.button>
      </div>
    </form>
  );
}
