"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { httpsCallable } from "firebase/functions";
import { Loader2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { functions } from "@/lib/firebase";

export function CreateRobotForm({
  onDone,
  onCancel,
}: {
  onDone: (secret: string, robotId: string) => void;
  onCancel: () => void;
}) {
  const t = useTranslations("Robots");
  const [error, setError] = useState<string | null>(null);

  const schema = z.object({
    robotId: z
      .string()
      .trim()
      .regex(/^[A-Za-z0-9-]{3,32}$/, t("idInvalid")),
    name: z.string().trim().min(2, t("nameShort")),
    type: z.string().trim().min(2, t("typeShort")),
    mode: z.enum(["education", "healthcare"]),
  });
  type Values = z.infer<typeof schema>;

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<Values>({
    resolver: zodResolver(schema),
    defaultValues: { mode: "education" },
  });

  async function onSubmit(values: Values) {
    setError(null);
    try {
      const fn = httpsCallable<Values, { robotId: string; secret: string }>(
        functions,
        "createRobot",
      );
      const res = await fn(values);
      onDone(res.data.secret, res.data.robotId);
    } catch (err) {
      const code = (err as { code?: string }).code;
      setError(
        code === "functions/already-exists" ? t("idExists") : t("createFailed"),
      );
    }
  }

  return (
    <form
      onSubmit={handleSubmit(onSubmit)}
      className="space-y-4 rounded-2xl border border-sky-200 bg-white/80 p-4 shadow-sm sm:p-5"
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-1.5">
          <Label htmlFor="robot-id" className="text-sky-900">
            {t("robotId")}
          </Label>
          <Input
            id="robot-id"
            placeholder="KB-001"
            autoComplete="off"
            className="h-10 border-sky-200 bg-white"
            {...register("robotId")}
          />
          <p className="text-[11px] text-sky-700">{t("robotIdHint")}</p>
          {errors.robotId && (
            <p className="text-xs text-red-600">{errors.robotId.message}</p>
          )}
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="robot-name" className="text-sky-900">
            {t("name")}
          </Label>
          <Input
            id="robot-name"
            className="h-10 border-sky-200 bg-white"
            {...register("name")}
          />
          {errors.name && (
            <p className="text-xs text-red-600">{errors.name.message}</p>
          )}
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="robot-type" className="text-sky-900">
            {t("type")}
          </Label>
          <Input
            id="robot-type"
            className="h-10 border-sky-200 bg-white"
            {...register("type")}
          />
          {errors.type && (
            <p className="text-xs text-red-600">{errors.type.message}</p>
          )}
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="robot-mode" className="text-sky-900">
            {t("mode")}
          </Label>
          <select
            id="robot-mode"
            className="h-10 w-full rounded-md border border-sky-200 bg-white px-3 text-sm text-sky-900 outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
            {...register("mode")}
          >
            <option value="education">{t("education")}</option>
            <option value="healthcare">{t("healthcare")}</option>
          </select>
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

      <div className="flex flex-wrap justify-end gap-2">
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
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
          className="flex h-10 items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-4 text-sm font-semibold text-white shadow-md shadow-sky-300/60 disabled:opacity-70"
        >
          {isSubmitting && <Loader2 className="size-4 animate-spin" />}
          {isSubmitting ? t("creating") : t("create")}
        </motion.button>
      </div>
    </form>
  );
}
