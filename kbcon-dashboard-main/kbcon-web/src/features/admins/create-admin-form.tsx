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

export function CreateAdminForm({
  onDone,
  onCancel,
}: {
  onDone: () => void;
  onCancel: () => void;
}) {
  const t = useTranslations("Admins");
  const tl = useTranslations("Login");
  const [error, setError] = useState<string | null>(null);

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

  async function onSubmit(values: Values) {
    setError(null);
    try {
      const createAdmin = httpsCallable(functions, "createAdmin");
      await createAdmin(values);
      onDone();
    } catch (err) {
      const code = (err as { code?: string }).code;
      setError(
        code === "functions/already-exists"
          ? t("emailExists")
          : t("createFailed"),
      );
    }
  }

  return (
    <form
      onSubmit={handleSubmit(onSubmit)}
      className="space-y-4 rounded-2xl border border-sky-200 bg-white/80 p-4 shadow-sm sm:p-5"
    >
      <div className="grid gap-4 sm:grid-cols-3">
        <div className="space-y-1.5">
          <Label htmlFor="name" className="text-sky-900">
            {t("name")}
          </Label>
          <Input
            id="name"
            className="h-10 border-sky-200 bg-white"
            {...register("name")}
          />
          {errors.name && (
            <p className="text-xs text-red-600">{errors.name.message}</p>
          )}
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="admin-email" className="text-sky-900">
            {t("email")}
          </Label>
          <Input
            id="admin-email"
            type="email"
            autoComplete="off"
            className="h-10 border-sky-200 bg-white"
            {...register("email")}
          />
          {errors.email && (
            <p className="text-xs text-red-600">{errors.email.message}</p>
          )}
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="admin-password" className="text-sky-900">
            {t("password")}
          </Label>
          <Input
            id="admin-password"
            type="password"
            autoComplete="new-password"
            className="h-10 border-sky-200 bg-white"
            {...register("password")}
          />
          {errors.password && (
            <p className="text-xs text-red-600">{errors.password.message}</p>
          )}
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
