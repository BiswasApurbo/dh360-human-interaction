"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { ArrowLeft, CheckCircle2, Loader2, Mail } from "lucide-react";
import { sendPasswordResetEmail } from "firebase/auth";
import { Link } from "@/i18n/navigation";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { auth } from "@/lib/firebase";

export function ForgotPasswordForm() {
  const t = useTranslations("ForgotPassword");
  const tl = useTranslations("Login");
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const schema = z.object({
    email: z.string().email(tl("invalidEmail")),
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
      await sendPasswordResetEmail(auth, values.email);
      setSent(true);
    } catch (err) {
      const code = (err as { code?: string }).code;
      // ইমেইল আছে কি নেই তা ফাঁস না করতে user-not-found হলেও সফল দেখানো হয়
      if (code === "auth/user-not-found") {
        setSent(true);
      } else {
        setError(t("failed"));
      }
    }
  }

  return (
    <div className="space-y-4">
      {sent ? (
        <motion.div
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-start gap-2 rounded-xl bg-emerald-50 px-3 py-3 text-sm text-emerald-700 ring-1 ring-emerald-200"
        >
          <CheckCircle2 className="mt-0.5 size-4 shrink-0" />
          <p>{t("sent")}</p>
        </motion.div>
      ) : (
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="email" className="text-sky-900">
              {tl("email")}
            </Label>
            <div className="relative">
              <Mail className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-sky-600" />
              <Input
                id="email"
                type="email"
                autoComplete="email"
                placeholder={tl("emailPlaceholder")}
                className="h-11 border-sky-200 bg-white/80 pl-9 focus-visible:ring-sky-400"
                {...register("email")}
              />
            </div>
            {errors.email && (
              <p className="text-xs text-red-600">{errors.email.message}</p>
            )}
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

          <motion.button
            type="submit"
            disabled={isSubmitting}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            className="flex h-11 w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 text-sm font-semibold text-white shadow-md shadow-sky-300/60 disabled:opacity-70"
          >
            {isSubmitting && <Loader2 className="size-4 animate-spin" />}
            {isSubmitting ? t("sending") : t("send")}
          </motion.button>
        </form>
      )}

      <Link
        href="/login"
        className="flex items-center justify-center gap-1.5 text-sm font-medium text-sky-700 hover:underline"
      >
        <ArrowLeft className="size-4" />
        {t("backToLogin")}
      </Link>
    </div>
  );
}
