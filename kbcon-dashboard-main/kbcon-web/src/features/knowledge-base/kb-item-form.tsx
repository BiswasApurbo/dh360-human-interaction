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
import { Textarea } from "@/components/ui/textarea";
import { functions } from "@/lib/firebase";
import type { ItemRow } from "./use-knowledge";

export function KbItemForm({
  kbId,
  initial,
  onDone,
  onCancel,
}: {
  kbId: string;
  initial: ItemRow | null;
  onDone: () => void;
  onCancel: () => void;
}) {
  const t = useTranslations("KnowledgeBase");
  const [error, setError] = useState<string | null>(null);

  const schema = z.object({
    question: z.string().trim().min(2, t("questionShort")).max(500),
    answer: z.string().trim().min(1, t("answerRequired")).max(3000),
    language: z.enum(["bn", "en"]),
    category: z.string().trim().max(60),
  });
  type Values = z.infer<typeof schema>;

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<Values>({
    resolver: zodResolver(schema),
    defaultValues: {
      question: initial?.question ?? "",
      answer: initial?.answer ?? "",
      language: initial?.language ?? "bn",
      category: initial?.category ?? "general",
    },
  });

  async function onSubmit(values: Values) {
    setError(null);
    try {
      const fn = httpsCallable(functions, "saveKnowledgeItem");
      await fn({
        kbId,
        itemId: initial?.id ?? null,
        item: { ...values, category: values.category || "general" },
      });
      onDone();
    } catch (err) {
      const code = (err as { code?: string }).code;
      setError(
        code === "functions/already-exists"
          ? t("duplicateItem")
          : t("saveFailed"),
      );
    }
  }

  return (
    <form
      onSubmit={handleSubmit(onSubmit)}
      className="space-y-4 rounded-2xl border border-sky-200 bg-white/80 p-4 shadow-sm sm:p-5"
    >
      <div className="space-y-1.5">
        <Label htmlFor="q" className="text-sky-900">
          {t("question")}
        </Label>
        <Input
          id="q"
          className="h-10 border-sky-200 bg-white"
          {...register("question")}
        />
        {errors.question && (
          <p className="text-xs text-red-600">{errors.question.message}</p>
        )}
      </div>

      <div className="space-y-1.5">
        <Label htmlFor="a" className="text-sky-900">
          {t("answer")}
        </Label>
        <Textarea
          id="a"
          rows={4}
          className="border-sky-200 bg-white"
          {...register("answer")}
        />
        {errors.answer && (
          <p className="text-xs text-red-600">{errors.answer.message}</p>
        )}
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-1.5">
          <Label htmlFor="lang" className="text-sky-900">
            {t("language")}
          </Label>
          <select
            id="lang"
            className="h-10 w-full rounded-md border border-sky-200 bg-white px-3 text-sm text-sky-900 outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
            {...register("language")}
          >
            <option value="bn">{t("langBn")}</option>
            <option value="en">{t("langEn")}</option>
          </select>
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="cat" className="text-sky-900">
            {t("category")}
          </Label>
          <Input
            id="cat"
            className="h-10 border-sky-200 bg-white"
            {...register("category")}
          />
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
          {isSubmitting ? t("saving") : t("save")}
        </motion.button>
      </div>
    </form>
  );
}