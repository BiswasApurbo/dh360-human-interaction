"use client";

import { useRef, useState } from "react";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import Papa from "papaparse";
import { z } from "zod";
import { httpsCallable } from "firebase/functions";
import { FileUp, Loader2, X } from "lucide-react";
import { functions } from "@/lib/firebase";

const MAX_BYTES = 5 * 1024 * 1024;
const MAX_ROWS = 5000;

const rowSchema = z.object({
  question: z.string().min(2).max(500),
  answer: z.string().min(1).max(3000),
  language: z.enum(["bn", "en"]),
  category: z.string().max(60),
});

type Cleaned = {
  question: string;
  answer: string;
  language: string;
  category: string;
};
type RowError = { row: number; reason: string };
type Parsed = {
  fileName: string;
  format: "csv" | "json";
  rows: Cleaned[];
  errors: RowError[];
  valid: number;
};
type ServerResult = {
  total: number;
  imported: number;
  skipped: number;
  errors: RowError[];
};

function str(v: unknown): string {
  return typeof v === "string" || typeof v === "number" ? String(v).trim() : "";
}

function clean(raw: unknown): Cleaned {
  const o = (raw && typeof raw === "object" ? raw : {}) as Record<
    string,
    unknown
  >;
  return {
    question: str(o.question),
    answer: str(o.answer),
    language: str(o.language).toLowerCase(),
    category: str(o.category) || "general",
  };
}

const cardClass =
  "rounded-3xl border border-sky-200 bg-gradient-to-br from-white via-sky-50 to-sky-100 p-5 shadow-lg shadow-sky-200/60";

export function KbImport({ kbId }: { kbId: string }) {
  const t = useTranslations("KnowledgeBase");
  const inputRef = useRef<HTMLInputElement>(null);
  const [parsed, setParsed] = useState<Parsed | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<ServerResult | null>(null);
  const [failed, setFailed] = useState(false);

  function reasonKey(issuePath: PropertyKey | undefined) {
    if (issuePath === "language") return "reasonLanguage";
    if (issuePath === "question") return "reasonQuestion";
    if (issuePath === "answer") return "reasonAnswer";
    return "reasonInvalid";
  }

  async function onFile(file: File | undefined) {
    setProblem(null);
    setParsed(null);
    setResult(null);
    setFailed(false);
    if (!file) return;

    const lower = file.name.toLowerCase();
    const format = lower.endsWith(".csv")
      ? "csv"
      : lower.endsWith(".json")
        ? "json"
        : null;
    if (!format) return setProblem(t("badFile"));
    if (file.size > MAX_BYTES) return setProblem(t("fileTooBig"));

    let rawRows: unknown[];
    try {
      const text = (await file.text()).replace(/^\uFEFF/, "");
      if (format === "csv") {
        const res = Papa.parse<Record<string, string>>(text, {
          header: true,
          skipEmptyLines: true,
          transformHeader: (h) => h.trim().toLowerCase(),
        });
        rawRows = res.data;
      } else {
        const data: unknown = JSON.parse(text);
        if (!Array.isArray(data)) return setProblem(t("badJson"));
        rawRows = data;
      }
    } catch {
      return setProblem(format === "json" ? t("badJson") : t("readFailed"));
    }

    if (rawRows.length > MAX_ROWS) return setProblem(t("tooManyRows"));

    const rows = rawRows.map(clean);
    const errors: RowError[] = [];
    rows.forEach((r, i) => {
      const check = rowSchema.safeParse(r);
      if (!check.success) {
        errors.push({
          row: i + 1,
          reason: reasonKey(check.error.issues[0]?.path[0]),
        });
      }
    });

    setParsed({
      fileName: file.name,
      format,
      rows,
      errors,
      valid: rows.length - errors.length,
    });
  }

  async function runImport() {
    if (!parsed) return;
    setBusy(true);
    setFailed(false);
    try {
      const fn = httpsCallable<unknown, ServerResult>(
        functions,
        "importKnowledge",
      );
      const res = await fn({
        kbId,
        fileName: parsed.fileName,
        format: parsed.format,
        rows: parsed.rows,
      });
      setResult(res.data);
      setParsed(null);
    } catch {
      setFailed(true);
    } finally {
      setBusy(false);
    }
  }

  function serverReason(reason: string) {
    return reason.includes("Duplicate")
      ? t("reasonDuplicate")
      : t("reasonInvalid");
  }

  return (
    <section className={cardClass}>
      <h2 className="text-base font-bold text-sky-900">{t("importTitle")}</h2>
      <p className="mb-4 text-xs text-sky-700">{t("importDesc")}</p>

      <input
        ref={inputRef}
        type="file"
        accept=".csv,.json,text/csv,application/json"
        className="hidden"
        onChange={(e) => {
          onFile(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      <motion.button
        type="button"
        onClick={() => inputRef.current?.click()}
        whileHover={{ scale: 1.03 }}
        whileTap={{ scale: 0.97 }}
        className="flex h-10 items-center gap-2 rounded-xl bg-white/80 px-4 text-sm font-semibold text-sky-800 ring-1 ring-sky-300 hover:bg-sky-50"
      >
        <FileUp className="size-4" />
        {t("chooseFile")}
      </motion.button>

      {problem && (
        <p className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 ring-1 ring-red-200">
          {problem}
        </p>
      )}

      {parsed && (
        <div className="mt-4 space-y-3">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="font-semibold text-sky-900">
              {parsed.fileName}
            </span>
            <span className="rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-semibold text-emerald-700">
              {t("validRows")}: {parsed.valid}
            </span>
            <span className="rounded-full bg-red-100 px-2.5 py-0.5 text-xs font-semibold text-red-700">
              {t("invalidRows")}: {parsed.errors.length}
            </span>
            <button
              type="button"
              onClick={() => setParsed(null)}
              className="ml-auto inline-flex items-center gap-1 text-xs text-sky-700 hover:underline"
            >
              <X className="size-3" />
              {t("clear")}
            </button>
          </div>

          <h3 className="text-sm font-bold text-sky-900">
            {t("previewTitle")}
          </h3>
          <div className="overflow-x-auto rounded-xl border border-sky-200 bg-white">
            <table className="w-full min-w-[480px] text-left text-xs">
              <thead className="bg-sky-100 text-sky-900">
                <tr>
                  <th className="px-3 py-2">{t("question")}</th>
                  <th className="px-3 py-2">{t("answer")}</th>
                  <th className="px-3 py-2">{t("language")}</th>
                  <th className="px-3 py-2">{t("category")}</th>
                </tr>
              </thead>
              <tbody>
                {parsed.rows.slice(0, 8).map((r, i) => (
                  <tr key={i} className="border-t border-sky-100">
                    <td className="max-w-[200px] truncate px-3 py-2">
                      {r.question}
                    </td>
                    <td className="max-w-[240px] truncate px-3 py-2">
                      {r.answer}
                    </td>
                    <td className="px-3 py-2">{r.language}</td>
                    <td className="px-3 py-2">{r.category}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {parsed.errors.length > 0 && (
            <div className="rounded-xl border border-red-200 bg-red-50 p-3">
              <h3 className="mb-2 text-sm font-bold text-red-700">
                {t("errorReport")}
              </h3>
              <ul className="max-h-40 space-y-1 overflow-y-auto text-xs text-red-700">
                {parsed.errors.slice(0, 100).map((e) => (
                  <li key={e.row}>
                    {t("row")} {e.row}: {t(e.reason)}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <motion.button
            type="button"
            disabled={busy || parsed.valid === 0}
            onClick={runImport}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            className="flex h-10 items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-5 text-sm font-semibold text-white shadow-md shadow-sky-300/60 disabled:opacity-60"
          >
            {busy && <Loader2 className="size-4 animate-spin" />}
            {busy ? t("importing") : t("importBtn")}
          </motion.button>
        </div>
      )}

      {failed && (
        <p className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 ring-1 ring-red-200">
          {t("importFailed")}
        </p>
      )}

      {result && (
        <div className="mt-4 space-y-3">
          <p className="rounded-lg bg-emerald-50 px-3 py-2 text-sm text-emerald-700 ring-1 ring-emerald-200">
            {t("importDone", {
              imported: result.imported,
              total: result.total,
              skipped: result.skipped,
            })}
          </p>
          {result.errors.length > 0 && (
            <div className="rounded-xl border border-red-200 bg-red-50 p-3">
              <h3 className="mb-2 text-sm font-bold text-red-700">
                {t("errorReport")}
              </h3>
              <ul className="max-h-40 space-y-1 overflow-y-auto text-xs text-red-700">
                {result.errors.slice(0, 100).map((e, i) => (
                  <li key={i}>
                    {t("row")} {e.row}: {serverReason(e.reason)}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
