"use client";

import { useMemo, useState } from "react";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import { httpsCallable } from "firebase/functions";
import { useAuth } from "@/features/auth/use-auth";
import {
  ArrowLeft,
  Check,
  Loader2,
  Pencil,
  Plus,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { functions } from "@/lib/firebase";
import type { Role } from "@/features/auth/use-role";
import { useRobots } from "@/features/robots/use-robots";
import {
  useImportHistory,
  useKbChanges,
  useKbItems,
  type ItemRow,
  type KbRow,
} from "./use-knowledge";
import { KbItemForm } from "./kb-item-form";
import { KbImport } from "./kb-import";

const cardClass =
  "rounded-3xl border border-sky-200 bg-gradient-to-br from-white via-sky-50 to-sky-100 p-5 shadow-lg shadow-sky-200/60";
const SHOW_LIMIT = 200;

export function KbDetail({
  kb,
  role,
  onBack,
}: {
  kb: KbRow;
  role: Role;
  onBack: () => void;
}) {
  const t = useTranslations("KnowledgeBase");
  const { items, loading, error } = useKbItems(kb.id);
  const { history, loading: histLoading } = useImportHistory(kb.id, role);
  const { robots } = useRobots(role);
  const { changes, loading: changesLoading } = useKbChanges(kb.id);
  const { user } = useAuth();

  const [editing, setEditing] = useState<ItemRow | "new" | null>(null);
  const [search, setSearch] = useState("");
  const [busyId, setBusyId] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);
  const [newName, setNewName] = useState(kb.name);
  const [renameBusy, setRenameBusy] = useState(false);

  async function saveName() {
    const n = newName.trim();
    if (n.length < 2) return;
    setRenameBusy(true);
    setMsg(null);
    try {
      const fn = httpsCallable(functions, "renameKnowledgeBase");
      await fn({ kbId: kb.id, name: n });
      setRenaming(false);
    } catch {
      setMsg(t("renameFailed"));
    } finally {
      setRenameBusy(false);
    }
  }

  const matching = robots.filter((r) => r.mode === kb.mode);

  const filtered = useMemo(() => {
    const s = search.trim().toLowerCase();
    if (!s) return items;
    return items.filter(
      (i) =>
        i.question.toLowerCase().includes(s) ||
        i.answer.toLowerCase().includes(s),
    );
  }, [items, search]);

  async function toggleLink(robotId: string, linked: boolean) {
    setBusyId(robotId);
    setMsg(null);
    try {
      const fn = httpsCallable(functions, "linkKnowledgeBase");
      await fn({ robotId, kbId: linked ? null : kb.id });
    } catch {
      setMsg(t("linkFailed"));
    } finally {
      setBusyId(null);
    }
  }

  async function remove(itemId: string) {
    if (!window.confirm(t("confirmDelete"))) return;
    setBusyId(itemId);
    setMsg(null);
    try {
      const fn = httpsCallable(functions, "deleteKnowledgeItem");
      await fn({ kbId: kb.id, itemId });
    } catch {
      setMsg(t("deleteFailed"));
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex h-9 items-center gap-1.5 rounded-lg px-3 text-xs font-semibold text-sky-800 ring-1 ring-sky-300 hover:bg-sky-50"
        >
          <ArrowLeft className="size-4" />
          {t("back")}
        </button>
        <div>
          {renaming ? (
            <div className="flex items-center gap-1.5">
              <input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                className="h-9 w-48 rounded-md border border-sky-200 bg-white px-3 text-sm text-sky-900 outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
              />
              <button
                type="button"
                aria-label={t("save")}
                disabled={renameBusy}
                onClick={saveName}
                className="inline-flex size-9 items-center justify-center rounded-lg bg-sky-500 text-white disabled:opacity-60"
              >
                {renameBusy ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Check className="size-4" />
                )}
              </button>
              <button
                type="button"
                aria-label={t("cancel")}
                onClick={() => {
                  setRenaming(false);
                  setNewName(kb.name);
                }}
                className="inline-flex size-9 items-center justify-center rounded-lg text-sky-800 ring-1 ring-sky-300 hover:bg-sky-50"
              >
                <X className="size-4" />
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-bold text-sky-900">{kb.name}</h2>
              <button
                type="button"
                aria-label={t("rename")}
                onClick={() => {
                  setNewName(kb.name);
                  setRenaming(true);
                }}
                className="inline-flex size-8 items-center justify-center rounded-lg text-sky-800 ring-1 ring-sky-300 hover:bg-sky-50"
              >
                <Pencil className="size-3.5" />
              </button>
            </div>
          )}
          <p className="text-xs text-sky-700">
            {t(kb.mode)} · {t("version")} {kb.version} · {kb.itemCount}{" "}
            {t("itemCount")}
          </p>
        </div>
      </div>

      {msg && (
        <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 ring-1 ring-red-200">
          {msg}
        </p>
      )}

      {/* Robot যুক্ত করা */}
      <section className={cardClass}>
        <h3 className="text-base font-bold text-sky-900">{t("robotsTitle")}</h3>
        <p className="mb-3 text-xs text-sky-700">{t("robotsDesc")}</p>
        {matching.length === 0 ? (
          <p className="text-sm text-sky-700">{t("noMatchingRobots")}</p>
        ) : (
          <ul className="space-y-2">
            {matching.map((r) => {
              const linked = kb.robotIds.includes(r.id);
              return (
                <li
                  key={r.id}
                  className="flex items-center justify-between gap-3 rounded-xl bg-white/70 px-3 py-2 ring-1 ring-sky-100"
                >
                  <span className="min-w-0 truncate text-sm text-sky-900">
                    <span className="font-mono text-xs font-semibold">
                      {r.id}
                    </span>{" "}
                    {r.name}
                  </span>
                  <button
                    type="button"
                    disabled={busyId === r.id}
                    onClick={() => toggleLink(r.id, linked)}
                    className={`inline-flex h-8 shrink-0 items-center gap-1.5 rounded-lg px-3 text-xs font-semibold ring-1 disabled:opacity-60 ${
                      linked
                        ? "bg-sky-500 text-white ring-sky-500"
                        : "text-sky-800 ring-sky-300 hover:bg-sky-50"
                    }`}
                  >
                    {busyId === r.id && (
                      <Loader2 className="size-3 animate-spin" />
                    )}
                    {linked ? t("unlink") : t("link")}
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      {/* প্রশ্ন-উত্তর */}
      <section className={cardClass}>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <h3 className="text-base font-bold text-sky-900">
            {t("itemsTitle")}
          </h3>
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-sky-600" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder={t("search")}
                className="h-9 w-44 rounded-full border-0 bg-white pl-9 pr-3 text-sm text-sky-900 outline-none ring-1 ring-sky-300 focus:ring-2 focus:ring-sky-500"
              />
            </div>
            {editing === null && (
              <motion.button
                type="button"
                onClick={() => setEditing("new")}
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.96 }}
                className="flex h-9 items-center gap-1.5 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-3 text-xs font-semibold text-white shadow-md shadow-sky-300/60"
              >
                <Plus className="size-4" />
                {t("addItem")}
              </motion.button>
            )}
          </div>
        </div>

        {editing !== null && (
          <div className="mb-4">
            <KbItemForm
              key={editing === "new" ? "new" : editing.id}
              kbId={kb.id}
              initial={editing === "new" ? null : editing}
              onCancel={() => setEditing(null)}
              onDone={() => setEditing(null)}
            />
          </div>
        )}

        <div className="overflow-x-auto rounded-2xl border border-sky-200 bg-white/80">
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead className="bg-sky-100 text-sky-900">
              <tr>
                <th className="px-4 py-3 font-semibold">{t("question")}</th>
                <th className="px-4 py-3 font-semibold">{t("answer")}</th>
                <th className="px-4 py-3 font-semibold">{t("language")}</th>
                <th className="px-4 py-3 font-semibold">{t("category")}</th>
                <th className="px-4 py-3 text-right font-semibold">
                  {t("actions")}
                </th>
              </tr>
            </thead>
            <tbody>
              {loading && (
                <tr>
                  <td colSpan={5} className="px-4 py-8 text-center">
                    <Loader2 className="mx-auto size-6 animate-spin text-sky-500" />
                  </td>
                </tr>
              )}
              {!loading && error && (
                <tr>
                  <td
                    colSpan={5}
                    className="px-4 py-8 text-center text-red-600"
                  >
                    {t("loadFailed")}
                  </td>
                </tr>
              )}
              {!loading && !error && filtered.length === 0 && (
                <tr>
                  <td
                    colSpan={5}
                    className="px-4 py-8 text-center text-sky-700"
                  >
                    {t("noItems")}
                  </td>
                </tr>
              )}
              {filtered.slice(0, SHOW_LIMIT).map((i) => (
                <tr key={i.id} className="border-t border-sky-100 align-top">
                  <td className="max-w-[220px] px-4 py-3 font-medium text-sky-900">
                    {i.question}
                  </td>
                  <td className="max-w-[300px] px-4 py-3 text-sky-800">
                    {i.answer}
                  </td>
                  <td className="px-4 py-3 text-sky-800">
                    {t(i.language === "bn" ? "langBn" : "langEn")}
                  </td>
                  <td className="px-4 py-3 text-sky-800">{i.category}</td>
                  <td className="px-4 py-3 text-right">
                    <div className="inline-flex gap-1.5">
                      <button
                        type="button"
                        aria-label={t("edit")}
                        onClick={() => setEditing(i)}
                        className="inline-flex size-8 items-center justify-center rounded-lg text-sky-800 ring-1 ring-sky-300 hover:bg-sky-50"
                      >
                        <Pencil className="size-3.5" />
                      </button>
                      <button
                        type="button"
                        aria-label={t("delete")}
                        disabled={busyId === i.id}
                        onClick={() => remove(i.id)}
                        className="inline-flex size-8 items-center justify-center rounded-lg text-red-600 ring-1 ring-red-200 hover:bg-red-50 disabled:opacity-60"
                      >
                        {busyId === i.id ? (
                          <Loader2 className="size-3.5 animate-spin" />
                        ) : (
                          <Trash2 className="size-3.5" />
                        )}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {filtered.length > SHOW_LIMIT && (
          <p className="mt-2 text-xs text-sky-700">
            {t("showingFirst", { n: SHOW_LIMIT })}
          </p>
        )}
      </section>

      <KbImport kbId={kb.id} />

      {/* Import ইতিহাস */}
      <section className={cardClass}>
        <h3 className="mb-3 text-base font-bold text-sky-900">
          {t("historyTitle")}
        </h3>
        <div className="overflow-x-auto rounded-2xl border border-sky-200 bg-white/80">
          <table className="w-full min-w-[520px] text-left text-sm">
            <thead className="bg-sky-100 text-sky-900">
              <tr>
                <th className="px-4 py-3 font-semibold">{t("file")}</th>
                <th className="px-4 py-3 font-semibold">{t("total")}</th>
                <th className="px-4 py-3 font-semibold">{t("imported")}</th>
                <th className="px-4 py-3 font-semibold">{t("skipped")}</th>
                <th className="px-4 py-3 font-semibold">{t("date")}</th>
              </tr>
            </thead>
            <tbody>
              {histLoading && (
                <tr>
                  <td colSpan={5} className="px-4 py-6 text-center">
                    <Loader2 className="mx-auto size-5 animate-spin text-sky-500" />
                  </td>
                </tr>
              )}
              {!histLoading && history.length === 0 && (
                <tr>
                  <td
                    colSpan={5}
                    className="px-4 py-6 text-center text-sky-700"
                  >
                    {t("noHistory")}
                  </td>
                </tr>
              )}
              {history.map((h) => (
                <tr key={h.id} className="border-t border-sky-100">
                  <td className="px-4 py-3 text-sky-900">{h.fileName}</td>
                  <td className="px-4 py-3 text-sky-800">{h.total}</td>
                  <td className="px-4 py-3 text-emerald-700">{h.imported}</td>
                  <td className="px-4 py-3 text-slate-600">{h.skipped}</td>
                  <td className="px-4 py-3 text-sky-800">
                    {h.at ? new Date(h.at).toLocaleString() : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className={cardClass}>
        <h3 className="mb-3 text-base font-bold text-sky-900">
          {t("changesTitle")}
        </h3>
        <div className="overflow-x-auto rounded-2xl border border-sky-200 bg-white/80">
          <table className="w-full min-w-[520px] text-left text-sm">
            <thead className="bg-sky-100 text-sky-900">
              <tr>
                <th className="px-4 py-3 font-semibold">{t("changeAction")}</th>
                <th className="px-4 py-3 font-semibold">
                  {t("changeQuestion")}
                </th>
                <th className="px-4 py-3 font-semibold">{t("changeBy")}</th>
                <th className="px-4 py-3 font-semibold">{t("date")}</th>
              </tr>
            </thead>
            <tbody>
              {changesLoading && (
                <tr>
                  <td colSpan={4} className="px-4 py-6 text-center">
                    <Loader2 className="mx-auto size-5 animate-spin text-sky-500" />
                  </td>
                </tr>
              )}
              {!changesLoading && changes.length === 0 && (
                <tr>
                  <td
                    colSpan={4}
                    className="px-4 py-6 text-center text-sky-700"
                  >
                    {t("noChanges")}
                  </td>
                </tr>
              )}
              {changes.map((c) => (
                <tr key={c.id} className="border-t border-sky-100">
                  <td className="px-4 py-3 font-semibold text-sky-900">
                    {t(`act_${c.action}`)}
                  </td>
                  <td className="max-w-[320px] truncate px-4 py-3 text-sky-800">
                    {c.question}
                  </td>
                  <td className="px-4 py-3 text-sky-800">
                    {c.by === user?.uid ? t("you") : c.by.slice(0, 6)}
                  </td>
                  <td className="px-4 py-3 text-sky-800">
                    {c.at ? new Date(c.at).toLocaleString() : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
