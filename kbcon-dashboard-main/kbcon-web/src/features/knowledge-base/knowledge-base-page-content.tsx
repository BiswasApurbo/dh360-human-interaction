"use client";

import { useMemo, useState } from "react";
import { useTranslations } from "next-intl";
import { motion, AnimatePresence } from "motion/react";
import {
  BookOpen,
  Filter,
  GraduationCap,
  Hospital,
  Globe,
  Loader2,
  Pencil,
  Plus,
  RefreshCw,
  Search,
  ShieldAlert,
  Trash2,
  X,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import { useRole } from "@/features/auth/use-role";
import {
  useCanonicalKnowledge,
  type CanonicalKnowledgeItem,
} from "./use-knowledge";

type ModeFilter = "all" | "university" | "clinic" | "general";

export function KnowledgeBasePageContent() {
  const t = useTranslations("KnowledgeBase");
  const { role, loading: roleLoading } = useRole();

  const [search, setSearch] = useState("");
  const [selectedMode, setSelectedMode] = useState<ModeFilter>("all");

  const {
    items,
    total,
    loading,
    error,
    refresh,
    createItem,
    updateItem,
    deleteItem,
    syncRuntime,
  } = useCanonicalKnowledge(search, selectedMode);

  // Form / Dialog state
  const [editingItem, setEditingItem] = useState<CanonicalKnowledgeItem | "new" | null>(null);
  const [deletingItem, setDeletingItem] = useState<CanonicalKnowledgeItem | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);

  // Form fields
  const [formQEn, setFormQEn] = useState("");
  const [formAEn, setFormAEn] = useState("");
  const [formQBn, setFormQBn] = useState("");
  const [formABn, setFormABn] = useState("");
  const [formMode, setFormMode] = useState<"university" | "clinic" | "general">("general");
  const [submitting, setSubmitting] = useState(false);

  // Open Edit Form
  const handleOpenEdit = (item: CanonicalKnowledgeItem) => {
    setEditingItem(item);
    setFormQEn(item.question_en);
    setFormAEn(item.answer_en);
    setFormQBn(item.question_bn);
    setFormABn(item.answer_bn);
    setFormMode(item.institution_type);
    setActionError(null);
  };

  // Open New Form
  const handleOpenNew = () => {
    setEditingItem("new");
    setFormQEn("");
    setFormAEn("");
    setFormQBn("");
    setFormABn("");
    setFormMode(selectedMode !== "all" ? selectedMode : "general");
    setActionError(null);
  };

  // Close Form
  const handleCloseForm = () => {
    setEditingItem(null);
    setActionError(null);
  };

  // Save (Create or Update)
  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!((formQEn.trim() && formAEn.trim()) || (formQBn.trim() && formABn.trim()))) {
      setActionError(
        t("questionShort") + " / " + t("answerRequired")
      );
      return;
    }

    setSubmitting(true);
    setActionError(null);
    try {
      if (editingItem === "new") {
        await createItem({
          question_en: formQEn.trim(),
          answer_en: formAEn.trim(),
          question_bn: formQBn.trim(),
          answer_bn: formABn.trim(),
          institution_type: formMode,
        });
        setActionSuccess(t("itemAdded"));
      } else if (editingItem) {
        await updateItem(editingItem.id, {
          question_en: formQEn.trim(),
          answer_en: formAEn.trim(),
          question_bn: formQBn.trim(),
          answer_bn: formABn.trim(),
          institution_type: formMode,
        });
        setActionSuccess(t("itemUpdated"));
      }
      handleCloseForm();
      setTimeout(() => setActionSuccess(null), 4000);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : t("saveFailed"));
    } finally {
      setSubmitting(false);
    }
  };

  // Delete Item
  const handleConfirmDelete = async () => {
    if (!deletingItem) return;
    setSubmitting(true);
    try {
      await deleteItem(deletingItem.id);
      setActionSuccess(t("itemDeleted"));
      setDeletingItem(null);
      setTimeout(() => setActionSuccess(null), 4000);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : t("deleteFailed"));
    } finally {
      setSubmitting(false);
    }
  };

  // Sync Runtime
  const handleSyncRuntime = async () => {
    setSyncing(true);
    setActionError(null);
    setActionSuccess(null);
    try {
      const count = await syncRuntime();
      setActionSuccess(t("syncSuccess", { count }));
      setTimeout(() => setActionSuccess(null), 5000);
    } catch {
      setActionError(t("syncFailed"));
    } finally {
      setSyncing(false);
    }
  };

  // Filter items in memory for instant responsiveness
  const filteredItems = useMemo(() => {
    return items.filter((item) => {
      if (selectedMode !== "all" && item.institution_type !== selectedMode) {
        return false;
      }
      if (search.trim()) {
        const q = search.trim().toLowerCase();
        const haystack = `${item.question_en} ${item.answer_en} ${item.question_bn} ${item.answer_bn}`.toLowerCase();
        if (!haystack.includes(q)) return false;
      }
      return true;
    });
  }, [items, selectedMode, search]);

  if (roleLoading) {
    return (
      <div className="flex justify-center py-16">
        <Loader2 className="size-8 animate-spin text-sky-500" />
      </div>
    );
  }

  if (!role) {
    return (
      <div className="flex items-center gap-2 rounded-2xl bg-red-50 p-4 text-sm text-red-700 ring-1 ring-red-200">
        <ShieldAlert className="size-5 shrink-0" />
        {t("notAllowed")}
      </div>
    );
  }

  const renderModeBadge = (type: string) => {
    if (type === "university") {
      return (
        <span className="inline-flex items-center gap-1 rounded-xl bg-blue-50 px-2.5 py-0.5 text-xs font-bold text-blue-700 ring-1 ring-blue-300">
          <GraduationCap className="size-3.5" />
          {t("university")}
        </span>
      );
    }
    if (type === "clinic") {
      return (
        <span className="inline-flex items-center gap-1 rounded-xl bg-emerald-50 px-2.5 py-0.5 text-xs font-bold text-emerald-700 ring-1 ring-emerald-300">
          <Hospital className="size-3.5" />
          {t("clinic")}
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1 rounded-xl bg-purple-50 px-2.5 py-0.5 text-xs font-bold text-purple-700 ring-1 ring-purple-300">
        <Globe className="size-3.5" />
        {t("general")}
      </span>
    );
  };

  return (
    <div className="space-y-5">
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold text-sky-950 sm:text-2xl">
              {t("title")}
            </h1>
            <span className="rounded-xl bg-sky-100 px-2.5 py-0.5 text-xs font-extrabold text-sky-800 ring-1 ring-sky-300">
              CSV
            </span>
          </div>
          <p className="text-sm text-sky-800/80">{t("subtitle")}</p>
        </div>

        {/* Header Action Buttons */}
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={handleSyncRuntime}
            disabled={syncing}
            className="flex h-10 items-center gap-2 rounded-xl border border-sky-300 bg-white px-3.5 text-xs sm:text-sm font-semibold text-sky-800 shadow-sm transition hover:bg-sky-50 disabled:opacity-60"
            title="Reload public facts into running robot runtime memory"
          >
            <RefreshCw
              className={`size-4 ${syncing ? "animate-spin text-sky-500" : "text-sky-600"}`}
            />
            {syncing ? t("syncingRobot") : t("syncRobot")}
          </button>

          {(role === "admin" || role === "superAdmin") && (
            <button
              type="button"
              onClick={handleOpenNew}
              className="flex h-10 items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-4 text-xs sm:text-sm font-semibold text-white shadow-md shadow-sky-300/60 transition hover:from-sky-600 hover:to-sky-700"
            >
              <Plus className="size-4" />
              {t("addItem")}
            </button>
          )}
        </div>
      </div>

      {/* Global Toast / Alerts */}
      {actionSuccess && (
        <div className="flex items-center gap-2 rounded-2xl bg-emerald-50 px-4 py-3 text-xs sm:text-sm font-semibold text-emerald-800 ring-1 ring-emerald-300 shadow-sm">
          <CheckCircle2 className="size-4 shrink-0 text-emerald-600" />
          <span>{actionSuccess}</span>
        </div>
      )}
      {actionError && (
        <div className="flex items-center gap-2 rounded-2xl bg-red-50 px-4 py-3 text-xs sm:text-sm font-semibold text-red-800 ring-1 ring-red-300 shadow-sm">
          <AlertCircle className="size-4 shrink-0 text-red-600" />
          <span>{actionError}</span>
        </div>
      )}

      {/* Search & Mode Filters Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-sky-200 bg-white/90 p-3 shadow-sm backdrop-blur-md">
        {/* Search Input */}
        <div className="relative min-w-[240px] flex-1 sm:max-w-md">
          <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-sky-500" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={t("search")}
            className="h-10 w-full rounded-xl border border-sky-200 bg-sky-50/50 pl-9 pr-3 text-sm text-sky-950 outline-none transition focus:border-sky-400 focus:bg-white focus:ring-2 focus:ring-sky-200"
          />
          {search && (
            <button
              type="button"
              onClick={() => setSearch("")}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 rounded-md p-1 text-slate-400 hover:text-slate-600"
            >
              <X className="size-3.5" />
            </button>
          )}
        </div>

        {/* Mode Selector Tabs */}
        <div className="flex flex-wrap items-center gap-1.5 rounded-xl bg-slate-100 p-1 ring-1 ring-slate-200">
          {(["all", "university", "clinic", "general"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setSelectedMode(m)}
              className={`rounded-lg px-3 py-1 text-xs font-bold transition ${
                selectedMode === m
                  ? "bg-white text-sky-900 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              {m === "all"
                ? t("allModes")
                : m === "university"
                ? t("university")
                : m === "clinic"
                ? t("clinic")
                : t("general")}
            </button>
          ))}
        </div>

        {/* Facts Counter */}
        <div className="text-xs font-semibold text-sky-800">
          {t("showingEntries", {
            count: filteredItems.length,
            total,
          })}
        </div>
      </div>

      {/* Loading State */}
      {loading && items.length === 0 && (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <Loader2 className="size-8 animate-spin text-sky-500" />
          <p className="mt-2 text-sm text-sky-800">Loading canonical knowledge...</p>
        </div>
      )}

      {/* Error State */}
      {!loading && error && (
        <div className="rounded-2xl border border-red-200 bg-red-50 p-6 text-center text-sm text-red-700">
          <AlertCircle className="mx-auto size-8 text-red-500" />
          <p className="mt-2 font-bold">{t("loadFailed")}</p>
          <p className="mt-1 text-xs">{error}</p>
          <button
            type="button"
            onClick={() => void refresh()}
            className="mt-3 rounded-xl bg-red-600 px-4 py-1.5 text-xs font-bold text-white shadow-sm hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && filteredItems.length === 0 && (
        <div className="rounded-3xl border border-dashed border-sky-300 bg-white/70 p-12 text-center text-sky-800">
          <BookOpen className="mx-auto size-12 text-sky-400" />
          <h3 className="mt-3 text-base font-bold text-sky-950">{t("noItems")}</h3>
          <p className="mt-1 text-xs text-sky-700">
            {search
              ? "No knowledge records match your search filter."
              : "No canonical facts recorded yet for this mode."}
          </p>
          {(role === "admin" || role === "superAdmin") && (
            <button
              type="button"
              onClick={handleOpenNew}
              className="mt-4 inline-flex items-center gap-1.5 rounded-xl bg-sky-600 px-4 py-2 text-xs font-bold text-white shadow-sm hover:bg-sky-700"
            >
              <Plus className="size-4" />
              {t("addItem")}
            </button>
          )}
        </div>
      )}

      {/* Knowledge Items Grid */}
      <div className="grid gap-4 sm:grid-cols-1 lg:grid-cols-2">
        {filteredItems.map((item, idx) => (
          <motion.div
            key={item.id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25, delay: idx * 0.03 }}
            className="flex flex-col justify-between rounded-3xl border border-sky-200 bg-white/95 p-5 shadow-sm transition hover:shadow-md backdrop-blur-sm"
          >
            <div>
              {/* Card Header: Mode & Actions */}
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-sky-100 pb-2.5">
                <div className="flex items-center gap-2">
                  <span className="rounded-lg bg-sky-100 px-2 py-0.5 text-[11px] font-extrabold text-sky-800">
                    {item.id}
                  </span>
                  {renderModeBadge(item.institution_type)}
                </div>

                {(role === "admin" || role === "superAdmin") && (
                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => handleOpenEdit(item)}
                      className="rounded-lg p-1.5 text-sky-700 hover:bg-sky-50 hover:text-sky-900"
                      title={t("edit")}
                    >
                      <Pencil className="size-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => setDeletingItem(item)}
                      className="rounded-lg p-1.5 text-slate-400 hover:bg-red-50 hover:text-red-600"
                      title={t("delete")}
                    >
                      <Trash2 className="size-3.5" />
                    </button>
                  </div>
                )}
              </div>

              {/* Bilingual Fact Content */}
              <div className="space-y-3">
                {/* English Q&A */}
                {(item.question_en || item.answer_en) && (
                  <div className="rounded-2xl bg-sky-50/50 p-3 ring-1 ring-sky-100">
                    <div className="flex items-center gap-1.5 text-[11px] font-extrabold uppercase tracking-wide text-sky-800">
                      <span>🇬🇧</span>
                      <span>English</span>
                    </div>
                    <p className="mt-1 text-sm font-bold text-sky-950">
                      {item.question_en || "—"}
                    </p>
                    <p className="mt-1 text-xs text-slate-700 whitespace-pre-wrap leading-relaxed">
                      {item.answer_en || "—"}
                    </p>
                  </div>
                )}

                {/* Bangla Q&A */}
                {(item.question_bn || item.answer_bn) && (
                  <div className="rounded-2xl bg-emerald-50/40 p-3 ring-1 ring-emerald-100">
                    <div className="flex items-center gap-1.5 text-[11px] font-extrabold uppercase tracking-wide text-emerald-800">
                      <span>🇧🇩</span>
                      <span>বাংলা</span>
                    </div>
                    <p className="mt-1 text-sm font-bold text-emerald-950">
                      {item.question_bn || "—"}
                    </p>
                    <p className="mt-1 text-xs text-slate-700 whitespace-pre-wrap leading-relaxed">
                      {item.answer_bn || "—"}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Create / Edit Dialog Modal */}
      <AnimatePresence>
        {editingItem && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm">
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-3xl border border-sky-200 bg-white p-5 sm:p-6 shadow-2xl"
            >
              <div className="flex items-center justify-between border-b border-sky-100 pb-3">
                <div className="flex items-center gap-2">
                  <div className="flex size-9 items-center justify-center rounded-2xl bg-sky-100 text-sky-700">
                    {editingItem === "new" ? <Plus className="size-5" /> : <Pencil className="size-4" />}
                  </div>
                  <div>
                    <h2 className="text-base font-bold text-sky-950 sm:text-lg">
                      {editingItem === "new" ? t("addItem") : `${t("edit")} (${editingItem.id})`}
                    </h2>
                    <p className="text-xs text-sky-700">
                      Persisted to canonical knowledge.csv
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={handleCloseForm}
                  className="rounded-xl p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
                >
                  <X className="size-5" />
                </button>
              </div>

              {actionError && (
                <div className="mt-3 flex items-center gap-2 rounded-xl bg-red-50 p-2.5 text-xs text-red-700 ring-1 ring-red-200">
                  <AlertCircle className="size-4 shrink-0 text-red-600" />
                  <span>{actionError}</span>
                </div>
              )}

              <form onSubmit={handleSave} className="mt-4 space-y-4">
                {/* Mode Selector */}
                <div>
                  <label className="text-xs font-bold text-sky-900">
                    {t("institutionType")} *
                  </label>
                  <select
                    value={formMode}
                    onChange={(e) =>
                      setFormMode(e.target.value as "university" | "clinic" | "general")
                    }
                    className="mt-1 h-10 w-full rounded-xl border border-sky-200 bg-white px-3 text-sm text-sky-950 outline-none focus:border-sky-400 focus:ring-2 focus:ring-sky-200"
                  >
                    <option value="general">{t("general")}</option>
                    <option value="university">{t("university")}</option>
                    <option value="clinic">{t("clinic")}</option>
                  </select>
                </div>

                {/* English Q&A Section */}
                <div className="rounded-2xl border border-sky-100 bg-sky-50/40 p-4 space-y-3">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-sky-900">
                    <span>🇬🇧</span>
                    <span>{t("questionEn")} / {t("answerEn")}</span>
                  </div>
                  <div>
                    <label className="text-[11px] font-semibold text-sky-800">
                      {t("questionEn")}
                    </label>
                    <input
                      type="text"
                      value={formQEn}
                      onChange={(e) => setFormQEn(e.target.value)}
                      placeholder="e.g. Where is the central library located?"
                      className="mt-1 h-10 w-full rounded-xl border border-sky-200 bg-white px-3 text-sm text-sky-950 outline-none focus:border-sky-400 focus:ring-2 focus:ring-sky-200"
                    />
                  </div>
                  <div>
                    <label className="text-[11px] font-semibold text-sky-800">
                      {t("answerEn")}
                    </label>
                    <textarea
                      rows={3}
                      value={formAEn}
                      onChange={(e) => setFormAEn(e.target.value)}
                      placeholder="e.g. The central library is located on the second floor of the academic building..."
                      className="mt-1 w-full rounded-xl border border-sky-200 bg-white p-3 text-sm text-sky-950 outline-none focus:border-sky-400 focus:ring-2 focus:ring-sky-200"
                    />
                  </div>
                </div>

                {/* Bangla Q&A Section */}
                <div className="rounded-2xl border border-emerald-100 bg-emerald-50/40 p-4 space-y-3">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-emerald-900">
                    <span>🇧🇩</span>
                    <span>{t("questionBn")} / {t("answerBn")}</span>
                  </div>
                  <div>
                    <label className="text-[11px] font-semibold text-emerald-800">
                      {t("questionBn")}
                    </label>
                    <input
                      type="text"
                      value={formQBn}
                      onChange={(e) => setFormQBn(e.target.value)}
                      placeholder="যেমন: কেন্দ্রীয় লাইব্রেরি কোথায় অবস্থিত?"
                      className="mt-1 h-10 w-full rounded-xl border border-emerald-200 bg-white px-3 text-sm text-sky-950 outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-200"
                    />
                  </div>
                  <div>
                    <label className="text-[11px] font-semibold text-emerald-800">
                      {t("answerBn")}
                    </label>
                    <textarea
                      rows={3}
                      value={formABn}
                      onChange={(e) => setFormABn(e.target.value)}
                      placeholder="যেমন: কেন্দ্রীয় লাইব্রেরি একাডেমিক ভবনের দ্বিতীয় তলায় অবস্থিত..."
                      className="mt-1 w-full rounded-xl border border-emerald-200 bg-white p-3 text-sm text-sky-950 outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-200"
                    />
                  </div>
                </div>

                {/* Form Buttons */}
                <div className="flex items-center justify-end gap-2.5 pt-2">
                  <button
                    type="button"
                    onClick={handleCloseForm}
                    disabled={submitting}
                    className="h-10 rounded-xl px-4 text-xs font-bold text-slate-600 hover:bg-slate-100"
                  >
                    {t("cancel")}
                  </button>
                  <button
                    type="submit"
                    disabled={submitting}
                    className="flex h-10 items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-5 text-xs font-bold text-white shadow-md shadow-sky-300/60 hover:from-sky-600 hover:to-sky-700 disabled:opacity-50"
                  >
                    {submitting && <Loader2 className="size-3.5 animate-spin" />}
                    {submitting ? t("saving") : t("save")}
                  </button>
                </div>
              </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Delete Confirmation Modal */}
      <AnimatePresence>
        {deletingItem && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm">
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-md rounded-3xl border border-red-200 bg-white p-6 shadow-2xl"
            >
              <div className="flex items-center gap-3 text-red-600">
                <div className="flex size-10 items-center justify-center rounded-2xl bg-red-100">
                  <Trash2 className="size-5" />
                </div>
                <h3 className="text-base font-bold text-slate-900">
                  {t("deleteConfirmTitle")}
                </h3>
              </div>

              <p className="mt-3 text-xs text-slate-600 leading-relaxed">
                {t("deleteConfirmText")}
              </p>

              <div className="mt-3 rounded-2xl bg-slate-50 p-3 text-xs ring-1 ring-slate-200">
                <span className="font-bold text-slate-800">
                  {deletingItem.id}:
                </span>{" "}
                <span className="text-slate-700">
                  {deletingItem.question_en || deletingItem.question_bn}
                </span>
              </div>

              <div className="mt-5 flex items-center justify-end gap-2.5">
                <button
                  type="button"
                  onClick={() => setDeletingItem(null)}
                  disabled={submitting}
                  className="h-9 rounded-xl px-4 text-xs font-bold text-slate-600 hover:bg-slate-100"
                >
                  {t("cancel")}
                </button>
                <button
                  type="button"
                  onClick={handleConfirmDelete}
                  disabled={submitting}
                  className="flex h-9 items-center gap-1.5 rounded-xl bg-red-600 px-4 text-xs font-bold text-white shadow-md shadow-red-200 hover:bg-red-700 disabled:opacity-50"
                >
                  {submitting && <Loader2 className="size-3.5 animate-spin" />}
                  {t("delete")}
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
