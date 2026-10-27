"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { AnimatePresence, motion } from "motion/react";
import { httpsCallable } from "firebase/functions";
import { BookOpen, Loader2, Plus, RefreshCw, ShieldAlert } from "lucide-react";
import { functions } from "@/lib/firebase";
import { useAuth } from "@/features/auth/use-auth";
import { useRole } from "@/features/auth/use-role";
import { useKnowledgeBases } from "./use-knowledge";
import { KbDetail } from "./kb-detail";

export function KnowledgeBasePageContent() {
  const t = useTranslations("KnowledgeBase");
  const { user } = useAuth();
  const { role, loading: roleLoading } = useRole();
  const { kbs, loading, error } = useKnowledgeBases(role);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [name, setName] = useState("");
  const [mode, setMode] = useState<"education" | "healthcare">("education");
  const [creating, setCreating] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  async function syncRobotRuntime() {
    setSyncing(true);
    setMsg(null);
    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (user) {
        try {
          const token = await user.getIdToken();
          headers["Authorization"] = `Bearer ${token}`;
        } catch {
          // Token fetch fallback
        }
      }
      const res = await fetch("/api/dh360/knowledge/reload", {
        method: "POST",
        headers,
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      setMsg({
        ok: true,
        text: t("syncSuccess", { count: data.facts_count ?? 0 }),
      });
    } catch {
      setMsg({ ok: false, text: t("syncFailed") });
    } finally {
      setSyncing(false);
    }
  }

  async function create() {
    if (name.trim().length < 2) {
      setMsg({ ok: false, text: t("nameShort") });
      return;
    }
    setCreating(true);
    setMsg(null);
    try {
      const fn = httpsCallable<
        { name: string; mode: string },
        { kbId: string }
      >(functions, "createKnowledgeBase");
      const res = await fn({ name: name.trim(), mode });
      setName("");
      setShowForm(false);
      setSelectedId(res.data.kbId);
      setMsg({ ok: true, text: t("kbCreated") });
    } catch {
      setMsg({ ok: false, text: t("createFailed") });
    } finally {
      setCreating(false);
    }
  }

  if (roleLoading) {
    return (
      <div className="flex justify-center py-16">
        <Loader2 className="size-7 animate-spin text-sky-500" />
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

  const selected = kbs.find((k) => k.id === selectedId) ?? null;

  if (selected) {
    return (
      <KbDetail
        key={selected.id}
        kb={selected}
        role={role}
        onBack={() => setSelectedId(null)}
      />
    );
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-sky-900 sm:text-2xl">
            {t("title")}
          </h1>
          <p className="text-sm text-sky-800/80">{t("subtitle")}</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {(role === "admin" || role === "superAdmin") && (
            <motion.button
              type="button"
              onClick={syncRobotRuntime}
              disabled={syncing}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              className="flex h-10 items-center gap-2 rounded-xl border border-sky-300 bg-white px-3.5 text-sm font-semibold text-sky-800 shadow-sm hover:bg-sky-50 disabled:opacity-60"
            >
              <RefreshCw
                className={`size-4 ${syncing ? "animate-spin text-sky-500" : "text-sky-600"}`}
              />
              {syncing ? t("syncingRobot") : t("syncRobot")}
            </motion.button>
          )}
          {!showForm && (
            <motion.button
              type="button"
              onClick={() => {
                setShowForm(true);
                setMsg(null);
              }}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              className="flex h-10 items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-4 text-sm font-semibold text-white shadow-md shadow-sky-300/60"
            >
              <Plus className="size-4" />
              {t("newKb")}
            </motion.button>
          )}
        </div>
      </div>

      <AnimatePresence>
        {showForm && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="space-y-4 rounded-2xl border border-sky-200 bg-white/80 p-4 shadow-sm sm:p-5"
          >
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-1.5">
                <label className="text-sm font-medium text-sky-900">
                  {t("kbName")}
                </label>
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="h-10 w-full rounded-md border border-sky-200 bg-white px-3 text-sm text-sky-900 outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-sm font-medium text-sky-900">
                  {t("mode")}
                </label>
                <select
                  value={mode}
                  onChange={(e) =>
                    setMode(e.target.value as "education" | "healthcare")
                  }
                  className="h-10 w-full rounded-md border border-sky-200 bg-white px-3 text-sm text-sky-900 outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
                >
                  <option value="education">{t("education")}</option>
                  <option value="healthcare">{t("healthcare")}</option>
                </select>
              </div>
            </div>
            <div className="flex flex-wrap justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowForm(false)}
                className="h-10 rounded-xl px-4 text-sm font-medium text-sky-800 ring-1 ring-sky-300 hover:bg-sky-50"
              >
                {t("cancel")}
              </button>
              <motion.button
                type="button"
                onClick={create}
                disabled={creating}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                className="flex h-10 items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-4 text-sm font-semibold text-white shadow-md shadow-sky-300/60 disabled:opacity-70"
              >
                {creating && <Loader2 className="size-4 animate-spin" />}
                {creating ? t("creating") : t("createKb")}
              </motion.button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {msg && (
        <p
          className={`rounded-lg px-3 py-2 text-sm ring-1 ${
            msg.ok
              ? "bg-emerald-50 text-emerald-700 ring-emerald-200"
              : "bg-red-50 text-red-600 ring-red-200"
          }`}
        >
          {msg.text}
        </p>
      )}

      {loading && (
        <div className="flex justify-center py-10">
          <Loader2 className="size-6 animate-spin text-sky-500" />
        </div>
      )}
      {!loading && error && (
        <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 ring-1 ring-red-200">
          {t("loadFailed")}
        </p>
      )}
      {!loading && !error && kbs.length === 0 && (
        <p className="py-8 text-center text-sm text-sky-700">{t("emptyKbs")}</p>
      )}

      {kbs.length > 0 && (
        <>
          <p className="text-xs text-sky-700">{t("selectHint")}</p>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {kbs.map((k, i) => (
              <motion.button
                key={k.id}
                type="button"
                onClick={() => setSelectedId(k.id)}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: i * 0.05 }}
                whileHover={{ y: -4, scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="rounded-3xl border border-sky-200 bg-gradient-to-br from-sky-100 via-sky-200 to-sky-100 p-5 text-left text-sky-900 shadow-lg shadow-sky-200/60"
              >
                <div className="mb-3 flex items-center gap-3">
                  <span className="flex size-10 items-center justify-center rounded-xl bg-white/80 text-sky-700 ring-1 ring-sky-300">
                    <BookOpen className="size-5" />
                  </span>
                  <div className="min-w-0">
                    <p className="truncate text-base font-bold">{k.name}</p>
                    <p className="text-xs text-sky-700">{t(k.mode)}</p>
                  </div>
                </div>
                <p className="text-xs text-sky-800">
                  {k.itemCount} {t("itemCount")} · {t("version")} {k.version}
                </p>
              </motion.button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
