"use client";

import { useEffect, useState } from "react";
import {
  collection,
  onSnapshot,
  query,
  where,
  type Query,
} from "firebase/firestore";
import { db } from "@/lib/firebase";
import { useAuth } from "@/features/auth/use-auth";
import type { Role } from "@/features/auth/use-role";

export type KbRow = {
  id: string;
  name: string;
  mode: "education" | "healthcare";
  version: number;
  itemCount: number;
  robotIds: string[];
};

export type ItemRow = {
  id: string;
  question: string;
  answer: string;
  language: "bn" | "en";
  category: string;
  createdAt: string;
};

export type HistoryRow = {
  id: string;
  fileName: string;
  total: number;
  imported: number;
  skipped: number;
  at: string;
};

export function useKnowledgeBases(role: Role) {
  const { user } = useAuth();
  const [kbs, setKbs] = useState<KbRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!user || !role) return;
    const base = collection(db, "knowledgeBases");
    // Admin শুধু নিজেরটা (Rules-এর সাথে মিল রাখতে)
    const q: Query =
      role === "superAdmin"
        ? base
        : query(base, where("ownerAdminId", "==", user.uid));

    const unsub = onSnapshot(
      q,
      (snap) => {
        const rows = snap.docs.map((d) => {
          const x = d.data();
          return {
            id: d.id,
            name: x.name ?? "",
            mode: x.mode === "healthcare" ? "healthcare" : "education",
            version: Number(x.version ?? 1),
            itemCount: Number(x.itemCount ?? 0),
            robotIds: Array.isArray(x.robotIds) ? x.robotIds : [],
          } as KbRow;
        });
        rows.sort((a, b) => a.name.localeCompare(b.name));
        setKbs(rows);
        setError(false);
        setLoading(false);
      },
      () => {
        setError(true);
        setLoading(false);
      },
    );
    return () => unsub();
  }, [user, role]);

  return { kbs, loading, error };
}

export function useKbItems(kbId: string) {
  const [items, setItems] = useState<ItemRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    const unsub = onSnapshot(
      collection(db, "knowledgeBases", kbId, "items"),
      (snap) => {
        const rows = snap.docs.map((d) => {
          const x = d.data();
          return {
            id: d.id,
            question: x.question ?? "",
            answer: x.answer ?? "",
            language: x.language === "bn" ? "bn" : "en",
            category: x.category ?? "",
            createdAt: x.createdAt ?? "",
          } as ItemRow;
        });
        rows.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
        setItems(rows);
        setError(false);
        setLoading(false);
      },
      () => {
        setError(true);
        setLoading(false);
      },
    );
    return () => unsub();
  }, [kbId]);

  return { items, loading, error };
}

export function useImportHistory(kbId: string, role: Role) {
  const { user } = useAuth();
  const [history, setHistory] = useState<HistoryRow[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user || !role) return;
    const base = collection(db, "importHistory");
    const q =
      role === "superAdmin"
        ? query(base, where("kbId", "==", kbId))
        : query(base, where("kbId", "==", kbId), where("by", "==", user.uid));

    const unsub = onSnapshot(
      q,
      (snap) => {
        const rows = snap.docs.map((d) => {
          const x = d.data();
          return {
            id: d.id,
            fileName: x.fileName ?? "",
            total: Number(x.total ?? 0),
            imported: Number(x.imported ?? 0),
            skipped: Number(x.skipped ?? 0),
            at: x.at ?? "",
          } as HistoryRow;
        });
        rows.sort((a, b) => b.at.localeCompare(a.at));
        setHistory(rows);
        setLoading(false);
      },
      () => setLoading(false),
    );
    return () => unsub();
  }, [user, role, kbId]);

  return { history, loading };
}
export type ChangeRow = {
  id: string;
  by: string;
  action: "create" | "update" | "delete";
  question: string;
  at: string;
};

export function useKbChanges(kbId: string) {
  const [changes, setChanges] = useState<ChangeRow[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsub = onSnapshot(
      collection(db, "knowledgeBases", kbId, "changes"),
      (snap) => {
        const rows = snap.docs.map((d) => {
          const x = d.data();
          return {
            id: d.id,
            by: String(x.by ?? ""),
            action: x.action,
            question: String(x.question ?? ""),
            at: String(x.at ?? ""),
          } as ChangeRow;
        });
        rows.sort((a, b) => b.at.localeCompare(a.at));
        setChanges(rows.slice(0, 50));
        setLoading(false);
      },
      () => setLoading(false),
    );
    return () => unsub();
  }, [kbId]);

  return { changes, loading };
}

export type CanonicalKnowledgeItem = {
  id: string;
  index: number;
  question_en: string;
  answer_en: string;
  question_bn: string;
  answer_bn: string;
  institution_type: "university" | "clinic" | "general";
};

export function useCanonicalKnowledge(initialQuery = "", initialType = "all") {
  const { user, loading: authLoading } = useAuth();
  const [items, setItems] = useState<CanonicalKnowledgeItem[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const getAuthHeaders = async () => {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (user) {
      try {
        const token = await user.getIdToken();
        headers["Authorization"] = `Bearer ${token}`;
      } catch (err) {
        console.warn("[useCanonicalKnowledge] Failed to obtain Firebase ID token:", err);
      }
    }
    return headers;
  };

  const loadData = async (queryStr = initialQuery, typeStr = initialType) => {
    if (authLoading) return;
    setLoading(true);
    setError(null);
    try {
      const headers = await getAuthHeaders();
      const params = new URLSearchParams();
      if (queryStr.trim()) params.set("query", queryStr.trim());
      if (typeStr && typeStr !== "all") params.set("institution_type", typeStr);

      const res = await fetch(`/api/dh360/knowledge/items?${params.toString()}`, {
        headers,
      });

      if (!res.ok) {
        throw new Error(`Failed to load knowledge items: HTTP ${res.status}`);
      }

      const data = await res.json();
      setItems(data.items ?? []);
      setTotal(data.total ?? 0);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Error fetching knowledge items";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!authLoading) {
      void loadData(initialQuery, initialType);
    }
  }, [authLoading, initialQuery, initialType]);

  const createItem = async (payload: {
    question_en: string;
    answer_en: string;
    question_bn: string;
    answer_bn: string;
    institution_type: string;
  }) => {
    const headers = await getAuthHeaders();
    const res = await fetch("/api/dh360/knowledge/items", {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Failed to create item: HTTP ${res.status}`);
    }
    const created: CanonicalKnowledgeItem = await res.json();
    await loadData();
    return created;
  };

  const updateItem = async (
    id: string,
    payload: {
      question_en?: string;
      answer_en?: string;
      question_bn?: string;
      answer_bn?: string;
      institution_type?: string;
    }
  ) => {
    const headers = await getAuthHeaders();
    const res = await fetch(`/api/dh360/knowledge/items/${id}`, {
      method: "PUT",
      headers,
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Failed to update item: HTTP ${res.status}`);
    }
    const updated: CanonicalKnowledgeItem = await res.json();
    await loadData();
    return updated;
  };

  const deleteItem = async (id: string) => {
    const headers = await getAuthHeaders();
    const res = await fetch(`/api/dh360/knowledge/items/${id}`, {
      method: "DELETE",
      headers,
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Failed to delete item: HTTP ${res.status}`);
    }
    await loadData();
  };

  const syncRuntime = async () => {
    const headers = await getAuthHeaders();
    const res = await fetch("/api/dh360/knowledge/reload", {
      method: "POST",
      headers,
    });
    if (!res.ok) {
      throw new Error(`Failed to sync runtime: HTTP ${res.status}`);
    }
    const data = await res.json();
    await loadData();
    return data.facts_count as number;
  };

  return {
    items,
    total,
    loading,
    error,
    refresh: loadData,
    createItem,
    updateItem,
    deleteItem,
    syncRuntime,
  };
}
