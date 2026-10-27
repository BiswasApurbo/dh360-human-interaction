"use client";

import { useEffect, useState } from "react";
import { collection, onSnapshot, query, where } from "firebase/firestore";
import { db } from "@/lib/firebase";

export type AdminRow = {
  uid: string;
  name: string;
  email: string;
  active: boolean;
};

export function useAdmins(enabled: boolean) {
  const [admins, setAdmins] = useState<AdminRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    const q = query(collection(db, "users"), where("role", "==", "admin"));
    const unsub = onSnapshot(
      q,
      (snap) => {
        setAdmins(
          snap.docs.map((d) => ({
            uid: d.id,
            name: d.data().name ?? "",
            email: d.data().email ?? "",
            active: d.data().active !== false,
          })),
        );
        setLoading(false);
      },
      () => {
        setError(true);
        setLoading(false);
      },
    );
    return () => unsub();
  }, [enabled]);

  return { admins, loading, error };
}
