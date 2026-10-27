"use client";

import { useEffect, useState } from "react";
import {
  collection,
  onSnapshot,
  orderBy,
  query,
  where,
  type Query,
} from "firebase/firestore";
import { db } from "@/lib/firebase";
import { useAuth } from "@/features/auth/use-auth";
import type { Role } from "@/features/auth/use-role";

export type Activity = "idle" | "speaking" | "listening" | "charging";

export type TelemetryRobot = {
  id: string;
  name: string;
  lastSeen: string | null;
  hasData: boolean;
  battery: number;
  temperature: number;
  activity: Activity;
  errorCode: string | null;
  config: Record<string, unknown>;
};

export type HistoryPoint = {
  id: string;
  timestamp: string;
  battery: number;
  temperature: number;
};

// Robot তালিকা ও সর্বশেষ অবস্থা (robots ডকুমেন্টের telemetry ফিল্ড)
export function useTelemetryRobots(role: Role) {
  const { user } = useAuth();
  const [robots, setRobots] = useState<TelemetryRobot[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!user || !role) return;
    const base = collection(db, "robots");
    const q: Query =
      role === "superAdmin"
        ? base
        : query(base, where("assignedAdminId", "==", user.uid));

    const unsub = onSnapshot(
      q,
      (snap) => {
        const rows = snap.docs.map((d) => {
          const x = d.data();
          const tel = x.telemetry as Record<string, unknown> | undefined;
          return {
            id: d.id,
            name: x.name ?? "",
            lastSeen: typeof x.lastSeen === "string" ? x.lastSeen : null,
            hasData: !!tel,
            battery: Number(tel?.battery ?? 0),
            temperature: Number(tel?.temperature ?? 0),
            activity: (tel?.activity as Activity) ?? "idle",
            errorCode:
              typeof tel?.errorCode === "string" ? tel.errorCode : null,
            config: (x.config ?? {}) as Record<string, unknown>,
          } as TelemetryRobot;
        });
        rows.sort((a, b) => a.id.localeCompare(b.id));
        setRobots(rows);
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

  return { robots, loading, error };
}

// নির্বাচিত Robot-এর ইতিহাস (hours ঘণ্টা আগে থেকে এখন পর্যন্ত)
// নির্বাচিত Robot-এর ইতিহাস (fromIso থেকে toIso পর্যন্ত)
export function useTelemetryHistory(
  robotId: string | null,
  fromIso: string,
  toIso: string,
) {
  const [points, setPoints] = useState<HistoryPoint[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!robotId) return;
    const q = query(
      collection(db, "robots", robotId, "telemetry"),
      where("timestamp", ">=", fromIso),
      where("timestamp", "<=", toIso),
      orderBy("timestamp", "asc"),
    );
    const unsub = onSnapshot(
      q,
      (snap) => {
        setPoints(
          snap.docs.map((d) => {
            const x = d.data();
            return {
              id: d.id,
              timestamp: String(x.timestamp ?? ""),
              battery: Number(x.battery ?? 0),
              temperature: Number(x.temperature ?? 0),
            };
          }),
        );
        setLoading(false);
      },
      () => {
        setPoints([]);
        setLoading(false);
      },
    );
    return () => unsub();
  }, [robotId, fromIso, toIso]);

  return { points: robotId ? points : [], loading };
}
