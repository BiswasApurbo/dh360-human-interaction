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
import type { RobotStatus } from "@/lib/robot-status";

export type RobotRow = {
  id: string;
  name: string;
  type: string;
  language?: "en" | "bn";
  mode: "education" | "healthcare";
  status: RobotStatus;
  lastSeen: string | null;
  assignedAdminId: string | null;
  assignedAdminName: string | null;
  config: Record<string, unknown>;
  firmwareVersion: string | null;
  knowledgeBaseId: string | null;
  deviceId?: string | null;
  deviceStatus?: "paired" | "unpaired" | null;
};

export function useRobots(role: Role) {
  const { user } = useAuth();
  const [robots, setRobots] = useState<RobotRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!user || !role) return;

    // Super Admin সব দেখবে, Admin শুধু নিজের (Rules-এর সাথে মিল রাখতে)
    const base = collection(db, "robots");
    const q: Query =
      role === "superAdmin"
        ? base
        : query(base, where("assignedAdminId", "==", user.uid));

    const unsub = onSnapshot(
      q,
      (snap) => {
        setRobots(
          snap.docs.map((d) => {
            const x = d.data();
            return {
              id: d.id,
              name: x.name ?? "",
              type: x.type ?? "",
              language: x.language === "en" ? "en" : "bn",
              mode: x.mode === "healthcare" ? "healthcare" : "education",
              status: x.status ?? "offline",
              lastSeen: typeof x.lastSeen === "string" ? x.lastSeen : null,
              assignedAdminId: x.assignedAdminId ?? null,
              assignedAdminName: x.assignedAdminName ?? null,
              config: (x.config ?? {}) as Record<string, unknown>,
              firmwareVersion:
                typeof x.firmwareVersion === "string"
                  ? x.firmwareVersion
                  : null,
              knowledgeBaseId:
                typeof x.knowledgeBaseId === "string"
                  ? x.knowledgeBaseId
                  : null,
              deviceId: typeof x.deviceId === "string" ? x.deviceId : null,
              deviceStatus:
                typeof x.deviceStatus === "string"
                  ? (x.deviceStatus as "paired" | "unpaired")
                  : null,
            };
          }),
        );
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
