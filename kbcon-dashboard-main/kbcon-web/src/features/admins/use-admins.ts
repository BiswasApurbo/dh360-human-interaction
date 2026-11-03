"use client";

import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/features/auth/use-auth";

export type AdminRow = {
  uid: string;
  name: string;
  email: string;
  role: "superAdmin" | "admin";
  active: boolean;
  mustChangePassword?: boolean;
  assignedRobots?: string[];
  permissions: {
    dashboard: boolean;
    robots: boolean;
    assistant: boolean;
    people: boolean;
    knowledgeBase: boolean;
    telemetry: boolean;
    settings: boolean;
  };
  createdAt?: string;
  lastSignInTime?: string;
};

export function useAdmins(enabled: boolean) {
  const { user } = useAuth();
  const [admins, setAdmins] = useState<AdminRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAdmins = useCallback(async () => {
    if (!enabled || !user) return;
    setLoading(true);
    setError(null);
    try {
      const token = await user.getIdToken();
      const res = await fetch("/api/admin/users", {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.message || `Failed to load administrators (HTTP ${res.status})`);
      }
      const data = await res.json();
      setAdmins(data.users || []);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load administrators");
    } finally {
      setLoading(false);
    }
  }, [enabled, user]);

  useEffect(() => {
    void fetchAdmins();
  }, [fetchAdmins]);

  return { admins, loading, error, refetch: fetchAdmins };
}
