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
  const [error, setError] = useState(false);

  const fetchAdmins = useCallback(async () => {
    if (!enabled || !user) return;
    setLoading(true);
    setError(false);
    try {
      const token = await user.getIdToken();
      const res = await fetch("/api/admin/users", {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      setAdmins(data.users || []);
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  }, [enabled, user]);

  useEffect(() => {
    void fetchAdmins();
  }, [fetchAdmins]);

  return { admins, loading, error, refetch: fetchAdmins };
}
