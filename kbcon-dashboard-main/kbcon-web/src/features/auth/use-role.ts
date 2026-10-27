"use client";

import { useEffect, useState } from "react";
import { useAuth } from "./use-auth";

export type Role = "superAdmin" | "admin" | null;

export function useRole() {
  const { user, loading } = useAuth();
  const [role, setRole] = useState<Role>(null);
  const [roleLoading, setRoleLoading] = useState(true);

  useEffect(() => {
    if (loading) return;
    if (!user) {
      setRole(null);
      setRoleLoading(false);
      return;
    }
    user
      .getIdTokenResult()
      .then((result) => {
        const r = result.claims.role;
        setRole(r === "superAdmin" || r === "admin" ? r : null);
      })
      .finally(() => setRoleLoading(false));
  }, [user, loading]);

  return { role, loading: loading || roleLoading };
}
