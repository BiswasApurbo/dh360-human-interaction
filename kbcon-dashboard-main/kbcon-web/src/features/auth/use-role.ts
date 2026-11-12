"use client";

import { useEffect, useState } from "react";
import { useAuth } from "./use-auth";

export type Role = "superAdmin" | "admin" | null;

export interface AdminPermissions {
  dashboard: boolean;
  robots: boolean;
  assistant: boolean;
  people: boolean;
  knowledgeBase: boolean;
  telemetry: boolean;
  settings: boolean;
}

export const DEFAULT_SUPER_ADMIN_PERMISSIONS: AdminPermissions = {
  dashboard: true,
  robots: true,
  assistant: true,
  people: true,
  knowledgeBase: true,
  telemetry: true,
  settings: true,
};

export const DEFAULT_ADMIN_PERMISSIONS: AdminPermissions = {
  dashboard: true,
  robots: true,
  assistant: true,
  people: true,
  knowledgeBase: true,
  telemetry: true,
  settings: true,
};

export function useRole() {
  const { user, loading } = useAuth();
  const [role, setRole] = useState<Role>(null);
  const [mustChangePassword, setMustChangePassword] = useState(false);
  const [permissions, setPermissions] = useState<AdminPermissions>(
    DEFAULT_ADMIN_PERMISSIONS
  );
  const [roleLoading, setRoleLoading] = useState(true);

  useEffect(() => {
    if (loading) return;
    if (!user) {
      const id = setTimeout(() => {
        setRole(null);
        setMustChangePassword(false);
        setPermissions(DEFAULT_ADMIN_PERMISSIONS);
        setRoleLoading(false);
      }, 0);
      return () => clearTimeout(id);
    }
    user
      .getIdTokenResult(true)
      .then((result) => {
        const r =
          result.claims.role ||
          (user.email === "26digital360@gmail.com" ||
            user.email === "apurbobiswas32@gmail.com" ||
            user.email === "superadmin@gmail.com"
            ? "superAdmin"
            : null);
        const validRole = r === "superAdmin" || r === "admin" ? r : null;
        setRole(validRole);
        const sessionResolved =
          typeof window !== "undefined" &&
          sessionStorage.getItem(`mustChangePasswordResolved_${user.uid}`) === "true";
        setMustChangePassword(
          Boolean(result.claims.mustChangePassword) && !sessionResolved
        );
        if (validRole === "superAdmin") {
          setPermissions(DEFAULT_SUPER_ADMIN_PERMISSIONS);
        } else if (
          result.claims.permissions &&
          typeof result.claims.permissions === "object"
        ) {
          setPermissions({
            ...DEFAULT_ADMIN_PERMISSIONS,
            ...(result.claims.permissions as Partial<AdminPermissions>),
          });
        } else {
          setPermissions(DEFAULT_ADMIN_PERMISSIONS);
        }
      })
      .finally(() => setRoleLoading(false));
  }, [user, loading]);

  return {
    role,
    mustChangePassword,
    permissions,
    loading: loading || roleLoading,
  };
}
