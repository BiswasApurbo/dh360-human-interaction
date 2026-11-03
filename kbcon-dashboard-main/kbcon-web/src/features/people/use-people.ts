"use client";

import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/features/auth/use-auth";
import { useRole } from "@/features/auth/use-role";
import type {
  ConsentAuditRecord,
  CreatePersonInput,
  PersonProfile,
  UpdatePersonInput,
} from "./types";

export function usePeople(robotId?: string) {
  const { user, loading: authLoading } = useAuth();
  const { role, loading: roleLoading } = useRole();
  const [people, setPeople] = useState<PersonProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const getHeaders = useCallback(async (): Promise<HeadersInit> => {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (robotId) {
      headers["X-Robot-ID"] = robotId;
    }
    if (user) {
      try {
        const token = await user.getIdToken();
        headers["Authorization"] = `Bearer ${token}`;
      } catch (err) {
        console.warn("[usePeople] Failed to obtain Firebase ID token:", err);
      }
    }
    return headers;
  }, [user, robotId]);

  const loadData = useCallback(async (isRefresh = false) => {
    if (authLoading || roleLoading) return;
    if (role === "superAdmin") {
      setPeople([]);
      setLoading(false);
      return;
    }
    if (!user) {
      setPeople([]);
      setLoading(false);
      return;
    }
    if (isRefresh) {
      setLoading(true);
    }
    setError(null);
    try {
      const headers = await getHeaders();
      const url = robotId
        ? `/api/dh360/persons?robot_id=${encodeURIComponent(robotId)}`
        : "/api/dh360/persons";
      const res = await fetch(url, { headers });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        if (res.status === 401) {
          throw new Error("Authentication required. Please check your login session.");
        }
        if (res.status === 403) {
          throw new Error("Access forbidden: SuperAdmin or Admin role required.");
        }
        if (res.status === 502) {
          throw new Error(errData?.message || "Gateway error: Backend rejected operator token.");
        }
        if (res.status === 503) {
          throw new Error(errData?.message || errData?.detail || "DH360 backend service is currently unreachable (HTTP 503). Ensure backend is running on port 8000.");
        }
        throw new Error(errData?.detail || errData?.message || `Failed to load people (status ${res.status})`);
      }

      const data = await res.json();
      setPeople(Array.isArray(data) ? data : []);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "An unexpected error occurred";
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [authLoading, roleLoading, role, user, robotId, getHeaders]);

  useEffect(() => {
    if (authLoading || roleLoading) return;
    if (!user || role === "superAdmin") {
      const id = setTimeout(() => {
        setPeople([]);
        setLoading(false);
      }, 0);
      return () => clearTimeout(id);
    }
    const id = setTimeout(() => {
      void loadData(false);
    }, 0);
    return () => clearTimeout(id);
  }, [authLoading, roleLoading, role, user, loadData]);

  const createPerson = useCallback(
    async (input: CreatePersonInput): Promise<PersonProfile> => {
      const headers = await getHeaders();
      const res = await fetch("/api/dh360/persons", {
        method: "POST",
        headers,
        body: JSON.stringify(input),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(
          errData?.detail || `Failed to create person (status ${res.status})`
        );
      }

      const created: PersonProfile = await res.json();
      setPeople((prev) => [created, ...prev]);
      return created;
    },
    [getHeaders]
  );

  const updatePerson = useCallback(
    async (
      personId: string,
      input: UpdatePersonInput
    ): Promise<PersonProfile> => {
      const headers = await getHeaders();
      const res = await fetch(`/api/dh360/persons/${personId}`, {
        method: "PATCH",
        headers,
        body: JSON.stringify(input),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(
          errData?.detail || `Failed to update person (status ${res.status})`
        );
      }

      const updated: PersonProfile = await res.json();
      setPeople((prev) =>
        prev.map((p) => (p.person_id === personId ? updated : p))
      );
      return updated;
    },
    [getHeaders]
  );

  const deletePerson = useCallback(
    async (personId: string): Promise<void> => {
      const headers = await getHeaders();
      const res = await fetch(`/api/dh360/persons/${personId}`, {
        method: "DELETE",
        headers,
      });

      if (!res.ok) {
        if (res.status === 403) {
          throw new Error("You are not authorized to delete this person profile.");
        }
        const errData = await res.json().catch(() => null);
        throw new Error(
          errData?.detail || `Failed to delete person (status ${res.status})`
        );
      }

      setPeople((prev) => prev.filter((p) => p.person_id !== personId));
    },
    [getHeaders]
  );

  const resetBiometrics = useCallback(
    async (personId: string): Promise<void> => {
      const headers = await getHeaders();
      const res = await fetch(`/api/dh360/persons/${personId}/biometrics`, {
        method: "DELETE",
        headers,
      });

      if (!res.ok) {
        if (res.status === 403) {
          throw new Error("Super Admin role required to reset biometric templates.");
        }
        const errData = await res.json().catch(() => null);
        throw new Error(
          errData?.detail || `Failed to reset biometrics (status ${res.status})`
        );
      }

      await loadData(true);
    },
    [getHeaders, loadData]
  );

  const updateConsent = useCallback(
    async (
      personId: string,
      consentStatus: string,
      notes?: string
    ): Promise<{ person: PersonProfile; audit: ConsentAuditRecord }> => {
      const headers = await getHeaders();
      const res = await fetch(`/api/dh360/persons/${personId}/consent`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          consent_status: consentStatus,
          notes: notes || null,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(
          errData?.detail || `Failed to update consent (status ${res.status})`
        );
      }

      const result = await res.json();
      const updatedPerson: PersonProfile = result.person;
      setPeople((prev) =>
        prev.map((p) => (p.person_id === personId ? updatedPerson : p))
      );
      return result;
    },
    [getHeaders]
  );

  const getConsentHistory = useCallback(
    async (personId: string): Promise<ConsentAuditRecord[]> => {
      const headers = await getHeaders();
      const res = await fetch(`/api/dh360/persons/${personId}/consent-history`, {
        headers,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(
          errData?.detail || `Failed to load consent history (status ${res.status})`
        );
      }

      return (await res.json()) as ConsentAuditRecord[];
    },
    [getHeaders]
  );

  return {
    people,
    loading,
    error,
    refresh: () => loadData(true),
    createPerson,
    updatePerson,
    updateConsent,
    getConsentHistory,
    deletePerson,
    resetBiometrics,
  };
}
