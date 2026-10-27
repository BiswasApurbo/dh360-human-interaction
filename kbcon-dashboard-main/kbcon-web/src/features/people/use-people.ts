"use client";

import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/features/auth/use-auth";
import type {
  CreatePersonInput,
  PersonProfile,
  UpdatePersonInput,
} from "./types";

export function usePeople() {
  const { user } = useAuth();
  const [people, setPeople] = useState<PersonProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const getHeaders = useCallback(async (): Promise<HeadersInit> => {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (user) {
      try {
        const token = await user.getIdToken();
        headers["Authorization"] = `Bearer ${token}`;
      } catch {
        // Fallback for emulator / unauthenticated
      }
    }
    return headers;
  }, [user]);

  const loadData = useCallback(async (isRefresh = false) => {
    if (isRefresh) {
      setLoading(true);
    }
    setError(null);
    try {
      const headers = await getHeaders();
      const res = await fetch("/api/dh360/persons", { headers });

      if (!res.ok) {
        if (res.status === 401 || res.status === 403) {
          throw new Error("Authentication required to view registered people.");
        }
        if (res.status === 503) {
          throw new Error("DH360 backend is currently unreachable.");
        }
        throw new Error(`Failed to load people (status ${res.status})`);
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
  }, [getHeaders]);

  useEffect(() => {
    let ignore = false;

    async function execute() {
      try {
        const headers: Record<string, string> = {
          "Content-Type": "application/json",
        };
        if (user) {
          try {
            const token = await user.getIdToken();
            headers["Authorization"] = `Bearer ${token}`;
          } catch {
            // Unauthenticated fallback
          }
        }
        const res = await fetch("/api/dh360/persons", { headers });
        if (!ignore) {
          if (!res.ok) {
            if (res.status === 401 || res.status === 403) {
              setError("Authentication required to view registered people.");
            } else if (res.status === 503) {
              setError("DH360 backend is currently unreachable.");
            } else {
              setError(`Failed to load people (status ${res.status})`);
            }
          } else {
            const data = await res.json();
            setPeople(Array.isArray(data) ? data : []);
          }
          setLoading(false);
        }
      } catch (err: unknown) {
        if (!ignore) {
          setError(
            err instanceof Error ? err.message : "An unexpected error occurred"
          );
          setLoading(false);
        }
      }
    }

    execute();

    return () => {
      ignore = true;
    };
  }, [user]);

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
          throw new Error("Super Admin role required to delete person profiles.");
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

  return {
    people,
    loading,
    error,
    refresh: () => loadData(true),
    createPerson,
    updatePerson,
    deletePerson,
    resetBiometrics,
  };
}
