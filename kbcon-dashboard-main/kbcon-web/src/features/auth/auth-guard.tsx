"use client";

import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { useRouter } from "@/i18n/navigation";
import { useAuth } from "./use-auth";
import { useRole } from "./use-role";
import { ForcePasswordChangeModal } from "./force-password-change-modal";

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { user, loading: authLoading } = useAuth();
  const { role, mustChangePassword, loading: roleLoading } = useRole();
  const router = useRouter();
  const [resolvedPasswordChange, setResolvedPasswordChange] = useState(false);

  useEffect(() => {
    if (!authLoading && !user) {
      router.replace("/login");
    }
  }, [authLoading, user, router]);

  if (authLoading || roleLoading || !user) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-sky-50 via-sky-100 to-sky-50">
        <Loader2 className="size-8 animate-spin text-sky-500" />
      </div>
    );
  }

  // Enforce mandatory password change for bootstrap accounts
  if (mustChangePassword && !resolvedPasswordChange) {
    return (
      <ForcePasswordChangeModal
        onPasswordChanged={() => {
          setResolvedPasswordChange(true);
        }}
      />
    );
  }

  return <>{children}</>;
}
