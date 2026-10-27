"use client";

import { useEffect } from "react";
import { Loader2 } from "lucide-react";
import { useRouter } from "@/i18n/navigation";
import { useAuth } from "./use-auth";

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) {
      router.replace("/login");
    }
  }, [loading, user, router]);

  if (loading || !user) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-sky-50 via-sky-100 to-sky-50">
        <Loader2 className="size-8 animate-spin text-sky-500" />
      </div>
    );
  }

  return <>{children}</>;
}
