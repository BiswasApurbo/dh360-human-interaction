import { Suspense } from "react";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { KeyRound, Loader2 } from "lucide-react";
import { AuthActionContent } from "@/features/auth/auth-action-content";
import { LoginCard } from "@/features/auth/login-card";
import { LanguageSwitcher } from "@/components/layout/language-switcher";

export default async function AuthActionPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("AuthAction");
  const tc = await getTranslations("Common");

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-gradient-to-br from-sky-100 via-sky-50 to-sky-200 p-4">
      <div className="pointer-events-none absolute -right-24 top-10 size-80 rounded-full bg-sky-300/40 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-24 left-10 size-80 rounded-full bg-cyan-300/30 blur-3xl" />

      <div className="absolute right-4 top-4">
        <LanguageSwitcher />
      </div>

      <LoginCard>
        <div className="mb-6 flex flex-col items-center text-center">
          <div className="mb-3 flex size-14 items-center justify-center rounded-2xl bg-white text-sky-700 shadow-sm ring-1 ring-sky-300">
            <KeyRound className="size-7" />
          </div>
          <p className="text-xs font-semibold uppercase tracking-widest text-sky-700">
            {tc("appName")}
          </p>
          <h1 className="mt-2 text-2xl font-bold text-sky-900">{t("title")}</h1>
          <p className="text-sm text-sky-800/80">{t("subtitle")}</p>
        </div>

        <Suspense
          fallback={
            <div className="flex flex-col items-center justify-center py-8">
              <Loader2 className="size-8 animate-spin text-sky-500 mb-2" />
              <p className="text-xs text-slate-500">Loading KBCON security module...</p>
            </div>
          }
        >
          <AuthActionContent />
        </Suspense>
      </LoginCard>
    </div>
  );
}
