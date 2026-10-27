import { setRequestLocale } from "next-intl/server";
import { AdminsPageContent } from "@/features/admins/admins-page-content";

export default async function AdminsPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  return <AdminsPageContent />;
}
