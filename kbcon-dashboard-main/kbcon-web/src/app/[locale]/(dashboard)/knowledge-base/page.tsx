import { setRequestLocale } from "next-intl/server";
import { KnowledgeBasePageContent } from "@/features/knowledge-base/knowledge-base-page-content";

export default async function KnowledgeBasePage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  return <KnowledgeBasePageContent />;
}
