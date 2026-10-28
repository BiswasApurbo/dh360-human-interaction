import { setRequestLocale } from "next-intl/server";
import { AssistantPageContent } from "@/features/assistant/assistant-page-content";

export default async function AssistantPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  return <AssistantPageContent />;
}
