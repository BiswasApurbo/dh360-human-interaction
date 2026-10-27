import { setRequestLocale } from "next-intl/server";
import { RobotsPageContent } from "@/features/robots/robots-page-content";

export default async function RobotsPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  return <RobotsPageContent />;
}
