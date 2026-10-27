import { setRequestLocale } from "next-intl/server";
import { TelemetryContent } from "@/features/telemetry/telemetry-content";

export default async function TelemetryPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  return <TelemetryContent locale={locale} />;
}
