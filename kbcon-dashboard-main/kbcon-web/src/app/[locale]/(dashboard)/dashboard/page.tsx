import { getTranslations, setRequestLocale } from "next-intl/server";
import { DashboardStats } from "@/features/dashboard/dashboard-stats";
import { DashboardCharts } from "@/features/dashboard/dashboard-charts";
import { RecentActivity } from "@/features/dashboard/recent-activity";

export default async function DashboardPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("Dashboard");

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold sm:text-2xl">{t("title")}</h1>
        <p className="text-sm text-muted-foreground">{t("subtitle")}</p>
      </div>

      <DashboardStats />
      <DashboardCharts />
      <RecentActivity />
    </div>
  );
}
