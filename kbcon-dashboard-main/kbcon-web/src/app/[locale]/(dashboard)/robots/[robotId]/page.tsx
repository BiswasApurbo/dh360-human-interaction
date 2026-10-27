import { setRequestLocale } from "next-intl/server";
import { RobotDetail } from "@/features/robots/robot-detail";

export default async function RobotDetailPage({
  params,
}: {
  params: Promise<{ locale: string; robotId: string }>;
}) {
  const { locale, robotId } = await params;
  setRequestLocale(locale);
  return <RobotDetail robotId={robotId} />;
}
