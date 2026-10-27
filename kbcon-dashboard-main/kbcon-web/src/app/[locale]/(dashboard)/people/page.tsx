import { setRequestLocale } from "next-intl/server";
import { PeoplePageContent } from "@/features/people/people-page-content";

export default async function PeoplePage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  return <PeoplePageContent />;
}
