"use client";

import { useTranslations } from "next-intl";
import { Link, usePathname } from "@/i18n/navigation";
import {
  LayoutDashboard,
  Bot,
  Sparkles,
  UserCheck,
  BookOpen,
  Activity,
  Settings,
  Users,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  useSidebar,
} from "@/components/ui/sidebar";
import { useRole } from "@/features/auth/use-role";

const items = [
  { key: "dashboard", href: "/dashboard", icon: LayoutDashboard },
  { key: "robots", href: "/robots", icon: Bot },
  { key: "assistant", href: "/assistant", icon: Sparkles },
  { key: "people", href: "/people", icon: UserCheck },
  { key: "knowledgeBase", href: "/knowledge-base", icon: BookOpen },
  { key: "telemetry", href: "/telemetry", icon: Activity },
  { key: "settings", href: "/settings", icon: Settings },
] as const;

export function AppSidebar() {
  const t = useTranslations("Nav");
  const tc = useTranslations("Common");
  const pathname = usePathname();
  const { setOpenMobile } = useSidebar();
  const { role } = useRole();

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="border-b border-sidebar-border">
        <div className="flex h-10 items-center gap-2 px-2">
          <div className="flex size-8 shrink-0 items-center justify-center rounded-md bg-sidebar-primary text-sm font-bold text-sidebar-primary-foreground">
            K
          </div>
          <span className="truncate font-semibold group-data-[collapsible=icon]:hidden">
            {tc("appName")}
          </span>
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              {items.map(({ key, href, icon: Icon }) => (
                <SidebarMenuItem key={key}>
                  <SidebarMenuButton
                    asChild
                    isActive={pathname.startsWith(href)}
                    tooltip={t(key)}
                  >
                    <Link href={href} onClick={() => setOpenMobile(false)}>
                      <Icon />
                      <span>{t(key)}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}

              {role === "superAdmin" && (
                <SidebarMenuItem>
                  <SidebarMenuButton
                    asChild
                    isActive={pathname.startsWith("/super-admin/admins")}
                    tooltip={t("admins")}
                  >
                    <Link
                      href="/super-admin/admins"
                      onClick={() => setOpenMobile(false)}
                    >
                      <Users />
                      <span>{t("admins")}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              )}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarRail />
    </Sidebar>
  );
}
