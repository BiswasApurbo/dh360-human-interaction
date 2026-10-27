"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import { Bell, Loader2, LogIn, LogOut, Search, User } from "lucide-react";
import { Link, usePathname, useRouter } from "@/i18n/navigation";
import { useAuth } from "@/features/auth/use-auth";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { LanguageSwitcher } from "./language-switcher";

const titles = [
  { href: "/dashboard", key: "dashboard" },
  { href: "/robots", key: "robots" },
  { href: "/knowledge-base", key: "knowledgeBase" },
  { href: "/telemetry", key: "telemetry" },
  { href: "/settings", key: "settings" },
] as const;

export function Navbar() {
  const t = useTranslations("Common");
  const tn = useTranslations("Nav");
  const pathname = usePathname();
  const router = useRouter();
  const { user, loading, logout } = useAuth();
  const [loggingOut, setLoggingOut] = useState(false);

  const current = titles.find((x) => pathname.startsWith(x.href));
  const title = current ? tn(current.key) : t("appName");

  const displayName = user?.displayName || user?.email?.split("@")[0] || "";

  async function handleLogout() {
    setLoggingOut(true);
    try {
      await logout();
      router.replace("/login");
    } finally {
      setLoggingOut(false);
    }
  }

  return (
    <motion.header
      initial={{ y: -56, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="sticky top-0 z-20 flex h-16 items-center gap-3 overflow-hidden border-b border-sky-200 bg-gradient-to-r from-sky-100 via-sky-200 to-sky-100 px-3 text-sky-900 shadow-sm shadow-sky-200/60 sm:px-5"
    >
      {/* সাজানোর বৃত্ত */}
      <div className="pointer-events-none absolute -right-10 -top-14 size-40 rounded-full bg-white/50" />
      <div className="pointer-events-none absolute -bottom-16 left-1/3 size-32 rounded-full bg-white/40" />

      {/* বাঁ দিক */}
      <motion.div whileHover={{ scale: 1.08 }} whileTap={{ scale: 0.9 }}>
        <SidebarTrigger className="relative size-9 rounded-xl bg-white/70 text-sky-800 ring-1 ring-sky-300 hover:bg-white hover:text-sky-900" />
      </motion.div>

      <div className="relative min-w-0">
        <motion.h2
          key={title}
          initial={{ opacity: 0, x: -12 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3 }}
          className="truncate text-base font-bold tracking-tight text-sky-900 sm:text-lg"
        >
          {title}
        </motion.h2>
        <p className="hidden truncate text-[11px] text-sky-700 sm:block">
          {t("appName")}
        </p>
      </div>

      {/* Search */}
      <div className="relative mx-auto hidden w-full max-w-sm md:block">
        <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-sky-600" />
        <input
          type="search"
          placeholder={t("search")}
          className="h-10 w-full rounded-full border-0 bg-white/70 pl-9 pr-4 text-sm text-sky-900 outline-none ring-1 ring-sky-300 transition placeholder:text-sky-600/70 focus:bg-white focus:ring-2 focus:ring-sky-500"
        />
      </div>

      {/* ডান দিক */}
      <div className="relative ml-auto flex items-center gap-2 sm:gap-3">
        {user && (
          <motion.button
            type="button"
            aria-label={t("notifications")}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.88 }}
            className="relative flex size-10 items-center justify-center rounded-xl bg-white/70 text-sky-800 ring-1 ring-sky-300 transition-colors hover:bg-white"
          >
            <motion.span
              animate={{ rotate: [0, -14, 12, -8, 0] }}
              transition={{
                duration: 0.8,
                repeat: Infinity,
                repeatDelay: 4,
                ease: "easeInOut",
              }}
              className="inline-flex"
            >
              <Bell className="size-5" />
            </motion.span>
            <span className="absolute right-2 top-2 flex size-2.5">
              <span className="absolute inline-flex size-full animate-ping rounded-full bg-rose-300 opacity-75" />
              <span className="relative inline-flex size-2.5 rounded-full bg-rose-500 ring-2 ring-white" />
            </span>
          </motion.button>
        )}

        <LanguageSwitcher />

        {/* লোড হওয়ার সময় skeleton */}
        {loading && (
          <div className="size-10 animate-pulse rounded-full bg-white/70 ring-2 ring-sky-300" />
        )}

        {/* লগইন থাকলে: নাম + Avatar মেনু */}
        {!loading && user && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <motion.button
                type="button"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="flex items-center gap-2 rounded-full bg-white/70 py-1 pl-1 pr-1 outline-none ring-2 ring-sky-300 focus-visible:ring-4 sm:pr-3"
              >
                <Avatar className="size-8">
                  <AvatarFallback className="bg-sky-500 font-bold text-white">
                    {displayName ? (
                      displayName.charAt(0).toUpperCase()
                    ) : (
                      <User className="size-4" />
                    )}
                  </AvatarFallback>
                </Avatar>
                <span className="hidden max-w-[120px] truncate text-sm font-semibold text-sky-900 sm:block">
                  {displayName}
                </span>
              </motion.button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuLabel>
                <p className="text-xs font-normal text-muted-foreground">
                  {t("profile")}
                </p>
                <p className="truncate text-sm font-semibold">{user.email}</p>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                disabled={loggingOut}
                onSelect={(e) => {
                  e.preventDefault();
                  handleLogout();
                }}
                className="cursor-pointer text-red-600 focus:text-red-600"
              >
                {loggingOut ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <LogOut className="size-4" />
                )}
                {loggingOut ? t("loggingOut") : t("logout")}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}

        {/* লগইন না থাকলে: Login বাটন */}
        {!loading && !user && (
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Link
              href="/login"
              className="flex h-10 items-center gap-2 rounded-full bg-gradient-to-r from-sky-500 to-sky-600 px-4 text-sm font-semibold text-white shadow-md shadow-sky-300/60"
            >
              <LogIn className="size-4" />
              {t("login")}
            </Link>
          </motion.div>
        )}
      </div>
    </motion.header>
  );
}
