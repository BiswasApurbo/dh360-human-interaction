"use client";

import { useLocale } from "next-intl";
import { motion } from "motion/react";
import { usePathname, useRouter } from "@/i18n/navigation";

const langs = [
  { code: "en", label: "EN" },
  { code: "bn", label: "বাং" },
] as const;

export function LanguageSwitcher() {
  const locale = useLocale();
  const router = useRouter();
  const pathname = usePathname();

  return (
    <div className="flex items-center rounded-full border border-sky-200 bg-white p-0.5 shadow-sm">
      {langs.map((l) => {
        const active = locale === l.code;
        return (
          <motion.button
            key={l.code}
            type="button"
            whileHover={{ scale: 1.06 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => router.replace(pathname, { locale: l.code })}
            className="relative z-10 rounded-full px-3 py-1 text-xs font-semibold"
          >
            {active && (
              <motion.span
                layoutId="lang-pill"
                className="absolute inset-0 -z-10 rounded-full bg-gradient-to-r from-sky-400 to-sky-600 shadow-md"
                transition={{ type: "spring", stiffness: 400, damping: 30 }}
              />
            )}
            <span
              className={`transition-colors duration-200 ${
                active ? "text-white" : "text-slate-500"
              }`}
            >
              {l.label}
            </span>
          </motion.button>
        );
      })}
    </div>
  );
}
