"use client";

import type { ReactNode } from "react";
import { motion } from "motion/react";

export function LoginCard({ children }: { children: ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30, scale: 0.96 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="relative w-full max-w-md rounded-3xl border border-sky-200 bg-gradient-to-br from-white via-sky-50 to-sky-100 p-6 shadow-xl shadow-sky-200/70 sm:p-8"
    >
      {children}
    </motion.div>
  );
}
