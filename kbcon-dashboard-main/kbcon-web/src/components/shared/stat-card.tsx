"use client";

import { useEffect, type MouseEvent, type ReactNode } from "react";
import {
  animate,
  motion,
  useMotionValue,
  useSpring,
  useTransform,
} from "motion/react";
import { Bot, Wifi, WifiOff, TriangleAlert, TrendingUp } from "lucide-react";

const icons = {
  bot: Bot,
  online: Wifi,
  offline: WifiOff,
  error: TriangleAlert,
};

const sparkPaths = [
  "M0 30 L20 22 L40 26 L60 14 L80 18 L100 6",
  "M0 28 L20 30 L40 18 L60 22 L80 10 L100 12",
  "M0 20 L20 26 L40 16 L60 24 L80 14 L100 20",
  "M0 32 L20 24 L40 28 L60 12 L80 16 L100 4",
];

type StatCardProps = {
  title: string;
  value: number;
  icon: keyof typeof icons;
  tone?: "sky" | "green" | "slate" | "red"; // আর ব্যবহার হয় না, সবার এক রং
  footer?: ReactNode;
  index?: number;
};

export function StatCard({
  title,
  value,
  icon,
  footer,
  index = 0,
}: StatCardProps) {
  const Icon = icons[icon];

  // সংখ্যা ০ থেকে গুনে ওঠে
  const count = useMotionValue(0);
  const rounded = useTransform(count, (v) => Math.round(v));
  useEffect(() => {
    const controls = animate(count, value, { duration: 1.2, ease: "easeOut" });
    return () => controls.stop();
  }, [count, value]);

  // মাউসের সাথে 3D হেলে পড়া
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const rotateX = useSpring(useTransform(my, [-0.5, 0.5], [9, -9]), {
    stiffness: 200,
    damping: 20,
  });
  const rotateY = useSpring(useTransform(mx, [-0.5, 0.5], [-9, 9]), {
    stiffness: 200,
    damping: 20,
  });

  function onMove(e: MouseEvent<HTMLDivElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    mx.set((e.clientX - r.left) / r.width - 0.5);
    my.set((e.clientY - r.top) / r.height - 0.5);
  }
  function onLeave() {
    mx.set(0);
    my.set(0);
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 30, scale: 0.94 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, delay: index * 0.1, ease: "easeOut" }}
      whileHover={{ y: -8, scale: 1.03 }}
      whileTap={{ scale: 0.97 }}
      onMouseMove={onMove}
      onMouseLeave={onLeave}
      style={{ rotateX, rotateY, transformPerspective: 800 }}
      className="group relative overflow-hidden rounded-3xl border border-sky-200 bg-gradient-to-br from-sky-100 via-sky-200 to-sky-100 p-5 text-sky-900 shadow-lg shadow-sky-200/60 transition-shadow duration-300 hover:shadow-xl hover:shadow-sky-300/70"
    >
      {/* সাজানোর বৃত্ত */}
      <div className="pointer-events-none absolute -right-10 -top-10 size-40 rounded-full bg-white/50 transition-transform duration-500 group-hover:scale-125" />
      <div className="pointer-events-none absolute -bottom-12 -left-8 size-32 rounded-full bg-white/40 transition-transform duration-500 group-hover:scale-125" />

      {/* চকচকে shine */}
      <div className="pointer-events-none absolute inset-y-0 -left-1/2 w-1/2 -translate-x-full -skew-x-12 bg-white/50 transition-transform duration-700 group-hover:translate-x-[320%]" />

      <div className="relative flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-sky-800">{title}</p>
          <motion.span className="mt-2 block text-5xl font-extrabold tracking-tight text-sky-900">
            {rounded}
          </motion.span>
        </div>

        {/* ভাসমান আইকন */}
        <motion.div
          animate={{ y: [0, -6, 0] }}
          transition={{
            repeat: Infinity,
            duration: 3 + index * 0.4,
            ease: "easeInOut",
          }}
          whileHover={{ rotate: 15, scale: 1.15 }}
          className="flex size-14 shrink-0 items-center justify-center rounded-2xl bg-white/80 text-sky-700 shadow-sm ring-1 ring-sky-300"
        >
          <Icon className="size-7" />
        </motion.div>
      </div>

      {/* গ্রাফ রেখা */}
      <svg
        viewBox="0 0 100 36"
        preserveAspectRatio="none"
        className="relative mt-4 h-10 w-full"
      >
        <motion.path
          d={sparkPaths[index % sparkPaths.length]}
          fill="none"
          stroke="#0284c7"
          strokeOpacity={0.8}
          strokeWidth={2.5}
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.4, delay: 0.3 + index * 0.1 }}
        />
      </svg>
      <div className="relative mt-3 flex items-center justify-between text-xs">
        <span className="inline-flex items-center gap-1 rounded-full bg-white/70 px-2 py-0.5 font-medium text-sky-800 ring-1 ring-sky-200">
          <TrendingUp className="size-3" />
          {footer}
        </span>
      </div>
    </motion.div>
  );
}
