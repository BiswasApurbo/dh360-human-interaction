"use client";

import { useState } from "react";
import { motion } from "motion/react";
import { Cpu, Link2, Loader2, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { auth } from "@/lib/firebase";
import type { RobotRow } from "./use-robots";

export function PairDeviceForm({
  robots,
  preselectedRobotId,
  onDone,
  onCancel,
}: {
  robots: RobotRow[];
  preselectedRobotId?: string | null;
  onDone: (msg: string) => void;
  onCancel: () => void;
}) {
  const [code, setCode] = useState("");
  const [selectedRobotId, setSelectedRobotId] = useState(
    preselectedRobotId || (robots[0]?.id ?? "")
  );
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Auto-format uppercase and remove unwanted characters
  function handleCodeChange(e: React.ChangeEvent<HTMLInputElement>) {
    const val = e.target.value.toUpperCase().replace(/[^A-Z0-9-]/g, "");
    setCode(val);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    const cleanCode = code.trim();
    if (!cleanCode) {
      setError("Please enter the pairing code displayed on the robot device.");
      return;
    }

    if (!selectedRobotId) {
      setError("Please select the target robot to pair with this device.");
      return;
    }

    setSubmitting(true);
    try {
      const token = await auth.currentUser?.getIdToken(true);
      if (!token) {
        setError("User authentication session expired. Please sign in again.");
        return;
      }

      const res = await fetch("/api/admin/devices/pair", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          code: cleanCode,
          robotId: selectedRobotId,
        }),
      });

      const data = await res.json().catch(() => ({}));
      if (res.ok && data.success) {
        onDone(data.message || `Device successfully paired to ${selectedRobotId}!`);
        return;
      }

      setError(data.message || data.error || "Device pairing failed. Check the code.");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Pairing request failed.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="space-y-4 rounded-2xl border border-sky-300 bg-sky-50/70 p-4 shadow-sm sm:p-5"
    >
      <div className="flex items-center justify-between border-b border-sky-200 pb-3">
        <div className="flex items-center gap-2 text-sky-950 font-bold text-base">
          <Cpu className="size-5 text-sky-600" />
          <span>Pair Physical Robot Device</span>
        </div>
        <button
          type="button"
          onClick={onCancel}
          className="rounded-lg p-1 text-slate-500 hover:bg-sky-200/50"
        >
          <X className="size-4" />
        </button>
      </div>

      <p className="text-xs text-sky-800">
        Run <code className="bg-white px-1.5 py-0.5 rounded font-mono text-sky-900 border border-sky-200">kbcon-device pair</code> on the Raspberry Pi 5 to generate a temporary 10-minute pairing code.
      </p>

      {error && (
        <p className="rounded-lg bg-red-50 p-2.5 text-xs font-semibold text-red-600 ring-1 ring-red-200">
          {error}
        </p>
      )}

      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-1.5">
          <Label htmlFor="device-code" className="text-sky-900 font-medium text-xs">
            Pairing Code (from Raspberry Pi)
          </Label>
          <Input
            id="device-code"
            placeholder="XXXX-XXXX-XXXX"
            value={code}
            onChange={handleCodeChange}
            maxLength={14}
            autoComplete="off"
            className="h-10 border-sky-300 bg-white font-mono text-sm tracking-wider uppercase"
          />
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="target-robot" className="text-sky-900 font-medium text-xs">
            Target Robot
          </Label>
          <select
            id="target-robot"
            value={selectedRobotId}
            onChange={(e) => setSelectedRobotId(e.target.value)}
            className="flex h-10 w-full rounded-md border border-sky-300 bg-white px-3 py-2 text-sm text-sky-900 shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-sky-500"
          >
            {robots.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name} ({r.id}) {r.deviceId ? "• [Has Device]" : "• [Unpaired]"}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-end gap-2 pt-2">
        <button
          type="button"
          onClick={onCancel}
          disabled={submitting}
          className="inline-flex h-9 items-center rounded-xl px-4 text-xs font-semibold text-sky-800 ring-1 ring-sky-300 hover:bg-sky-100/60"
        >
          Cancel
        </button>
        <motion.button
          type="submit"
          disabled={submitting}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="inline-flex h-9 items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-4 text-xs font-semibold text-white shadow-md shadow-sky-300/60 hover:from-sky-600 hover:to-sky-700 disabled:opacity-50"
        >
          {submitting ? (
            <Loader2 className="size-3.5 animate-spin" />
          ) : (
            <Link2 className="size-3.5" />
          )}
          Pair Device
        </motion.button>
      </div>
    </form>
  );
}
