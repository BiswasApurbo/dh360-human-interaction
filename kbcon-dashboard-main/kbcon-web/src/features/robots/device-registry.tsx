"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Cpu,
  AlertCircle,
  Loader2,
  RefreshCw,
  Link2,
  Unlink2,
  X,
  Server,
  Radio,
} from "lucide-react";
import { auth } from "@/lib/firebase";
import type { RobotRow } from "./use-robots";

export interface DeviceRegistryItem {
  deviceId: string;
  publicKey: string;
  fingerprint: string;
  status: string;
  bindingStatus: "bound" | "unbound";
  robotId: string | null;
  robotName: string | null;
  tenantId: string | null;
  tenantName: string | null;
  language?: string;
  mode?: string;
  deviceInfo: Record<string, unknown>;
  lastHeartbeat: string | null;
  isOnline: boolean;
  registeredAt?: string | null;
  pairedAt?: string | null;
}

export function DeviceRegistry({
  robots,
  preselectedRobotId,
  onClose,
  onDone,
}: {
  robots: RobotRow[];
  preselectedRobotId?: string | null;
  onClose: () => void;
  onDone: (msg: string) => void;
}) {
  const [devices, setDevices] = useState<DeviceRegistryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [bindTargetRobot, setBindTargetRobot] = useState<Record<string, string>>({});

  const fetchDevices = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const token = await auth.currentUser?.getIdToken(true);
      if (!token) {
        setError("Authentication session expired.");
        return;
      }
      const res = await fetch("/api/admin/devices", {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.message || `Failed to fetch devices (HTTP ${res.status})`);
      }
      const data = await res.json();
      setDevices(data.devices || []);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Error loading device registry.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchDevices();
  }, [fetchDevices]);

  async function handleBind(deviceId: string) {
    const robotId = bindTargetRobot[deviceId] || preselectedRobotId || robots[0]?.id;
    if (!robotId) {
      setError("Please select a target robot to bind.");
      return;
    }

    setActionLoading(deviceId);
    setError(null);
    try {
      const token = await auth.currentUser?.getIdToken(true);
      const res = await fetch("/api/admin/devices/bind", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ deviceId, robotId }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.success) {
        onDone(data.message || `Device ${deviceId} bound to ${robotId}!`);
        await fetchDevices();
      } else {
        setError(data.message || data.error || "Failed to bind device.");
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Bind request failed.");
    } finally {
      setActionLoading(null);
    }
  }

  async function handleUnbind(deviceId: string) {
    if (!confirm(`Are you sure you want to unbind device ${deviceId}?`)) return;

    setActionLoading(deviceId);
    setError(null);
    try {
      const token = await auth.currentUser?.getIdToken(true);
      const res = await fetch("/api/admin/devices/unbind", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ deviceId }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.success) {
        onDone(data.message || `Device ${deviceId} unbound.`);
        await fetchDevices();
      } else {
        setError(data.message || data.error || "Failed to unbind device.");
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unbind request failed.");
    } finally {
      setActionLoading(null);
    }
  }

  return (
    <div className="rounded-3xl border border-sky-200 bg-white p-6 shadow-xl space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-sky-100 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-2xl bg-sky-500/10 text-sky-600 ring-1 ring-sky-500/20">
            <Cpu className="size-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-sky-950">Super Admin Device Registry</h2>
            <p className="text-xs text-sky-700/80">
              Manage permanent hardware identities, cryptographic bindings, and tenant assignments.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => void fetchDevices()}
            disabled={loading}
            className="flex items-center gap-1.5 rounded-xl border border-sky-200 bg-white px-3 py-1.5 text-xs font-semibold text-sky-800 hover:bg-sky-50"
          >
            <RefreshCw className={`size-3.5 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl p-1.5 text-sky-600 hover:bg-sky-100"
            aria-label="Close"
          >
            <X className="size-5" />
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-start gap-2.5 rounded-2xl bg-red-50 p-3.5 text-sm text-red-700 ring-1 ring-red-200">
          <AlertCircle className="size-4 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {/* Device List */}
      {loading ? (
        <div className="flex items-center justify-center py-12 text-sm text-sky-600 gap-2">
          <Loader2 className="size-5 animate-spin" />
          Loading registered devices...
        </div>
      ) : devices.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-sky-200 bg-sky-50/50 p-8 text-center">
          <Server className="mx-auto size-10 text-sky-400 mb-2" />
          <h3 className="text-sm font-bold text-sky-900">No physical devices registered yet</h3>
          <p className="text-xs text-sky-700 mt-1 max-w-md mx-auto">
            When a Raspberry Pi 5 runs KBCON software, it registers its permanent cryptographic
            identity automatically.
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-2xl border border-sky-100 bg-white">
          <table className="w-full text-left text-sm">
            <thead className="bg-sky-50/70 text-xs font-bold text-sky-900 border-b border-sky-100">
              <tr>
                <th className="py-3 px-4">Device Identity</th>
                <th className="py-3 px-4">Ed25519 Fingerprint</th>
                <th className="py-3 px-4">Status & Connectivity</th>
                <th className="py-3 px-4">Bound Robot & Tenant</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-sky-100">
              {devices.map((dev) => {
                const isBound = dev.bindingStatus === "bound" || Boolean(dev.robotId);
                const selectedRobot = bindTargetRobot[dev.deviceId] || dev.robotId || robots[0]?.id || "";

                return (
                  <tr key={dev.deviceId} className="hover:bg-sky-50/40">
                    <td className="py-3 px-4">
                      <div className="font-semibold text-sky-950 font-mono text-xs">
                        {dev.deviceId}
                      </div>
                      <div className="text-[11px] text-slate-500">
                        {String(dev.deviceInfo?.hostname || dev.deviceInfo?.os || "Linux Pi 5")}
                      </div>
                    </td>
                    <td className="py-3 px-4 font-mono text-xs text-slate-600">
                      {dev.fingerprint || "—"}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-1.5">
                        <span
                          className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold ${
                            dev.isOnline
                              ? "bg-emerald-100 text-emerald-800"
                              : "bg-slate-100 text-slate-600"
                          }`}
                        >
                          <Radio className="size-2.5" />
                          {dev.isOnline ? "Online" : "Offline"}
                        </span>
                        <span className="inline-flex rounded-full bg-sky-100 px-2 py-0.5 text-[11px] font-medium text-sky-800">
                          {isBound ? "Bound" : "Unbound"}
                        </span>
                      </div>
                      {dev.lastHeartbeat && (
                        <div className="text-[10px] text-slate-400 mt-0.5">
                          Seen {new Date(dev.lastHeartbeat).toLocaleTimeString()}
                        </div>
                      )}
                    </td>
                    <td className="py-3 px-4">
                      {isBound ? (
                        <div>
                          <div className="font-semibold text-sky-900">
                            {dev.robotName || dev.robotId}
                            <span className="ml-1 text-xs text-slate-500 font-mono">
                              ({dev.robotId})
                            </span>
                          </div>
                          <div className="text-xs text-slate-600">
                            Tenant: <span className="font-semibold text-sky-800">{dev.tenantName || "Salute"}</span>
                          </div>
                        </div>
                      ) : (
                        <span className="text-xs text-amber-700 italic">
                          Not assigned to any robot
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-right">
                      {isBound ? (
                        <button
                          type="button"
                          onClick={() => void handleUnbind(dev.deviceId)}
                          disabled={actionLoading === dev.deviceId}
                          className="inline-flex items-center gap-1.5 rounded-xl border border-rose-200 bg-rose-50 px-3 py-1.5 text-xs font-semibold text-rose-700 hover:bg-rose-100 disabled:opacity-50"
                        >
                          {actionLoading === dev.deviceId ? (
                            <Loader2 className="size-3.5 animate-spin" />
                          ) : (
                            <Unlink2 className="size-3.5" />
                          )}
                          Unbind
                        </button>
                      ) : (
                        <div className="flex items-center justify-end gap-2">
                          <select
                            value={selectedRobot}
                            onChange={(e) =>
                              setBindTargetRobot((prev) => ({
                                ...prev,
                                [dev.deviceId]: e.target.value,
                              }))
                            }
                            className="rounded-xl border border-sky-300 bg-white px-2.5 py-1 text-xs text-sky-900 focus:outline-none focus:ring-1 focus:ring-sky-500"
                          >
                            {robots.map((r) => (
                              <option key={r.id} value={r.id}>
                                {r.name} ({r.id})
                              </option>
                            ))}
                          </select>
                          <button
                            type="button"
                            onClick={() => void handleBind(dev.deviceId)}
                            disabled={actionLoading === dev.deviceId}
                            className="inline-flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm hover:from-sky-600 hover:to-sky-700 disabled:opacity-50"
                          >
                            {actionLoading === dev.deviceId ? (
                              <Loader2 className="size-3.5 animate-spin" />
                            ) : (
                              <Link2 className="size-3.5" />
                            )}
                            Bind Device
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
