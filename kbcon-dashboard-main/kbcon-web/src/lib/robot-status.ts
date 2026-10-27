export type RobotStatus = "online" | "offline" | "connecting" | "error";

// কমপক্ষে এত সময় চুপ থাকলে Offline
export const MIN_OFFLINE_MS = 2 * 60 * 1000;

// Offline ধরার সময় = Robot-এর ব্যবধানের ৩ গুণ (কমপক্ষে ২ মিনিট)
export function offlineAfterMs(config?: Record<string, unknown>): number {
  const sec = Number(config?.telemetryIntervalSec);
  const interval = Number.isFinite(sec) && sec >= 10 ? sec : 30;
  return Math.max(MIN_OFFLINE_MS, interval * 3 * 1000);
}

export function effectiveStatus(
  robot: {
    status: RobotStatus;
    lastSeen: string | null;
    config?: Record<string, unknown>;
  },
  now: number,
): RobotStatus {
  const seen = robot.lastSeen ? Date.parse(robot.lastSeen) : 0;
  if (!seen || now - seen > offlineAfterMs(robot.config)) return "offline";
  return robot.status === "offline" ? "online" : robot.status;
}
