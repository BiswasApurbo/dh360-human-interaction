import { onRequest } from "firebase-functions/https";
import { z } from "zod";
import { db } from "../shared/firebase";
import { authenticateRobot } from "./auth";

const HISTORY_EVERY_MS = 5 * 60 * 1000;
const DEFAULT_RETENTION_DAYS = 30;

const schema = z.object({
  battery: z.number().min(0).max(100),
  temperature: z.number().min(-20).max(120),
  activity: z.enum(["idle", "speaking", "listening", "charging"]),
  errorCode: z.string().trim().max(30).nullable().optional(),
  firmwareVersion: z.string().trim().max(30).optional(),
});

export const robotTelemetry = onRequest(
  { region: "asia-south1" },
  async (req, res) => {
    // lastSeen ও status নিচে একবারেই লেখা হবে, তাই এখানে আলাদা লেখা নেই
    const auth = await authenticateRobot(req, res, ["POST"], false);
    if (!auth) return;
    const { robotId, robot } = auth;

    const parsed = schema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "invalid-data" });
      return;
    }
    const { battery, temperature, activity } = parsed.data;
    const errorCode = parsed.data.errorCode ?? null;
    const firmwareVersion = parsed.data.firmwareVersion ?? null;

    const now = new Date();
    const nowIso = now.toISOString();

    const prev = (robot.telemetry ?? null) as {
      errorCode?: string | null;
      historyAt?: string;
    } | null;

    const lastHistory = prev?.historyAt ? Date.parse(prev.historyAt) : 0;
    const errorChanged = (prev?.errorCode ?? null) !== errorCode;
    const writeHistory =
      errorChanged || now.getTime() - lastHistory >= HISTORY_EVERY_MS;

    if (writeHistory) {
      // মালিক Admin-এর সংরক্ষণের দিন সংখ্যা
      let days = DEFAULT_RETENTION_DAYS;
      if (robot.assignedAdminId) {
        const admin = await db
          .collection("users")
          .doc(robot.assignedAdminId)
          .get();
        const d = Number(admin.data()?.telemetryRetentionDays);
        if (Number.isFinite(d) && d >= 1 && d <= 90) days = d;
      }
      const expireAt = new Date(now.getTime() + days * 86400000);

      await db.collection("robots").doc(robotId).collection("telemetry").add({
        battery,
        temperature,
        activity,
        errorCode,
        timestamp: nowIso,
        expireAt: expireAt.toISOString(),
      });
    }

    await db
      .collection("robots")
      .doc(robotId)
      .update({
        status: errorCode ? "error" : "online",
        lastSeen: nowIso,
        ...(firmwareVersion ? { firmwareVersion } : {}),
        telemetry: {
          battery,
          temperature,
          activity,
          errorCode,
          updatedAt: nowIso,
          historyAt: writeHistory ? nowIso : (prev?.historyAt ?? null),
        },
      });

    const config = (robot.config ?? {}) as Record<string, unknown>;
    const interval = Number(config.telemetryIntervalSec);
    res.json({
      ok: true,
      nextInSec: Number.isFinite(interval) && interval >= 10 ? interval : 30,
    });
  },
);
