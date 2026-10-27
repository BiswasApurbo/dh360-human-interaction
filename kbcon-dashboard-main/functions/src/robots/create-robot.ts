import { onCall, HttpsError } from "firebase-functions/https";
import { randomBytes, createHash } from "node:crypto";
import { z } from "zod";
import { db } from "../shared/firebase";
import { requireSuperAdmin } from "../shared/guards";

const schema = z.object({
  robotId: z
    .string()
    .trim()
    .regex(/^[A-Za-z0-9-]{3,32}$/),
  name: z.string().trim().min(2).max(80),
  type: z.string().trim().min(2).max(60),
  mode: z.enum(["education", "healthcare"]),
});

export const createRobot = onCall(
  { region: "asia-south1" },
  async (request) => {
    const actorId = requireSuperAdmin(request);

    const parsed = schema.safeParse(request.data);
    if (!parsed.success) {
      throw new HttpsError("invalid-argument", "ইনপুট ঠিক নেই");
    }
    const { robotId, name, type, mode } = parsed.data;

    const robotRef = db.collection("robots").doc(robotId);
    const credRef = db.collection("robotCredentials").doc(robotId);

    // গোপন Key: আসলটা একবারই ফেরত যাবে, ডেটাবেসে শুধু Hash
    const secret = randomBytes(24).toString("hex");
    const secretHash = createHash("sha256").update(secret).digest("hex");
    const now = new Date().toISOString();

    try {
      await db.runTransaction(async (tx) => {
        const existing = await tx.get(robotRef);
        if (existing.exists) {
          throw new HttpsError("already-exists", "এই Robot ID আগেই আছে");
        }
        tx.set(robotRef, {
          name,
          type,
          mode,
          assignedAdminId: null,
          assignedAdminName: null,
          status: "offline",
          lastSeen: null,
          firmwareVersion: null,
          config: {},
          knowledgeBaseId: null,
          createdAt: now,
          createdBy: actorId,
        });
        tx.set(credRef, {
          secretHash,
          status: "active",
          rotatedAt: now,
        });
      });
    } catch (err) {
      if (err instanceof HttpsError) throw err;
      throw new HttpsError("internal", "Robot তৈরি করা যায়নি");
    }

    await db.collection("auditLogs").add({
      actorId,
      action: "createRobot",
      target: robotId,
      at: now,
    });

    return { robotId, secret };
  },
);
