import { onCall, HttpsError } from "firebase-functions/https";
import { z } from "zod";
import { db } from "../shared/firebase";
import { requireAdminOrSuper } from "../shared/guards";

const schema = z.object({
  robotId: z.string().min(1),
  config: z.object({
    defaultLanguage: z.enum(["bn", "en"]),
    telemetryIntervalSec: z.number().int().min(10).max(300),
    fallbackBn: z.string().trim().min(1).max(300),
    fallbackEn: z.string().trim().min(1).max(300),
  }),
});

export const updateRobotConfig = onCall(
  { region: "asia-south1" },
  async (request) => {
    const { uid, role } = requireAdminOrSuper(request);

    const parsed = schema.safeParse(request.data);
    if (!parsed.success) {
      throw new HttpsError("invalid-argument", "ইনপুট ঠিক নেই");
    }
    const { robotId, config } = parsed.data;

    const ref = db.collection("robots").doc(robotId);
    const snap = await ref.get();
    if (!snap.exists) {
      throw new HttpsError("not-found", "Robot পাওয়া যায়নি");
    }
    if (role !== "superAdmin" && snap.data()?.assignedAdminId !== uid) {
      throw new HttpsError("permission-denied", "অনুমতি নেই");
    }

    await ref.update({ config });

    await db.collection("auditLogs").add({
      actorId: uid,
      action: "updateRobotConfig",
      target: robotId,
      at: new Date().toISOString(),
    });

    return { robotId };
  },
);
