import { onCall, HttpsError } from "firebase-functions/https";
import { z } from "zod";
import { db } from "../shared/firebase";
import { requireSuperAdmin } from "../shared/guards";

const schema = z.object({
  robotId: z.string().min(1),
  adminId: z.string().min(1).nullable(),
});

export const assignRobot = onCall(
  { region: "asia-south1" },
  async (request) => {
    const actorId = requireSuperAdmin(request);

    const parsed = schema.safeParse(request.data);
    if (!parsed.success) {
      throw new HttpsError("invalid-argument", "ইনপুট ঠিক নেই");
    }
    const { robotId, adminId } = parsed.data;

    const robotRef = db.collection("robots").doc(robotId);
    const robotSnap = await robotRef.get();
    if (!robotSnap.exists) {
      throw new HttpsError("not-found", "Robot পাওয়া যায়নি");
    }

    let adminName: string | null = null;
    if (adminId !== null) {
      const adminSnap = await db.collection("users").doc(adminId).get();
      const data = adminSnap.data();
      if (!adminSnap.exists || data?.role !== "admin") {
        throw new HttpsError("not-found", "Admin পাওয়া যায়নি");
      }
      if (data.active === false) {
        throw new HttpsError("failed-precondition", "Admin বন্ধ আছে");
      }
      adminName = data.name ?? "";
    }

    await robotRef.update({
      assignedAdminId: adminId,
      assignedAdminName: adminName,
    });

    await db.collection("auditLogs").add({
      actorId,
      action: adminId ? "assignRobot" : "unassignRobot",
      target: robotId,
      at: new Date().toISOString(),
    });

    return { robotId, adminId };
  },
);
