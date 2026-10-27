import { onCall, HttpsError } from "firebase-functions/https";
import { z } from "zod";
import { adminAuth, db } from "../shared/firebase";
import { requireSuperAdmin } from "../shared/guards";

const schema = z.object({
  uid: z.string().min(1),
  active: z.boolean(),
});

export const setAdminActive = onCall(
  { region: "asia-south1" },
  async (request) => {
    const actorId = requireSuperAdmin(request);

    const parsed = schema.safeParse(request.data);
    if (!parsed.success) {
      throw new HttpsError("invalid-argument", "ইনপুট ঠিক নেই");
    }
    const { uid, active } = parsed.data;

    const ref = db.collection("users").doc(uid);
    const snap = await ref.get();
    if (!snap.exists || snap.data()?.role !== "admin") {
      throw new HttpsError("not-found", "Admin পাওয়া যায়নি");
    }

    await adminAuth.updateUser(uid, { disabled: !active });
    if (!active) {
      await adminAuth.revokeRefreshTokens(uid);
    }
    await ref.update({ active });

    await db.collection("auditLogs").add({
      actorId,
      action: active ? "enableAdmin" : "disableAdmin",
      target: uid,
      at: new Date().toISOString(),
    });

    return { uid, active };
  },
);
