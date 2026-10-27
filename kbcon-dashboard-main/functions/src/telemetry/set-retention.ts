import { onCall, HttpsError } from "firebase-functions/https";
import { z } from "zod";
import { db } from "../shared/firebase";
import { requireAdminOrSuper } from "../shared/guards";

const schema = z.object({
  days: z.number().int().min(1).max(90),
});

export const setTelemetryRetention = onCall(
  { region: "asia-south1" },
  async (request) => {
    const { uid } = requireAdminOrSuper(request);

    const parsed = schema.safeParse(request.data);
    if (!parsed.success) {
      throw new HttpsError("invalid-argument", "দিন ১ থেকে ৯০ হতে হবে");
    }

    await db
      .collection("users")
      .doc(uid)
      .set({ telemetryRetentionDays: parsed.data.days }, { merge: true });

    return { days: parsed.data.days };
  },
);
