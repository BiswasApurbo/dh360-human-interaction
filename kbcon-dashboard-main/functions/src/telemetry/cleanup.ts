import { onSchedule } from "firebase-functions/scheduler";
import { db } from "../shared/firebase";

export const cleanupTelemetry = onSchedule(
  { schedule: "every 24 hours", region: "asia-south1" },
  async () => {
    const nowIso = new Date().toISOString();
    let deleted = 0;

    // কয়েক ধাপে মোছা, যাতে একসাথে বেশি না হয়
    for (let round = 0; round < 20; round++) {
      const snap = await db
        .collectionGroup("telemetry")
        .where("expireAt", "<", nowIso)
        .limit(400)
        .get();
      if (snap.empty) break;

      const batch = db.batch();
      snap.docs.forEach((d) => batch.delete(d.ref));
      await batch.commit();
      deleted += snap.size;
    }
    console.log(`cleanupTelemetry: ${deleted} records deleted`);
  },
);
