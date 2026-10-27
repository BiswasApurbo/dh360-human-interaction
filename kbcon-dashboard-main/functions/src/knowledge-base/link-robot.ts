import { onCall, HttpsError } from "firebase-functions/https";
import { FieldValue } from "firebase-admin/firestore";
import { z } from "zod";
import { db } from "../shared/firebase";
import { requireAdminOrSuper } from "../shared/guards";
import { requireKbAccess } from "./shared";

const schema = z.object({
  robotId: z.string().min(1),
  kbId: z.string().min(1).nullable(),
});

export const linkKnowledgeBase = onCall(
  { region: "asia-south1" },
  async (request) => {
    const { uid, role } = requireAdminOrSuper(request);

    const parsed = schema.safeParse(request.data);
    if (!parsed.success) {
      throw new HttpsError("invalid-argument", "ইনপুট ঠিক নেই");
    }
    const { robotId, kbId } = parsed.data;

    const robotRef = db.collection("robots").doc(robotId);
    const robotSnap = await robotRef.get();
    if (!robotSnap.exists) {
      throw new HttpsError("not-found", "Robot পাওয়া যায়নি");
    }
    const robot = robotSnap.data() ?? {};
    if (role !== "superAdmin" && robot.assignedAdminId !== uid) {
      throw new HttpsError("permission-denied", "অনুমতি নেই");
    }

    if (kbId) {
      const { data } = await requireKbAccess(kbId, uid, role);
      if (data.mode !== robot.mode) {
        throw new HttpsError(
          "failed-precondition",
          "Knowledge Base ও Robot-এর Mode মেলেনি",
        );
      }
    }

    // আগের Knowledge Base থেকে Robot সরানো
    const oldKb = robot.knowledgeBaseId as string | null;
    if (oldKb && oldKb !== kbId) {
      await db
        .collection("knowledgeBases")
        .doc(oldKb)
        .update({ robotIds: FieldValue.arrayRemove(robotId) });
    }
    if (kbId) {
      await db
        .collection("knowledgeBases")
        .doc(kbId)
        .update({ robotIds: FieldValue.arrayUnion(robotId) });
    }
    await robotRef.update({ knowledgeBaseId: kbId });

    return { robotId, kbId };
  },
);
