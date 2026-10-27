import { onCall, HttpsError } from "firebase-functions/https";
import { z } from "zod";
import { db } from "../shared/firebase";
import { requireAdminOrSuper } from "../shared/guards";
import { requireKbAccess } from "./shared";

const schema = z.object({
  kbId: z.string().min(1),
  name: z.string().trim().min(2).max(80),
});

export const renameKnowledgeBase = onCall(
  { region: "asia-south1" },
  async (request) => {
    const { uid, role } = requireAdminOrSuper(request);

    const parsed = schema.safeParse(request.data);
    if (!parsed.success) {
      throw new HttpsError("invalid-argument", "ইনপুট ঠিক নেই");
    }
    const { kbId, name } = parsed.data;
    const { ref } = await requireKbAccess(kbId, uid, role);

    const now = new Date().toISOString();
    await ref.update({ name, updatedAt: now });

    await db.collection("auditLogs").add({
      actorId: uid,
      action: "renameKnowledgeBase",
      target: kbId,
      at: now,
    });

    return { kbId, name };
  },
);
