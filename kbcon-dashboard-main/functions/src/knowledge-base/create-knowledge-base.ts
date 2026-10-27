import { onCall, HttpsError } from "firebase-functions/https";
import { z } from "zod";
import { db } from "../shared/firebase";
import { requireAdminOrSuper } from "../shared/guards";

const schema = z.object({
  name: z.string().trim().min(2).max(80),
  mode: z.enum(["education", "healthcare"]),
});

export const createKnowledgeBase = onCall(
  { region: "asia-south1" },
  async (request) => {
    const { uid } = requireAdminOrSuper(request);

    const parsed = schema.safeParse(request.data);
    if (!parsed.success) {
      throw new HttpsError("invalid-argument", "ইনপুট ঠিক নেই");
    }
    const { name, mode } = parsed.data;
    const now = new Date().toISOString();

    const ref = await db.collection("knowledgeBases").add({
      name,
      mode,
      ownerAdminId: uid,
      robotIds: [],
      version: 1,
      itemCount: 0,
      createdAt: now,
      updatedAt: now,
    });

    await db.collection("auditLogs").add({
      actorId: uid,
      action: "createKnowledgeBase",
      target: ref.id,
      at: now,
    });

    return { kbId: ref.id };
  },
);
