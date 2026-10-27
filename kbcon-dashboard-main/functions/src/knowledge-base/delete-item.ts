import { onCall, HttpsError } from "firebase-functions/https";
import { FieldValue } from "firebase-admin/firestore";
import { z } from "zod";
import { requireAdminOrSuper } from "../shared/guards";
import { logChange, requireKbAccess } from "./shared";

const schema = z.object({
  kbId: z.string().min(1),
  itemId: z.string().min(1),
});

export const deleteKnowledgeItem = onCall(
  { region: "asia-south1" },
  async (request) => {
    const { uid, role } = requireAdminOrSuper(request);

    const parsed = schema.safeParse(request.data);
    if (!parsed.success) {
      throw new HttpsError("invalid-argument", "ইনপুট ঠিক নেই");
    }
    const { kbId, itemId } = parsed.data;
    const { ref } = await requireKbAccess(kbId, uid, role);

    const itemRef = ref.collection("items").doc(itemId);
    const snap = await itemRef.get();
    if (!snap.exists) {
      throw new HttpsError("not-found", "Q&A পাওয়া যায়নি");
    }
    const question = String(snap.data()?.question ?? "");
    await itemRef.delete();
    await logChange(ref, uid, "delete", question);
    await ref.update({
      version: FieldValue.increment(1),
      itemCount: FieldValue.increment(-1),
      updatedAt: new Date().toISOString(),
    });
    return { itemId };
  },
);
