import { onCall, HttpsError } from "firebase-functions/https";
import { FieldValue } from "firebase-admin/firestore";
import { z } from "zod";
import { requireAdminOrSuper } from "../shared/guards";
import { itemSchema, logChange, normalize, requireKbAccess } from "./shared";

const schema = z.object({
  kbId: z.string().min(1),
  itemId: z.string().min(1).nullable(),
  item: itemSchema,
});

export const saveKnowledgeItem = onCall(
  { region: "asia-south1" },
  async (request) => {
    const { uid, role } = requireAdminOrSuper(request);

    const parsed = schema.safeParse(request.data);
    if (!parsed.success) {
      throw new HttpsError("invalid-argument", "ইনপুট ঠিক নেই");
    }
    const { kbId, itemId, item } = parsed.data;
    const { ref } = await requireKbAccess(kbId, uid, role);

    const key = normalize(item.question, item.language);
    const dup = await ref.collection("items").where("key", "==", key).get();
    if (dup.docs.some((d) => d.id !== itemId)) {
      throw new HttpsError("already-exists", "এই প্রশ্ন আগেই আছে");
    }

    const now = new Date().toISOString();
    if (itemId) {
      await ref
        .collection("items")
        .doc(itemId)
        .update({
          ...item,
          key,
          updatedAt: now,
        });
      await ref.update({ version: FieldValue.increment(1), updatedAt: now });
      await logChange(ref, uid, "update", item.question);
      return { itemId };
    }

    const created = await ref.collection("items").add({
      ...item,
      key,
      createdAt: now,
    });
    await ref.update({
      version: FieldValue.increment(1),
      itemCount: FieldValue.increment(1),
      updatedAt: now,
    });
    await logChange(ref, uid, "create", item.question);
    return { itemId: created.id };
  },
);
