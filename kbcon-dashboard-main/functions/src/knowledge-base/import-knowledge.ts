import { onCall, HttpsError } from "firebase-functions/https";
import { FieldValue } from "firebase-admin/firestore";
import { z } from "zod";
import { db } from "../shared/firebase";
import { requireAdminOrSuper } from "../shared/guards";
import { itemSchema, normalize, requireKbAccess } from "./shared";

const MAX_ROWS = 5000;

const schema = z.object({
  kbId: z.string().min(1),
  fileName: z.string().max(200),
  format: z.enum(["csv", "json"]),
  rows: z.array(z.unknown()).max(MAX_ROWS),
});

export const importKnowledge = onCall(
  { region: "asia-south1", timeoutSeconds: 120 },
  async (request) => {
    const { uid, role } = requireAdminOrSuper(request);

    const parsed = schema.safeParse(request.data);
    if (!parsed.success) {
      throw new HttpsError("invalid-argument", "ইনপুট ঠিক নেই বা সারি বেশি");
    }
    const { kbId, fileName, format, rows } = parsed.data;
    const { ref } = await requireKbAccess(kbId, uid, role);

    // আগের প্রশ্নগুলো, Duplicate ধরার জন্য
    const existing = await ref.collection("items").select("key").get();
    const seen = new Set<string>(
      existing.docs.map((d) => d.get("key") as string),
    );

    const valid: { key: string; item: z.infer<typeof itemSchema> }[] = [];
    const errors: { row: number; reason: string }[] = [];
    let skipped = 0;

    rows.forEach((raw, i) => {
      const r = itemSchema.safeParse(raw);
      if (!r.success) {
        errors.push({ row: i + 1, reason: "ডেটা ঠিক নেই" });
        return;
      }
      const key = normalize(r.data.question, r.data.language);
      if (seen.has(key)) {
        skipped++;
        errors.push({ row: i + 1, reason: "Duplicate প্রশ্ন" });
        return;
      }
      seen.add(key);
      valid.push({ key, item: r.data });
    });

    const now = new Date().toISOString();

    // Firestore batch-এ সর্বোচ্চ ৫০০ লেখা
    for (let i = 0; i < valid.length; i += 400) {
      const batch = db.batch();
      valid.slice(i, i + 400).forEach(({ key, item }) => {
        batch.set(ref.collection("items").doc(), {
          ...item,
          key,
          createdAt: now,
        });
      });
      await batch.commit();
    }

    if (valid.length > 0) {
      await ref.update({
        version: FieldValue.increment(1),
        itemCount: FieldValue.increment(valid.length),
        updatedAt: now,
      });
    }

    await db.collection("importHistory").add({
      kbId,
      fileName,
      format,
      total: rows.length,
      imported: valid.length,
      skipped,
      errors: errors.slice(0, 200),
      by: uid,
      at: now,
    });

    return {
      total: rows.length,
      imported: valid.length,
      skipped,
      errors: errors.slice(0, 200),
    };
  },
);
