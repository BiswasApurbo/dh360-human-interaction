import { HttpsError } from "firebase-functions/https";
import { z } from "zod";
import { db } from "../shared/firebase";

export const itemSchema = z.object({
  question: z.string().trim().min(2).max(500),
  answer: z.string().trim().min(1).max(3000),
  language: z.enum(["bn", "en"]),
  category: z.string().trim().max(60).default("general"),
});

export type Item = z.infer<typeof itemSchema>;

// Duplicate শনাক্ত করতে প্রশ্ন সাধারণ রূপে আনা
export function normalize(q: string, lang: string): string {
  return lang + "|" + q.toLowerCase().replace(/\s+/g, " ").trim();
}

// Knowledge Base-এর মালিক বা Super Admin কিনা যাচাই
export async function requireKbAccess(
  kbId: string,
  uid: string,
  role: "admin" | "superAdmin",
) {
  const ref = db.collection("knowledgeBases").doc(kbId);
  const snap = await ref.get();
  if (!snap.exists) {
    throw new HttpsError("not-found", "Knowledge Base পাওয়া যায়নি");
  }
  if (role !== "superAdmin" && snap.data()?.ownerAdminId !== uid) {
    throw new HttpsError("permission-denied", "অনুমতি নেই");
  }
  return { ref, data: snap.data() ?? {} };
}
export async function logChange(
  ref: FirebaseFirestore.DocumentReference,
  uid: string,
  action: "create" | "update" | "delete",
  question: string,
) {
  await ref.collection("changes").add({
    by: uid,
    action,
    question: question.slice(0, 200),
    at: new Date().toISOString(),
  });
}
