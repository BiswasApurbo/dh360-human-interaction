import { onCall, HttpsError } from "firebase-functions/https";
import { z } from "zod";
import { adminAuth, db } from "../shared/firebase";
import { requireSuperAdmin } from "../shared/guards";

const schema = z.object({
  name: z.string().trim().min(2).max(80),
  email: z.string().trim().email(),
  password: z.string().min(6).max(100),
});

export const createAdmin = onCall(
  { region: "asia-south1" },
  async (request) => {
    const actorId = requireSuperAdmin(request);

    const parsed = schema.safeParse(request.data);
    if (!parsed.success) {
      throw new HttpsError("invalid-argument", "ইনপুট ঠিক নেই");
    }
    const { name, email, password } = parsed.data;

    let uid: string;
    try {
      const user = await adminAuth.createUser({
        email,
        password,
        displayName: name,
      });
      uid = user.uid;
    } catch (err) {
      const code = (err as { code?: string }).code;
      if (code === "auth/email-already-exists") {
        throw new HttpsError("already-exists", "এই ইমেইলে অ্যাকাউন্ট আছে");
      }
      throw new HttpsError("internal", "Admin তৈরি করা যায়নি");
    }

    await adminAuth.setCustomUserClaims(uid, { role: "admin" });

    await db.collection("users").doc(uid).set({
      name,
      email,
      role: "admin",
      active: true,
      createdAt: new Date().toISOString(),
      createdBy: actorId,
    });

    await db.collection("auditLogs").add({
      actorId,
      action: "createAdmin",
      target: uid,
      at: new Date().toISOString(),
    });

    return { uid };
  },
);
