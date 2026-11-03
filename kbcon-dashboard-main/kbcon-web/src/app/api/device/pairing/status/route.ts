import { NextResponse } from "next/server";
import {
  getMemoryPairingSessions,
  type PairingSession,
} from "@/lib/device/provisioning";
import { getAdminFirestore, hasServerAdminCredentials } from "@/lib/server-admin";
import type { DocumentReference } from "firebase-admin/firestore";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const sessionId = searchParams.get("sessionId");
  const deviceId = searchParams.get("deviceId");

  if (!sessionId || !deviceId) {
    return NextResponse.json(
      { error: "INVALID_ARGUMENT", message: "sessionId and deviceId are required." },
      { status: 400 }
    );
  }

  let session: PairingSession | null = null;
  let docRef: DocumentReference | null = null;

  if (hasServerAdminCredentials()) {
    try {
      const db = await getAdminFirestore();
      docRef = db.collection("pairing_sessions").doc(sessionId);
      const snap = await docRef.get();
      if (snap.exists) {
        session = snap.data() as PairingSession;
      }
    } catch (err: unknown) {
      return NextResponse.json(
        {
          error: "INTERNAL_ERROR",
          message:
            err instanceof Error
              ? err.message
              : "Failed to retrieve pairing session.",
        },
        { status: 500 }
      );
    }
  } else {
    session = getMemoryPairingSessions().get(sessionId) || null;
  }

  if (!session || session.deviceId !== deviceId) {
    return NextResponse.json(
      { error: "NOT_FOUND", message: "Pairing session not found or invalid." },
      { status: 404 }
    );
  }

  // Check expiration (10-minute expiry window)
  if (new Date(session.expiresAt).getTime() < Date.now()) {
    if (session.status === "pending") {
      session.status = "expired";
      if (docRef) {
        await docRef.set({ status: "expired" }, { merge: true });
      }
    }
    return NextResponse.json({ status: "expired" });
  }

  if (session.status === "approved") {
    // Single-use: Mark as consumed once retrieved by device
    const approvedData = {
      status: "approved",
      robotId: session.robotId,
      tenantId: session.tenantId,
      deviceToken: session.deviceToken,
      robotConfig: session.robotConfig,
    };

    if (docRef) {
      await docRef.set({ status: "consumed", consumedAt: new Date().toISOString() }, { merge: true });
    } else {
      session.status = "consumed";
    }

    return NextResponse.json(approvedData);
  }

  return NextResponse.json({ status: session.status });
}
