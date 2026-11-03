import { NextResponse } from "next/server";
import { getAdminFirestore, hasServerAdminCredentials } from "@/lib/server-admin";

export async function POST(request: Request) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "BAD_REQUEST", message: "Invalid JSON body." }, { status: 400 });
  }

  const robotId = typeof body.robotId === "string" ? body.robotId.trim() : "";
  if (!robotId) {
    return NextResponse.json({ error: "INVALID_ARGUMENT", message: "robotId is required." }, { status: 400 });
  }

  if (!hasServerAdminCredentials()) {
    return NextResponse.json({ success: true, timestamp: new Date().toISOString() });
  }

  try {
    const adminDb = await getAdminFirestore();
    let robotRef = adminDb.collection("robots").doc(robotId);
    let docSnap = await robotRef.get();

    // Canonical resolution: match across case and hyphenation (KB001 <-> Kb-001 <-> KB-001)
    if (!docSnap.exists) {
      const cleanTarget = robotId.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
      const allRobots = await adminDb.collection("robots").get();
      for (const d of allRobots.docs) {
        if (d.id.replace(/[^a-zA-Z0-9]/g, "").toLowerCase() === cleanTarget) {
          robotRef = d.ref;
          docSnap = d;
          break;
        }
      }
    }

    if (!docSnap.exists) {
      return NextResponse.json({ error: "NOT_FOUND", message: `Robot ${robotId} not found.` }, { status: 404 });
    }

    const currentData = docSnap.data() || {};
    const now = new Date().toISOString();

    const updates: Record<string, unknown> = {
      status: "online",
      lastSeen: now,
      updatedAt: now,
    };

    if (body.telemetry && typeof body.telemetry === "object") {
      updates.telemetry = body.telemetry;
    }
    if (typeof body.firmwareVersion === "string") {
      updates.firmwareVersion = body.firmwareVersion;
    }

    await robotRef.set(updates, { merge: true });

    // Return the latest robot configuration for edge state synchronization
    return NextResponse.json({
      success: true,
      timestamp: now,
      config: currentData.config || {},
      mode: currentData.mode || "education",
      language: currentData.language || "en",
      knowledgeBaseId: currentData.knowledgeBaseId || null,
    });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: "INTERNAL_ERROR", message: err instanceof Error ? err.message : "Failed to process heartbeat." },
      { status: 500 }
    );
  }
}
