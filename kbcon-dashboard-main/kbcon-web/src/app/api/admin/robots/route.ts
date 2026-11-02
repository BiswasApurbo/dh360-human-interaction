import { NextResponse } from "next/server";
import crypto from "node:crypto";
import { authenticateFirebaseToken } from "@/lib/dh360/auth";
import {
  forwardAdminRequestToBackend,
  getAdminFirestore,
  hasServerAdminCredentials,
} from "@/lib/server-admin";

export async function POST(request: Request) {
  const authHeader = request.headers.get("authorization");
  const authResult = await authenticateFirebaseToken(authHeader, { checkRevoked: true });

  if (!authResult.success) {
    return NextResponse.json(
      { error: authResult.code, message: authResult.message },
      { status: authResult.status }
    );
  }

  if (authResult.principal.mustChangePassword) {
    return NextResponse.json(
      {
        error: "PASSWORD_CHANGE_REQUIRED",
        message: "Mandatory password change required before accessing administrative endpoints.",
      },
      { status: 403 }
    );
  }

  if (authResult.principal.role !== "superAdmin") {
    return NextResponse.json(
      { error: "FORBIDDEN", message: "Only Super Admin can create robots." },
      { status: 403 }
    );
  }

  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "BAD_REQUEST", message: "Invalid JSON body." }, { status: 400 });
  }

  const robotId = typeof body.robotId === "string" ? body.robotId.trim() : "";
  const name = typeof body.name === "string" ? body.name.trim() : "";
  const language = body.language === "en" ? "en" : "bn";
  const mode = body.mode === "healthcare" ? "healthcare" : "education";

  if (!robotId || robotId.length < 3 || robotId.length > 32) {
    return NextResponse.json(
      { error: "INVALID_ARGUMENT", message: "Robot ID must be between 3 and 32 characters." },
      { status: 400 }
    );
  }

  if (!name || name.length < 2) {
    return NextResponse.json(
      { error: "INVALID_ARGUMENT", message: "Robot Name must be at least 2 characters." },
      { status: 400 }
    );
  }

  if (!hasServerAdminCredentials()) {
    try {
      const res = await forwardAdminRequestToBackend("/admin/robots", "POST", {
        robotId,
        name,
        language,
        mode,
        type: "Humanoid",
      });
      const data = await res.json();
      return NextResponse.json(data, { status: res.status });
    } catch (err: unknown) {
      return NextResponse.json(
        { error: "CREATION_FAILED", message: err instanceof Error ? err.message : "Failed to create robot." },
        { status: 500 }
      );
    }
  }

  try {
    const adminDb = await getAdminFirestore();
    const robotRef = adminDb.collection("robots").doc(robotId);
    const docSnap = await robotRef.get();

    if (docSnap.exists) {
      return NextResponse.json(
        { error: "ALREADY_EXISTS", message: `Robot ${robotId} already exists.` },
        { status: 409 }
      );
    }

    const now = new Date().toISOString();
    const secret = crypto.randomBytes(24).toString("hex");

    await robotRef.set({
      id: robotId,
      name,
      type: "Humanoid",
      language,
      mode,
      status: "online",
      assignedAdminId: null,
      assignedAdminName: null,
      createdAt: now,
      updatedAt: now,
    });

    return NextResponse.json({ success: true, robotId, secret });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: "INTERNAL_ERROR", message: err instanceof Error ? err.message : "Failed to create robot." },
      { status: 500 }
    );
  }
}
