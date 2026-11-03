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
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const message =
          typeof data.detail === "string"
            ? data.detail
            : Array.isArray(data.detail)
              ? data.detail.map((d: { msg?: string }) => d.msg || "Validation error").join("; ")
              : (data.message || "Failed to create robot.");
        return NextResponse.json(
          { error: data.error || (res.status === 409 ? "ALREADY_EXISTS" : "CREATION_FAILED"), message },
          { status: res.status }
        );
      }
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

export async function PATCH(request: Request) {
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
    try {
      const res = await forwardAdminRequestToBackend("/admin/robots", "PATCH", body);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const message =
          typeof data.detail === "string"
            ? data.detail
            : Array.isArray(data.detail)
              ? data.detail.map((d: { msg?: string }) => d.msg || "Validation error").join("; ")
              : (data.message || "Failed to update robot.");
        return NextResponse.json({ error: data.error || "UPDATE_FAILED", message }, { status: res.status });
      }
      return NextResponse.json(data, { status: res.status });
    } catch (err: unknown) {
      return NextResponse.json(
        { error: "UPDATE_FAILED", message: err instanceof Error ? err.message : "Failed to update robot." },
        { status: 500 }
      );
    }
  }

  try {
    const adminDb = await getAdminFirestore();
    const robotRef = adminDb.collection("robots").doc(robotId);
    const docSnap = await robotRef.get();
    if (!docSnap.exists) {
      return NextResponse.json({ error: "NOT_FOUND", message: `Robot ${robotId} not found.` }, { status: 404 });
    }

    const currentData = docSnap.data() || {};
    const isSuper = authResult.principal.role === "superAdmin";
    const isAdminOwner =
      authResult.principal.role === "admin" &&
      currentData.assignedAdminId === authResult.principal.uid;

    if (!isSuper && !isAdminOwner) {
      return NextResponse.json({ error: "FORBIDDEN", message: "Not authorized to update this robot." }, { status: 403 });
    }

    const now = new Date().toISOString();
    const updates: Record<string, unknown> = { updatedAt: now };

    if ("adminId" in body) {
      if (!isSuper) {
        return NextResponse.json({ error: "FORBIDDEN", message: "Only Super Admin can assign robots." }, { status: 403 });
      }
      const adminId = typeof body.adminId === "string" && body.adminId.trim() ? body.adminId.trim() : null;
      let adminName: string | null = null;
      if (adminId) {
        const uDoc = await adminDb.collection("users").doc(adminId).get();
        if (uDoc.exists) {
          const uData = uDoc.data() || {};
          adminName = (uData.name as string) || (uData.email as string) || null;
        }
      }
      updates.assignedAdminId = adminId;
      updates.assignedAdminName = adminName;
    }

    if (body.config && typeof body.config === "object") {
      updates.config = body.config;
    }

    if (body.mode === "education" || body.mode === "healthcare") {
      updates.mode = body.mode;
    }

    await robotRef.set(updates, { merge: true });
    return NextResponse.json({ success: true, robotId, ...updates });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: "INTERNAL_ERROR", message: err instanceof Error ? err.message : "Failed to update robot." },
      { status: 500 }
    );
  }
}
