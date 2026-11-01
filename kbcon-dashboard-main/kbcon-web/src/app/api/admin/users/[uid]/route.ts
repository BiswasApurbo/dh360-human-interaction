import { NextResponse } from "next/server";
import { authenticateFirebaseToken } from "@/lib/dh360/auth";
import {
  forwardAdminRequestToBackend,
  getAdminAuth,
  getAdminFirestore,
  hasServerAdminCredentials,
} from "@/lib/server-admin";

interface RouteParams {
  params: Promise<{ uid: string }> | { uid: string };
}

export async function PATCH(request: Request, context: RouteParams) {
  const { uid } = await Promise.resolve(context.params);
  const authHeader = request.headers.get("authorization");
  const authResult = await authenticateFirebaseToken(authHeader);

  if (!authResult.success) {
    return NextResponse.json(
      { error: authResult.code, message: authResult.message },
      { status: authResult.status }
    );
  }

  if (authResult.principal.role !== "superAdmin") {
    return NextResponse.json(
      { error: "FORBIDDEN", message: "Only Super Admin can update administrator profiles." },
      { status: 403 }
    );
  }

  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "BAD_REQUEST", message: "Invalid JSON body." }, { status: 400 });
  }

  if (!hasServerAdminCredentials()) {
    try {
      const res = await forwardAdminRequestToBackend(`/admin/users/${uid}`, "PATCH", body);
      const data = await res.json();
      return NextResponse.json(data, { status: res.status });
    } catch (err: unknown) {
      return NextResponse.json(
        { error: "UPDATE_FAILED", message: err instanceof Error ? err.message : "Failed to update user." },
        { status: 500 }
      );
    }
  }

  try {
    const adminAuth = getAdminAuth();
    const adminDb = getAdminFirestore();

    const targetUser = await adminAuth.getUser(uid);
    const targetClaims = (targetUser.customClaims as Record<string, unknown>) || {};
    const isTargetSuper = targetClaims.role === "superAdmin" || targetUser.email === "superadmin@gmail.com";

    // Protect Super Admin from demotion or deactivation
    if (isTargetSuper && (body.active === false || body.role === "admin")) {
      return NextResponse.json(
        { error: "FORBIDDEN", message: "Cannot demote or deactivate the Super Admin account." },
        { status: 403 }
      );
    }

    const updates: Record<string, unknown> = {};

    // 1. Handle active / disabled status
    if (typeof body.active === "boolean") {
      await adminAuth.updateUser(uid, { disabled: !body.active });
      if (!body.active) {
        // Immediately invalidate existing refresh tokens and active sessions
        await adminAuth.revokeRefreshTokens(uid);
      }
      updates.active = body.active;
    }

    // 2. Handle permissions update
    if (body.permissions && typeof body.permissions === "object") {
      const newPermissions = body.permissions as Record<string, boolean>;
      await adminAuth.setCustomUserClaims(uid, {
        ...targetClaims,
        role: targetClaims.role || "admin",
        permissions: newPermissions,
      });
      updates.permissions = newPermissions;
    }

    // 3. Handle name update
    if (typeof body.name === "string" && body.name.trim()) {
      await adminAuth.updateUser(uid, { displayName: body.name.trim() });
      updates.name = body.name.trim();
    }

    updates.updatedAt = new Date().toISOString();

    // 4. Update Firestore doc
    await adminDb.collection("users").doc(uid).set(updates, { merge: true });

    return NextResponse.json({ success: true, uid, updates });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: "UPDATE_FAILED", message: err instanceof Error ? err.message : "Failed to update user." },
      { status: 500 }
    );
  }
}

export async function DELETE(request: Request, context: RouteParams) {
  const { uid } = await Promise.resolve(context.params);
  const authHeader = request.headers.get("authorization");
  const authResult = await authenticateFirebaseToken(authHeader);

  if (!authResult.success) {
    return NextResponse.json(
      { error: authResult.code, message: authResult.message },
      { status: authResult.status }
    );
  }

  if (authResult.principal.role !== "superAdmin") {
    return NextResponse.json(
      { error: "FORBIDDEN", message: "Only Super Admin can delete administrator accounts." },
      { status: 403 }
    );
  }

  // Prevent Super Admin from deleting their own active account
  if (uid === authResult.principal.uid) {
    return NextResponse.json(
      { error: "FORBIDDEN", message: "Cannot delete your own active Super Admin account." },
      { status: 403 }
    );
  }

  if (!hasServerAdminCredentials()) {
    try {
      const res = await forwardAdminRequestToBackend(`/admin/users/${uid}`, "DELETE");
      const data = await res.json();
      return NextResponse.json(data, { status: res.status });
    } catch (err: unknown) {
      return NextResponse.json(
        { error: "DELETE_FAILED", message: err instanceof Error ? err.message : "Failed to delete user." },
        { status: 500 }
      );
    }
  }

  try {
    const adminAuth = getAdminAuth();
    const adminDb = getAdminFirestore();

    const targetUser = await adminAuth.getUser(uid);
    const targetClaims = (targetUser.customClaims as Record<string, unknown>) || {};
    const isTargetSuper = targetClaims.role === "superAdmin" || targetUser.email === "superadmin@gmail.com";

    if (isTargetSuper) {
      return NextResponse.json(
        { error: "FORBIDDEN", message: "Super Admin account cannot be deleted." },
        { status: 403 }
      );
    }

    // Revoke tokens and delete user from Firebase Auth
    await adminAuth.revokeRefreshTokens(uid).catch(() => {});
    await adminAuth.deleteUser(uid);

    // Delete user doc from Firestore
    await adminDb.collection("users").doc(uid).delete().catch(() => {});

    return NextResponse.json({ success: true, deletedUid: uid });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: "DELETE_FAILED", message: err instanceof Error ? err.message : "Failed to delete user." },
      { status: 500 }
    );
  }
}
