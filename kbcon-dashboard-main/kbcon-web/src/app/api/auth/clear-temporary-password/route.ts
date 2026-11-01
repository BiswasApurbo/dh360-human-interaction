import { NextResponse } from "next/server";
import { authenticateFirebaseToken } from "@/lib/dh360/auth";
import { getAdminAuth, getAdminFirestore } from "@/lib/server-admin";

export async function POST(request: Request) {
  const authHeader = request.headers.get("authorization");
  const authResult = await authenticateFirebaseToken(authHeader);

  if (!authResult.success) {
    return NextResponse.json(
      { error: authResult.code, message: authResult.message },
      { status: authResult.status }
    );
  }

  const { uid } = authResult.principal;

  try {
    const adminAuth = getAdminAuth();
    const adminDb = getAdminFirestore();

    const user = await adminAuth.getUser(uid);
    const existingClaims = (user.customClaims as Record<string, unknown>) || {};
    
    // Clear mustChangePassword
    delete existingClaims.mustChangePassword;
    await adminAuth.setCustomUserClaims(uid, existingClaims);

    // Update Firestore record
    await adminDb.collection("users").doc(uid).set(
      {
        mustChangePassword: false,
        updatedAt: new Date().toISOString(),
      },
      { merge: true }
    );

    return NextResponse.json({
      success: true,
      message: "Temporary bootstrap status cleared successfully.",
    });
  } catch (err: unknown) {
    return NextResponse.json(
      {
        error: "INTERNAL_ERROR",
        message: err instanceof Error ? err.message : "Failed to clear bootstrap status.",
      },
      { status: 500 }
    );
  }
}
