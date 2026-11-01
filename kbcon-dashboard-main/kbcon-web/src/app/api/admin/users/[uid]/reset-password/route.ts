import { NextResponse } from "next/server";
import { authenticateFirebaseToken } from "@/lib/dh360/auth";
import {
  forwardAdminRequestToBackend,
  getAdminAuth,
  hasServerAdminCredentials,
} from "@/lib/server-admin";

interface RouteParams {
  params: Promise<{ uid: string }> | { uid: string };
}

export async function POST(request: Request, context: RouteParams) {
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
      { error: "FORBIDDEN", message: "Only Super Admin can trigger password resets for admins." },
      { status: 403 }
    );
  }

  if (!hasServerAdminCredentials()) {
    try {
      const res = await forwardAdminRequestToBackend(`/admin/users/${uid}/reset-password`, "POST");
      const data = await res.json();
      return NextResponse.json(data, { status: res.status });
    } catch (err: unknown) {
      return NextResponse.json(
        { error: "RESET_FAILED", message: err instanceof Error ? err.message : "Failed to generate password reset." },
        { status: 500 }
      );
    }
  }

  try {
    const adminAuth = await getAdminAuth();
    const user = await adminAuth.getUser(uid);

    if (!user.email) {
      return NextResponse.json(
        { error: "NO_EMAIL", message: "Target user does not have an email address." },
        { status: 400 }
      );
    }

    // Generate reset link via Firebase Admin SDK
    const actionCodeSettings = {
      url: "https://kbcon-dashboard-gamma.vercel.app/login",
      handleCodeInApp: false,
    };

    const resetLink = await adminAuth.generatePasswordResetLink(user.email, actionCodeSettings);

    return NextResponse.json({
      success: true,
      message: `Password reset initiated for ${user.email}.`,
      hasResetLink: Boolean(resetLink),
    });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: "RESET_FAILED", message: err instanceof Error ? err.message : "Failed to generate password reset." },
      { status: 500 }
    );
  }
}
