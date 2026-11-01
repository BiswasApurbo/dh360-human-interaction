import { NextResponse, type NextRequest } from "next/server";

export function GET(request: NextRequest) {
  const search = request.nextUrl.search;
  return NextResponse.redirect(new URL(`/en/auth/action${search}`, request.url), 307);
}
