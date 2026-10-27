import { createHash, timingSafeEqual } from "node:crypto";
import type { Request, Response } from "express";
import { db } from "../shared/firebase";

export type RobotDoc = Record<string, unknown> & {
  mode: "education" | "healthcare";
  knowledgeBaseId: string | null;
  assignedAdminId: string | null;
  config?: Record<string, unknown>;
};

// Robot-এর ID ও Key যাচাই করে। ব্যর্থ হলে নিজেই উত্তর পাঠিয়ে null ফেরত দেয়
export async function authenticateRobot(
  req: Request,
  res: Response,
  allowed: string[] = ["GET"],
  touch = true,
): Promise<{ robotId: string; robot: RobotDoc } | null> {
  if (!allowed.includes(req.method)) {
    res.status(405).json({ error: "method-not-allowed" });
    return null;
  }

  const robotId = String(req.header("x-robot-id") ?? "").trim();
  const key = String(req.header("x-robot-key") ?? "");

  if (!/^[A-Za-z0-9-]{3,32}$/.test(robotId) || key.length === 0) {
    res.status(401).json({ error: "unauthorized" });
    return null;
  }

  const credSnap = await db.collection("robotCredentials").doc(robotId).get();
  const cred = credSnap.data();
  if (!cred || cred.status !== "active") {
    res.status(401).json({ error: "unauthorized" });
    return null;
  }

  const given = createHash("sha256").update(key).digest();
  const stored = Buffer.from(String(cred.secretHash), "hex");
  if (given.length !== stored.length || !timingSafeEqual(given, stored)) {
    res.status(401).json({ error: "unauthorized" });
    return null;
  }

  const robotSnap = await db.collection("robots").doc(robotId).get();
  if (!robotSnap.exists) {
    res.status(401).json({ error: "unauthorized" });
    return null;
  }

  if (touch) {
    await robotSnap.ref.update({
      lastSeen: new Date().toISOString(),
      status: "online",
    });
  }

  return { robotId, robot: robotSnap.data() as RobotDoc };
}
