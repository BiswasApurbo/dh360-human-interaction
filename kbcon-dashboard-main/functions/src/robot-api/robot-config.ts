import { onRequest } from "firebase-functions/https";
import { authenticateRobot } from "./auth";

export const robotConfig = onRequest(
  { region: "asia-south1" },
  async (req, res) => {
    const auth = await authenticateRobot(req, res);
    if (!auth) return;
    const { robotId, robot } = auth;

    const config = (robot.config ?? {}) as Record<string, unknown>;
    const interval = Number(config.telemetryIntervalSec);

    res.json({
      robotId,
      name: robot.name ?? "",
      type: robot.type ?? "",
      mode: robot.mode,
      knowledgeBaseId: robot.knowledgeBaseId ?? null,
      telemetryIntervalSec:
        Number.isFinite(interval) && interval >= 10 ? interval : 30,
      config,
    });
  },
);
