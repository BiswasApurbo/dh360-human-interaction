import { onRequest } from "firebase-functions/https";
import { db } from "../shared/firebase";
import { authenticateRobot } from "./auth";

export const robotKnowledge = onRequest(
  { region: "asia-south1" },
  async (req, res) => {
    const auth = await authenticateRobot(req, res);
    if (!auth) return;
    const { robot } = auth;

    const kbId = robot.knowledgeBaseId;
    if (!kbId) {
      res.status(404).json({ error: "no-knowledge-base" });
      return;
    }

    const kbRef = db.collection("knowledgeBases").doc(kbId);
    const kbSnap = await kbRef.get();
    const kb = kbSnap.data();
    if (
      !kb ||
      !Array.isArray(kb.robotIds) ||
      !kb.robotIds.includes(auth.robotId)
    ) {
      res.status(404).json({ error: "no-knowledge-base" });
      return;
    }

    const version = Number(kb.version ?? 1);
    const known = Number(req.query.version);
    if (Number.isFinite(known) && known === version) {
      res.json({ changed: false, version });
      return;
    }

    const snap = await kbRef.collection("items").get();
    const items = snap.docs.map((d) => {
      const x = d.data();
      return {
        id: d.id,
        question: x.question,
        answer: x.answer,
        language: x.language,
        category: x.category,
      };
    });

    res.json({
      changed: true,
      version,
      mode: kb.mode,
      count: items.length,
      items,
    });
  },
);
