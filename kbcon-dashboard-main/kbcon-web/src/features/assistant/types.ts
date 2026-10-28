/**
 * Data contracts and UI types for the DH360 AI Assistant interface.
 */

export type AssistantLanguage = "bn" | "en";

export type AuthorizationStatus =
  | "active"
  | "authorized"
  | "unknown"
  | "uncertain"
  | "no_face"
  | "lost"
  | "timeout"
  | "cancelled";

export interface AuthorizationSession {
  authorization_id: string;
  status: AuthorizationStatus;
  person_id?: string | null;
  display_name?: string | null;
  created_at_iso?: string;
  updated_at_iso?: string;
  ended_at_iso?: string | null;
}

export type AssistantMode = "general" | "authorized";

export interface ChatMessage {
  id: string;
  role: "user" | "model" | "system";
  text: string;
  timestamp: number;
  language: AssistantLanguage;
  isPersonal?: boolean;
  isVoice?: boolean;
  error?: string;
}

export interface DialogueTurnResponse {
  role: "user" | "model" | "system";
  text: string;
  timestamp: number;
  language: string;
  metadata?: {
    user_text?: string;
    is_voice?: boolean;
    mock?: boolean;
    error?: string;
    [key: string]: unknown;
  };
}

export interface DialogueVoiceTurnResponse {
  user_text: string;
  role: "user" | "model" | "system";
  text: string;
  timestamp: number;
  language: string;
  metadata?: {
    is_voice?: boolean;
    mock?: boolean;
    error?: string;
    [key: string]: unknown;
  };
}
