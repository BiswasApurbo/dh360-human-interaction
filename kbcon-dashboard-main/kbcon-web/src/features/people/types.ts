export interface QAPair {
  question: string;
  answer: string;
}

export type PersonStatus = "active" | "suspended";
export type ConsentStatus = "granted" | "revoked" | "pending";

export interface PersonProfile {
  person_id: string;
  display_name: string;
  preferred_name: string | null;
  status: PersonStatus | string;
  consent_status: ConsentStatus | string;
  consent_timestamp: string | null;
  created_at: string;
  updated_at: string;
}

export interface PersonContext {
  person_id: string;
  display_name: string;
  preferred_name: string | null;
  role: string | null;
  department: string | null;
  office_room: string | null;
  working_hours: string | null;
  profile_description: string | null;
  qa_pairs: QAPair[];
}

export interface CreatePersonInput {
  display_name: string;
  preferred_name?: string | null;
  role?: string | null;
  department?: string | null;
  office_room?: string | null;
  working_hours?: string | null;
  profile_description?: string | null;
  qa_pairs?: QAPair[];
}

export interface UpdatePersonInput {
  display_name?: string;
  preferred_name?: string | null;
  status?: PersonStatus;
  consent_status?: ConsentStatus;
  role?: string | null;
  department?: string | null;
  office_room?: string | null;
  working_hours?: string | null;
  profile_description?: string | null;
  qa_pairs?: QAPair[];
}
