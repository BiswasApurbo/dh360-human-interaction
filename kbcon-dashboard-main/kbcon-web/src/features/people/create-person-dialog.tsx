"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { Plus, Trash2, UserPlus, Loader2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import type { CreatePersonInput, QAPair } from "./types";

interface CreatePersonDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit: (data: CreatePersonInput) => Promise<unknown>;
}

export function CreatePersonDialog({
  open,
  onOpenChange,
  onSubmit,
}: CreatePersonDialogProps) {
  const t = useTranslations("People");

  const [displayName, setDisplayName] = useState("");
  const [preferredName, setPreferredName] = useState("");
  const [role, setRole] = useState("");
  const [department, setDepartment] = useState("");
  const [officeRoom, setOfficeRoom] = useState("");
  const [workingHours, setWorkingHours] = useState("");
  const [profileDescription, setProfileDescription] = useState("");
  const [qaPairs, setQaPairs] = useState<QAPair[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const resetForm = () => {
    setDisplayName("");
    setPreferredName("");
    setRole("");
    setDepartment("");
    setOfficeRoom("");
    setWorkingHours("");
    setProfileDescription("");
    setQaPairs([]);
    setError(null);
  };

  const handleAddQa = () => {
    setQaPairs((prev) => [...prev, { question: "", answer: "" }]);
  };

  const handleUpdateQa = (
    index: number,
    field: "question" | "answer",
    val: string
  ) => {
    setQaPairs((prev) =>
      prev.map((item, i) => (i === index ? { ...item, [field]: val } : item))
    );
  };

  const handleRemoveQa = (index: number) => {
    setQaPairs((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!displayName.trim()) {
      setError("Display name is required.");
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const validQa = qaPairs.filter(
        (q) => q.question.trim() && q.answer.trim()
      );
      await onSubmit({
        display_name: displayName.trim(),
        preferred_name: preferredName.trim() || undefined,
        role: role.trim() || undefined,
        department: department.trim() || undefined,
        office_room: officeRoom.trim() || undefined,
        working_hours: workingHours.trim() || undefined,
        profile_description: profileDescription.trim() || undefined,
        qa_pairs: validQa.length > 0 ? validQa : undefined,
      });
      resetForm();
      onOpenChange(false);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to register person");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(val) => {
        if (!val) resetForm();
        onOpenChange(val);
      }}
    >
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <UserPlus className="size-5 text-sky-600" />
            {t("registerPerson")}
          </DialogTitle>
          <DialogDescription>{t("subtitle")}</DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4 py-2">
          {error && (
            <div className="rounded-md bg-red-50 p-3 text-sm text-red-600 border border-red-200">
              {error}
            </div>
          )}

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="displayName">{t("name")} *</Label>
              <Input
                id="displayName"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="Dr. Alice Smith"
                required
              />
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="preferredName">{t("preferredName")}</Label>
              <Input
                id="preferredName"
                value={preferredName}
                onChange={(e) => setPreferredName(e.target.value)}
                placeholder="Alice"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="role">{t("role")}</Label>
              <Input
                id="role"
                value={role}
                onChange={(e) => setRole(e.target.value)}
                placeholder="Cardiologist"
              />
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="department">{t("department")}</Label>
              <Input
                id="department"
                value={department}
                onChange={(e) => setDepartment(e.target.value)}
                placeholder="Cardiology"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="officeRoom">{t("officeRoom")}</Label>
              <Input
                id="officeRoom"
                value={officeRoom}
                onChange={(e) => setOfficeRoom(e.target.value)}
                placeholder="Room 402, Building B"
              />
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="workingHours">{t("workingHours")}</Label>
              <Input
                id="workingHours"
                value={workingHours}
                onChange={(e) => setWorkingHours(e.target.value)}
                placeholder="Mon-Thu 09:00 - 17:00"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="profileDescription">{t("profileDescription")}</Label>
            <Textarea
              id="profileDescription"
              value={profileDescription}
              onChange={(e) => setProfileDescription(e.target.value)}
              placeholder="Clinical lead for telemedicine inquiries and robotic assistance..."
              rows={2}
            />
          </div>

          {/* Custom Q&A Pairs */}
          <div className="space-y-2 pt-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm font-semibold">{t("qaPairs")}</Label>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleAddQa}
                className="h-8 gap-1 text-xs"
              >
                <Plus className="size-3.5" />
                {t("addQa")}
              </Button>
            </div>

            {qaPairs.length === 0 ? (
              <p className="text-xs text-muted-foreground italic">
                Optional custom questions for authorized AI responses.
              </p>
            ) : (
              <div className="space-y-3">
                {qaPairs.map((qa, idx) => (
                  <div
                    key={idx}
                    className="flex items-start gap-2 rounded-lg border bg-slate-50/50 p-2.5"
                  >
                    <div className="flex-1 space-y-2">
                      <Input
                        placeholder={t("question")}
                        value={qa.question}
                        onChange={(e) =>
                          handleUpdateQa(idx, "question", e.target.value)
                        }
                        className="h-8 text-xs"
                      />
                      <Input
                        placeholder={t("answer")}
                        value={qa.answer}
                        onChange={(e) =>
                          handleUpdateQa(idx, "answer", e.target.value)
                        }
                        className="h-8 text-xs"
                      />
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      onClick={() => handleRemoveQa(idx)}
                      className="size-8 text-red-500 hover:text-red-700"
                    >
                      <Trash2 className="size-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <DialogFooter className="pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={submitting}
            >
              {t("cancel")}
            </Button>
            <Button type="submit" disabled={submitting} className="gap-2">
              {submitting && <Loader2 className="size-4 animate-spin" />}
              {submitting ? t("saving") : t("save")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
