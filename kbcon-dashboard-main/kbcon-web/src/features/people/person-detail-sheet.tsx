"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import {
  Calendar,
  CheckCircle2,
  Clock,
  Edit2,
  Loader2,
  Save,
  Shield,
  User,
  X,
} from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useRole } from "@/features/auth/use-role";
import type {
  ConsentStatus,
  PersonProfile,
  PersonStatus,
  UpdatePersonInput,
} from "./types";

interface PersonDetailSheetProps {
  person: PersonProfile | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onUpdate: (personId: string, data: UpdatePersonInput) => Promise<PersonProfile>;
}

function PersonDetailContent({
  person,
  onUpdate,
}: {
  person: PersonProfile;
  onUpdate: (personId: string, data: UpdatePersonInput) => Promise<PersonProfile>;
}) {
  const t = useTranslations("People");
  const { role: userRole } = useRole();
  const isSuperAdmin = userRole === "superAdmin";

  const [isEditing, setIsEditing] = useState(false);
  const [displayName, setDisplayName] = useState(person.display_name || "");
  const [preferredName, setPreferredName] = useState(
    person.preferred_name || ""
  );
  const [status] = useState<PersonStatus>(
    (person.status as PersonStatus) || "active"
  );
  const [consentStatus, setConsentStatus] = useState<ConsentStatus>(
    (person.consent_status as ConsentStatus) || "granted"
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSave = async () => {
    if (!displayName.trim()) {
      setError("Display name cannot be empty.");
      return;
    }

    setSaving(true);
    setError(null);
    try {
      const updateData: UpdatePersonInput = {
        display_name: displayName.trim(),
        preferred_name: preferredName.trim() || null,
        status,
      };

      if (isSuperAdmin) {
        updateData.consent_status = consentStatus;
      }

      await onUpdate(person.person_id, updateData);
      setIsEditing(false);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to update profile");
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <SheetHeader className="pb-4">
        <div className="flex items-center justify-between">
          <SheetTitle className="flex items-center gap-2">
            <User className="size-5 text-sky-600" />
            {person.display_name}
          </SheetTitle>
          {!isEditing ? (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsEditing(true)}
              className="h-8 gap-1.5 text-xs"
            >
              <Edit2 className="size-3.5" />
              Edit
            </Button>
          ) : (
            <div className="flex items-center gap-1.5">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsEditing(false)}
                disabled={saving}
                className="h-8 text-xs"
              >
                <X className="size-3.5" />
              </Button>
              <Button
                size="sm"
                onClick={handleSave}
                disabled={saving}
                className="h-8 gap-1.5 text-xs"
              >
                {saving ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <Save className="size-3.5" />
                )}
                {t("save")}
              </Button>
            </div>
          )}
        </div>
        <SheetDescription className="font-mono text-xs">
          ID: {person.person_id}
        </SheetDescription>
      </SheetHeader>

      {error && (
        <div className="mb-4 rounded-md bg-red-50 p-2.5 text-xs text-red-600 border border-red-200">
          {error}
        </div>
      )}

      <div className="space-y-5 text-sm">
        {/* Status Badges */}
        <div className="flex flex-wrap items-center gap-2">
          <Badge
            variant="outline"
            className={
              person.status === "active"
                ? "border-emerald-500 bg-emerald-50 text-emerald-700"
                : "border-slate-300 bg-slate-50 text-slate-600"
            }
          >
            <CheckCircle2 className="mr-1 size-3" />
            {person.status}
          </Badge>

          <Badge
            variant="outline"
            className={
              person.consent_status === "granted"
                ? "border-sky-500 bg-sky-50 text-sky-700"
                : "border-amber-400 bg-amber-50 text-amber-700"
            }
          >
            <Shield className="mr-1 size-3" />
            Consent: {person.consent_status}
          </Badge>
        </div>

        <Separator />

        {/* Profile Fields */}
        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">{t("name")}</Label>
            {isEditing ? (
              <Input
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
              />
            ) : (
              <div className="font-medium text-slate-800">
                {person.display_name}
              </div>
            )}
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">
              {t("preferredName")}
            </Label>
            {isEditing ? (
              <Input
                value={preferredName}
                onChange={(e) => setPreferredName(e.target.value)}
                placeholder="None"
              />
            ) : (
              <div className="text-slate-700">
                {person.preferred_name || "—"}
              </div>
            )}
          </div>

          {isEditing && isSuperAdmin && (
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">
                {t("consentStatus")} (Super Admin)
              </Label>
              <select
                value={consentStatus}
                onChange={(e) => setConsentStatus(e.target.value as ConsentStatus)}
                className="w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm"
              >
                <option value="granted">Granted</option>
                <option value="revoked">Revoked</option>
                <option value="pending">Pending</option>
              </select>
            </div>
          )}
        </div>

        <Separator />

        {/* Metadata & Timestamps */}
        <div className="space-y-2 rounded-lg bg-slate-50 p-3 text-xs text-slate-600">
          <div className="flex items-center gap-2">
            <Calendar className="size-3.5 text-muted-foreground" />
            <span>Created: {person.created_at || "—"}</span>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="size-3.5 text-muted-foreground" />
            <span>Updated: {person.updated_at || "—"}</span>
          </div>
          {person.consent_timestamp && (
            <div className="flex items-center gap-2">
              <Shield className="size-3.5 text-muted-foreground" />
              <span>Consent Given: {person.consent_timestamp}</span>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export function PersonDetailSheet({
  person,
  open,
  onOpenChange,
  onUpdate,
}: PersonDetailSheetProps) {
  if (!person) return null;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full sm:max-w-md overflow-y-auto">
        <PersonDetailContent
          key={person.person_id}
          person={person}
          onUpdate={onUpdate}
        />
      </SheetContent>
    </Sheet>
  );
}
