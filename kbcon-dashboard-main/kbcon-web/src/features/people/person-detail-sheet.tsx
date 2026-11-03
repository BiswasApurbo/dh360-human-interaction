"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import {
  AlertTriangle,
  Calendar,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Edit2,
  History,
  Loader2,
  Save,
  Shield,
  ShieldAlert,
  ShieldCheck,
  User,
  X,
  XCircle,
} from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
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
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import type {
  ConsentAuditRecord,
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
  onUpdateConsent: (
    personId: string,
    status: string,
    notes?: string
  ) => Promise<{ person: PersonProfile; audit: ConsentAuditRecord }>;
  onGetConsentHistory: (personId: string) => Promise<ConsentAuditRecord[]>;
}

function PersonDetailContent({
  person,
  onUpdate,
  onUpdateConsent,
  onGetConsentHistory,
}: {
  person: PersonProfile;
  onUpdate: (personId: string, data: UpdatePersonInput) => Promise<PersonProfile>;
  onUpdateConsent: (
    personId: string,
    status: string,
    notes?: string
  ) => Promise<{ person: PersonProfile; audit: ConsentAuditRecord }>;
  onGetConsentHistory: (personId: string) => Promise<ConsentAuditRecord[]>;
}) {
  const t = useTranslations("People");

  // Profile metadata editing
  const [profile, setProfile] = useState<PersonProfile>(person);
  const [isEditing, setIsEditing] = useState(false);
  const [displayName, setDisplayName] = useState(person.display_name || "");
  const [preferredName, setPreferredName] = useState(person.preferred_name || "");
  const [status] = useState<PersonStatus>((person.status as PersonStatus) || "active");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Consent & Audit Trail State
  const [auditHistory, setAuditHistory] = useState<ConsentAuditRecord[]>([]);
  const [auditLoading, setAuditLoading] = useState(false);
  const [showHistory, setShowHistory] = useState(false);

  // Consent Actions Dialogs
  const [withdrawDialogOpen, setWithdrawDialogOpen] = useState(false);
  const [grantDialogOpen, setGrantDialogOpen] = useState(false);
  const [refuseDialogOpen, setRefuseDialogOpen] = useState(false);
  const [consentNotes, setConsentNotes] = useState("");
  const [consentActionLoading, setConsentActionLoading] = useState(false);

  // Load audit history
  useEffect(() => {
    let active = true;
    const fetchHistory = async () => {
      setAuditLoading(true);
      try {
        const history = await onGetConsentHistory(person.person_id);
        if (active) {
          setAuditHistory(history);
        }
      } catch (err) {
        console.warn("[PersonDetailSheet] Failed to load consent audit history:", err);
      } finally {
        if (active) {
          setAuditLoading(false);
        }
      }
    };
    void fetchHistory();
    return () => {
      active = false;
    };
  }, [person.person_id, onGetConsentHistory]);

  const handleSave = async () => {
    if (!displayName.trim()) {
      setError("Display name cannot be empty.");
      return;
    }

    setSaving(true);
    setError(null);
    try {
      // NOTE: Profile editing must NOT silently mutate consent status.
      // Consent is managed strictly through dedicated auditable consent controls.
      const updateData: UpdatePersonInput = {
        display_name: displayName.trim(),
        preferred_name: preferredName.trim() || null,
        status,
      };

      const updated = await onUpdate(profile.person_id, updateData);
      setProfile(updated);
      setIsEditing(false);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to update profile");
    } finally {
      setSaving(false);
    }
  };

  const executeConsentChange = async (newStatus: ConsentStatus) => {
    setConsentActionLoading(true);
    setError(null);
    try {
      const res = await onUpdateConsent(
        profile.person_id,
        newStatus,
        consentNotes.trim() || undefined
      );
      setProfile(res.person);
      setAuditHistory((prev) => [res.audit, ...prev]);
      setConsentNotes("");
      setWithdrawDialogOpen(false);
      setGrantDialogOpen(false);
      setRefuseDialogOpen(false);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to record consent transition.");
    } finally {
      setConsentActionLoading(false);
    }
  };

  const getConsentBadge = (statusValue: string) => {
    switch (statusValue) {
      case "granted":
        return (
          <Badge
            variant="outline"
            className="border-emerald-500 bg-emerald-50 text-emerald-700 font-medium"
          >
            <ShieldCheck className="mr-1 size-3.5" />
            Granted
          </Badge>
        );
      case "withdrawn":
        return (
          <Badge
            variant="outline"
            className="border-red-400 bg-red-50 text-red-700 font-medium"
          >
            <ShieldAlert className="mr-1 size-3.5" />
            Withdrawn
          </Badge>
        );
      case "refused":
        return (
          <Badge
            variant="outline"
            className="border-slate-400 bg-slate-50 text-slate-700 font-medium"
          >
            <XCircle className="mr-1 size-3.5" />
            Refused
          </Badge>
        );
      case "pending":
      default:
        return (
          <Badge
            variant="outline"
            className="border-amber-400 bg-amber-50 text-amber-700 font-medium"
          >
            <AlertTriangle className="mr-1 size-3.5" />
            Pending
          </Badge>
        );
    }
  };

  return (
    <>
      <SheetHeader className="pb-4">
        <div className="flex items-center justify-between">
          <SheetTitle className="flex items-center gap-2">
            <User className="size-5 text-sky-600" />
            {profile.display_name}
          </SheetTitle>
          {!isEditing ? (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsEditing(true)}
              className="h-8 gap-1.5 text-xs border-sky-200 text-sky-800 hover:bg-sky-50"
            >
              <Edit2 className="size-3.5" />
              Edit Profile
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
                className="h-8 gap-1.5 text-xs bg-sky-600 hover:bg-sky-700 text-white"
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
          ID: {profile.person_id}
        </SheetDescription>
      </SheetHeader>

      {error && (
        <div className="mb-4 rounded-md bg-red-50 p-2.5 text-xs text-red-600 border border-red-200">
          {error}
        </div>
      )}

      <div className="space-y-5 text-sm pb-8">
        {/* Status Badges */}
        <div className="flex flex-wrap items-center gap-2">
          <Badge
            variant="outline"
            className={
              profile.status === "active"
                ? "border-emerald-500 bg-emerald-50 text-emerald-700"
                : "border-slate-300 bg-slate-50 text-slate-600"
            }
          >
            <CheckCircle2 className="mr-1 size-3" />
            Status: {profile.status}
          </Badge>

          {getConsentBadge(profile.consent_status)}
        </div>

        <Separator />

        {/* Profile Basic Info */}
        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">{t("name")}</Label>
            {isEditing ? (
              <Input
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                className="h-9"
              />
            ) : (
              <div className="font-medium text-slate-800">
                {profile.display_name}
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
                className="h-9"
              />
            ) : (
              <div className="text-slate-700">
                {profile.preferred_name || "—"}
              </div>
            )}
          </div>
        </div>

        <Separator />

        {/* Dedicated Consent Management Section */}
        <div className="rounded-xl border border-sky-100 bg-gradient-to-b from-sky-50/50 to-white p-4 space-y-3.5 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sky-950 font-bold text-sm">
              <Shield className="size-4 text-sky-600" />
              <span>Consent Management</span>
            </div>
            {getConsentBadge(profile.consent_status)}
          </div>

          <p className="text-xs text-slate-600 leading-relaxed">
            {profile.consent_status === "granted" &&
              "Explicit consent is recorded. This person is authorized for biometric identification and camera face-recognition on assigned robots."}
            {profile.consent_status === "pending" &&
              "Consent is currently pending. Biometric enrollment and active facial identification cannot be initiated until explicit consent is granted."}
            {profile.consent_status === "withdrawn" &&
              "Consent was withdrawn. Biometric recognition is strictly deactivated and camera enrollment is blocked."}
            {profile.consent_status === "refused" &&
              "Subject refused biometric processing. Biometric operations and identification are blocked."}
          </p>

          {profile.consent_timestamp && (
            <div className="text-[11px] text-slate-500 font-mono">
              Recorded at: {profile.consent_timestamp}
            </div>
          )}

          {/* Consent Control Buttons */}
          <div className="flex flex-wrap gap-2 pt-1">
            {profile.consent_status !== "granted" ? (
              <Button
                size="sm"
                onClick={() => {
                  setConsentNotes("");
                  setGrantDialogOpen(true);
                }}
                className="h-8 gap-1.5 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold shadow-sm"
              >
                <ShieldCheck className="size-3.5" />
                Grant Consent
              </Button>
            ) : (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setConsentNotes("");
                  setWithdrawDialogOpen(true);
                }}
                className="h-8 gap-1.5 border-red-300 text-red-700 hover:bg-red-50 text-xs font-semibold"
              >
                <ShieldAlert className="size-3.5" />
                Withdraw Consent
              </Button>
            )}

            {profile.consent_status !== "refused" && profile.consent_status !== "withdrawn" && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setConsentNotes("");
                  setRefuseDialogOpen(true);
                }}
                className="h-8 text-xs text-slate-600 hover:bg-slate-100"
              >
                Record Refusal
              </Button>
            )}
          </div>

          {/* Immutable Audit Trail Section */}
          <div className="pt-2 border-t border-sky-100/70">
            <button
              type="button"
              onClick={() => setShowHistory((prev) => !prev)}
              className="flex items-center justify-between w-full text-xs font-semibold text-sky-900 hover:text-sky-700 py-1"
            >
              <div className="flex items-center gap-1.5">
                <History className="size-3.5 text-sky-600" />
                <span>Consent Audit History</span>
                <span className="rounded-full bg-sky-100 px-1.5 py-0.2 text-[10px] text-sky-800">
                  {auditHistory.length}
                </span>
              </div>
              {showHistory ? (
                <ChevronUp className="size-3.5" />
              ) : (
                <ChevronDown className="size-3.5" />
              )}
            </button>

            {showHistory && (
              <div className="mt-2.5 space-y-2 max-h-48 overflow-y-auto pr-1">
                {auditLoading ? (
                  <div className="py-3 text-center text-xs text-slate-500">
                    <Loader2 className="inline size-3.5 animate-spin mr-1 text-sky-600" />
                    Loading audit trail...
                  </div>
                ) : auditHistory.length === 0 ? (
                  <p className="text-xs text-slate-500 italic py-1">
                    No consent transitions recorded in audit log.
                  </p>
                ) : (
                  auditHistory.map((entry) => (
                    <div
                      key={entry.audit_id}
                      className="rounded-lg border border-slate-200/80 bg-white p-2 text-xs space-y-1 shadow-2xs"
                    >
                      <div className="flex items-center justify-between text-[11px] text-slate-500">
                        <span className="font-mono">
                          {entry.timestamp.replace("T", " ").split(".")[0]}
                        </span>
                        <span className="font-medium text-slate-700">
                          by: {entry.changed_by_name || entry.changed_by_uid}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5 font-semibold text-slate-800">
                        <span className="capitalize">{entry.previous_status}</span>
                        <span className="text-slate-400">→</span>
                        <span className="capitalize font-bold text-sky-800">
                          {entry.new_status}
                        </span>
                      </div>
                      {entry.notes && (
                        <p className="text-[11px] text-slate-600 italic bg-slate-50 rounded px-1.5 py-0.5">
                          &ldquo;{entry.notes}&rdquo;
                        </p>
                      )}
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </div>

        <Separator />

        {/* Timestamps */}
        <div className="space-y-2 rounded-lg bg-slate-50 p-3 text-xs text-slate-600">
          <div className="flex items-center gap-2">
            <Calendar className="size-3.5 text-muted-foreground" />
            <span>Created: {profile.created_at || "—"}</span>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="size-3.5 text-muted-foreground" />
            <span>Updated: {profile.updated_at || "—"}</span>
          </div>
        </div>
      </div>

      {/* Confirmation Dialog: Withdraw Consent */}
      <AlertDialog open={withdrawDialogOpen} onOpenChange={setWithdrawDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2 text-red-600">
              <ShieldAlert className="size-5" />
              Withdraw Biometric Consent
            </AlertDialogTitle>
            <AlertDialogDescription className="text-slate-600 space-y-2">
              <span className="block font-medium text-slate-800">
                Are you sure you want to withdraw consent for {profile.display_name}?
              </span>
              <span className="block text-xs">
                Withdrawing consent disables facial identification and biometric enrollment for this person on all assigned robots. An immutable record of this withdrawal will be saved to the audit log.
              </span>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="py-2 space-y-1.5">
            <Label className="text-xs font-semibold text-slate-700">
              Reason / Notes (optional)
            </Label>
            <Textarea
              placeholder="e.g. Subject requested withdrawal via email"
              value={consentNotes}
              onChange={(e) => setConsentNotes(e.target.value)}
              className="text-xs min-h-[60px]"
            />
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={consentActionLoading}>
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={(e) => {
                e.preventDefault();
                void executeConsentChange("withdrawn");
              }}
              disabled={consentActionLoading}
              className="bg-red-600 hover:bg-red-700 text-white font-semibold"
            >
              {consentActionLoading ? (
                <Loader2 className="size-3.5 animate-spin mr-1" />
              ) : (
                <ShieldAlert className="size-3.5 mr-1" />
              )}
              Withdraw Consent
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Dialog: Grant Explicit Consent */}
      <Dialog open={grantDialogOpen} onOpenChange={setGrantDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-emerald-700">
              <ShieldCheck className="size-5" />
              Record Granted Consent
            </DialogTitle>
            <DialogDescription className="text-xs text-slate-600">
              Confirm that {profile.display_name} has provided explicit agreement for biometric capture, facial feature storage, and recognition.
            </DialogDescription>
          </DialogHeader>
          <div className="py-2 space-y-1.5">
            <Label className="text-xs font-semibold text-slate-700">
              Consent Agreement Notes / Documentation (optional)
            </Label>
            <Textarea
              placeholder="e.g. Signed written form ref #2026-A12, verified in person"
              value={consentNotes}
              onChange={(e) => setConsentNotes(e.target.value)}
              className="text-xs min-h-[70px]"
            />
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setGrantDialogOpen(false)}
              disabled={consentActionLoading}
              className="text-xs"
            >
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={() => void executeConsentChange("granted")}
              disabled={consentActionLoading}
              className="bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold"
            >
              {consentActionLoading ? (
                <Loader2 className="size-3.5 animate-spin mr-1" />
              ) : (
                <ShieldCheck className="size-3.5 mr-1" />
              )}
              Record Granted Consent
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Dialog: Record Refusal */}
      <Dialog open={refuseDialogOpen} onOpenChange={setRefuseDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-slate-800">
              <XCircle className="size-5 text-slate-600" />
              Record Refused Consent
            </DialogTitle>
            <DialogDescription className="text-xs text-slate-600">
              Record that {profile.display_name} has explicitly declined consent for biometric feature extraction.
            </DialogDescription>
          </DialogHeader>
          <div className="py-2 space-y-1.5">
            <Label className="text-xs font-semibold text-slate-700">
              Notes (optional)
            </Label>
            <Textarea
              placeholder="e.g. Declined biometric enrollment during onboarding"
              value={consentNotes}
              onChange={(e) => setConsentNotes(e.target.value)}
              className="text-xs min-h-[60px]"
            />
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setRefuseDialogOpen(false)}
              disabled={consentActionLoading}
              className="text-xs"
            >
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={() => void executeConsentChange("refused")}
              disabled={consentActionLoading}
              className="bg-slate-700 hover:bg-slate-800 text-white text-xs font-semibold"
            >
              {consentActionLoading ? (
                <Loader2 className="size-3.5 animate-spin mr-1" />
              ) : null}
              Record Refusal
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

export function PersonDetailSheet({
  person,
  open,
  onOpenChange,
  onUpdate,
  onUpdateConsent,
  onGetConsentHistory,
}: PersonDetailSheetProps) {
  if (!person) return null;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full sm:max-w-md overflow-y-auto">
        <PersonDetailContent
          key={person.person_id}
          person={person}
          onUpdate={onUpdate}
          onUpdateConsent={onUpdateConsent}
          onGetConsentHistory={onGetConsentHistory}
        />
      </SheetContent>
    </Sheet>
  );
}
