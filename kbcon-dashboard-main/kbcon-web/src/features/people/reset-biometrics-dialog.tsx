"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { Loader2, RefreshCw } from "lucide-react";
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
import type { PersonProfile } from "./types";

interface ResetBiometricsDialogProps {
  person: PersonProfile | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: (personId: string) => Promise<void>;
}

export function ResetBiometricsDialog({
  person,
  open,
  onOpenChange,
  onConfirm,
}: ResetBiometricsDialogProps) {
  const t = useTranslations("People");
  const [resetting, setResetting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!person) return null;

  const handleConfirm = async () => {
    setResetting(true);
    setError(null);
    try {
      await onConfirm(person.person_id);
      onOpenChange(false);
    } catch (err: unknown) {
      setError(
        err instanceof Error ? err.message : "Failed to reset biometrics"
      );
    } finally {
      setResetting(false);
    }
  };

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle className="flex items-center gap-2 text-amber-600">
            <RefreshCw className="size-5" />
            {t("resetConfirmTitle")}
          </AlertDialogTitle>
          <AlertDialogDescription>
            {t("resetConfirmDesc", { name: person.display_name })}
          </AlertDialogDescription>
        </AlertDialogHeader>

        {error && (
          <div className="rounded-md bg-red-50 p-2.5 text-xs text-red-600 border border-red-200">
            {error}
          </div>
        )}

        <AlertDialogFooter>
          <AlertDialogCancel disabled={resetting}>{t("cancel")}</AlertDialogCancel>
          <AlertDialogAction
            onClick={(e) => {
              e.preventDefault();
              handleConfirm();
            }}
            disabled={resetting}
            className="bg-amber-600 hover:bg-amber-700 text-white gap-2"
          >
            {resetting && <Loader2 className="size-4 animate-spin" />}
            {t("resetBiometrics")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
