"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import { Plus, RefreshCw, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { usePeople } from "./use-people";
import { PeopleTable } from "./people-table";
import { CreatePersonDialog } from "./create-person-dialog";
import { DeletePersonDialog } from "./delete-person-dialog";
import { ResetBiometricsDialog } from "./reset-biometrics-dialog";
import { PersonDetailSheet } from "./person-detail-sheet";
import { EnrollmentDialog } from "./enrollment-dialog";
import { AuthorizationDialog } from "./authorization-dialog";
import type { PersonProfile } from "./types";

export function PeoplePageContent() {
  const t = useTranslations("People");
  const {
    people,
    loading,
    error,
    refresh,
    createPerson,
    updatePerson,
    deletePerson,
    resetBiometrics,
  } = usePeople();

  const [createOpen, setCreateOpen] = useState(false);
  const [selectedPerson, setSelectedPerson] = useState<PersonProfile | null>(null);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [personToDelete, setPersonToDelete] = useState<PersonProfile | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [personToReset, setPersonToReset] = useState<PersonProfile | null>(null);
  const [resetOpen, setResetOpen] = useState(false);
  const [personToEnroll, setPersonToEnroll] = useState<PersonProfile | null>(null);
  const [enrollOpen, setEnrollOpen] = useState(false);
  const [authOpen, setAuthOpen] = useState(false);

  const handleSelectPerson = (person: PersonProfile) => {
    setSelectedPerson(person);
    setSheetOpen(true);
  };

  const handleRequestDelete = (person: PersonProfile) => {
    setPersonToDelete(person);
    setDeleteOpen(true);
  };

  const handleRequestResetBiometrics = (person: PersonProfile) => {
    setPersonToReset(person);
    setResetOpen(true);
  };

  const handleRequestEnroll = (person: PersonProfile) => {
    setPersonToEnroll(person);
    setEnrollOpen(true);
  };

  return (
    <div className="space-y-5">
      {/* Page Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-sky-900 sm:text-2xl">
            {t("title")}
          </h1>
          <p className="text-sm text-sky-800/80">{t("subtitle")}</p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <motion.button
            type="button"
            onClick={() => refresh()}
            disabled={loading}
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            className="flex h-10 items-center gap-2 rounded-xl border border-sky-300 bg-white px-3.5 text-sm font-semibold text-sky-800 shadow-sm hover:bg-sky-50 disabled:opacity-60"
          >
            <RefreshCw
              className={`size-4 ${loading ? "animate-spin text-sky-500" : "text-sky-600"}`}
            />
            {t("refresh")}
          </motion.button>

          <motion.button
            type="button"
            onClick={() => setCreateOpen(true)}
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            className="flex h-10 items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-sky-600 px-4 text-sm font-semibold text-white shadow-md shadow-sky-300/60"
          >
            <Plus className="size-4" />
            {t("registerPerson")}
          </motion.button>
        </div>
      </div>

      {/* Error Alert if any */}
      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50/80 p-4 text-sm text-red-700 flex items-start gap-3">
          <AlertCircle className="size-5 shrink-0 text-red-500 mt-0.5" />
          <div className="flex-1">
            <p className="font-semibold">Connection Notice</p>
            <p className="text-xs text-red-600 mt-0.5">{error}</p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={refresh}
            className="h-7 text-xs bg-white"
          >
            Retry
          </Button>
        </div>
      )}

      {/* Main Table */}
      <PeopleTable
        people={people}
        loading={loading}
        onSelectPerson={handleSelectPerson}
        onRequestDelete={handleRequestDelete}
        onRequestResetBiometrics={handleRequestResetBiometrics}
        onRequestEnroll={handleRequestEnroll}
        onOpenAuthorization={() => setAuthOpen(true)}
        onRequestRegister={() => setCreateOpen(true)}
      />

      {/* Dialogs & Sheets */}
      <CreatePersonDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        onSubmit={createPerson}
      />

      <PersonDetailSheet
        person={selectedPerson}
        open={sheetOpen}
        onOpenChange={setSheetOpen}
        onUpdate={updatePerson}
      />

      <DeletePersonDialog
        person={personToDelete}
        open={deleteOpen}
        onOpenChange={setDeleteOpen}
        onConfirm={deletePerson}
      />

      <ResetBiometricsDialog
        person={personToReset}
        open={resetOpen}
        onOpenChange={setResetOpen}
        onConfirm={resetBiometrics}
      />

      <EnrollmentDialog
        person={personToEnroll}
        open={enrollOpen}
        onOpenChange={setEnrollOpen}
        onSuccess={refresh}
      />

      <AuthorizationDialog
        open={authOpen}
        onOpenChange={setAuthOpen}
      />
    </div>
  );
}
