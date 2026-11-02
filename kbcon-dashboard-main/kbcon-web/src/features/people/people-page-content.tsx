"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { motion } from "motion/react";
import { Plus, RefreshCw, AlertCircle, ShieldAlert, Bot, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useRole } from "@/features/auth/use-role";
import { useRobots } from "@/features/robots/use-robots";
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
  const { role, loading: roleLoading } = useRole();
  const { robots } = useRobots(role);
  const [selectedRobotFilter, setSelectedRobotFilter] = useState<string>("");

  const {
    people,
    loading,
    error,
    refresh,
    createPerson,
    updatePerson,
    deletePerson,
    resetBiometrics,
  } = usePeople(selectedRobotFilter || undefined);

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

  if (roleLoading) {
    return (
      <div className="flex justify-center py-16">
        <Loader2 className="size-8 animate-spin text-sky-500" />
      </div>
    );
  }

  if (role === "superAdmin") {
    return (
      <div className="flex items-start gap-3 rounded-2xl bg-amber-50 p-6 text-sm text-amber-900 ring-1 ring-amber-200">
        <ShieldAlert className="size-6 shrink-0 text-amber-600 mt-0.5" />
        <div>
          <h3 className="font-bold text-base text-amber-950 mb-1">
            Access Restricted to Super Admin
          </h3>
          <p className="text-amber-800 leading-relaxed">
            Super Admins are restricted from accessing tenant-private People and biometric records. Company/tenant individuals and biometrics are managed exclusively by designated tenant Admins.
          </p>
        </div>
      </div>
    );
  }

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
          {robots.length > 0 && (
            <div className="flex items-center gap-1.5 rounded-xl border border-sky-300 bg-white px-2.5 py-1.5 text-xs text-sky-900 shadow-sm">
              <Bot className="size-3.5 text-sky-600" />
              <select
                value={selectedRobotFilter}
                onChange={(e) => setSelectedRobotFilter(e.target.value)}
                className="bg-transparent font-medium outline-none cursor-pointer"
              >
                <option value="">All Assigned Robots</option>
                {robots.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.name} ({r.id})
                  </option>
                ))}
              </select>
            </div>
          )}

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
        robots={robots}
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
