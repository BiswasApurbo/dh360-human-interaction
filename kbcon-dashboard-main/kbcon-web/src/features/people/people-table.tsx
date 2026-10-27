"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import {
  Camera,
  MoreHorizontal,
  RefreshCw,
  Search,
  Shield,
  Trash2,
  UserCheck,
  UserX,
} from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { useRole } from "@/features/auth/use-role";
import type { PersonProfile } from "./types";

interface PeopleTableProps {
  people: PersonProfile[];
  loading: boolean;
  onSelectPerson: (person: PersonProfile) => void;
  onRequestDelete: (person: PersonProfile) => void;
  onRequestResetBiometrics: (person: PersonProfile) => void;
  onRequestEnroll: (person: PersonProfile) => void;
  onOpenAuthorization: () => void;
}

export function PeopleTable({
  people,
  loading,
  onSelectPerson,
  onRequestDelete,
  onRequestResetBiometrics,
  onRequestEnroll,
  onOpenAuthorization,
}: PeopleTableProps) {
  const t = useTranslations("People");
  const { role } = useRole();
  const isSuperAdmin = role === "superAdmin";
  const [searchTerm, setSearchTerm] = useState("");

  const filteredPeople = people.filter((p) => {
    const q = searchTerm.toLowerCase();
    return (
      p.display_name.toLowerCase().includes(q) ||
      (p.preferred_name && p.preferred_name.toLowerCase().includes(q)) ||
      p.person_id.toLowerCase().includes(q)
    );
  });

  return (
    <div className="space-y-4">
      {/* Search Bar & Test Authorization Action */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-2.5 top-2.5 size-4 text-muted-foreground" />
          <Input
            placeholder={t("searchPlaceholder")}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-8 h-9 text-sm"
          />
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={onOpenAuthorization}
          className="h-9 gap-1.5 border-sky-300 text-sky-800 hover:bg-sky-50 shadow-sm"
        >
          <Shield className="size-3.5 text-sky-600" />
          {t("testAuthorization")}
        </Button>
      </div>

      {/* Table */}
      <div className="rounded-2xl border border-sky-100 bg-white/80 shadow-sm overflow-hidden">
        <Table>
          <TableHeader className="bg-sky-50/50">
            <TableRow>
              <TableHead className="w-[280px] font-semibold text-slate-700">
                {t("name")}
              </TableHead>
              <TableHead className="font-semibold text-slate-700">
                {t("status")}
              </TableHead>
              <TableHead className="font-semibold text-slate-700">
                {t("consentStatus")}
              </TableHead>
              <TableHead className="font-semibold text-slate-700">
                Created
              </TableHead>
              <TableHead className="text-right font-semibold text-slate-700">
                {t("actions")}
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={5} className="h-32 text-center text-muted-foreground">
                  Loading registered people...
                </TableCell>
              </TableRow>
            ) : filteredPeople.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} className="h-36 text-center">
                  <div className="flex flex-col items-center justify-center space-y-1">
                    <UserX className="size-8 text-slate-300" />
                    <p className="text-sm font-medium text-slate-700">
                      {t("noPeople")}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {t("noPeopleDesc")}
                    </p>
                  </div>
                </TableCell>
              </TableRow>
            ) : (
              filteredPeople.map((person) => (
                <TableRow key={person.person_id} className="hover:bg-sky-50/30">
                  <TableCell className="font-medium">
                    <div className="flex items-center gap-2.5">
                      <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-sky-100 text-sky-700 font-semibold text-xs">
                        {person.display_name.charAt(0).toUpperCase()}
                      </div>
                      <div>
                        <div className="text-sm font-semibold text-slate-800">
                          {person.display_name}
                        </div>
                        {person.preferred_name && (
                          <div className="text-xs text-muted-foreground">
                            &quot;{person.preferred_name}&quot;
                          </div>
                        )}
                      </div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant="outline"
                      className={
                        person.status === "active"
                          ? "border-emerald-300 bg-emerald-50 text-emerald-700 text-xs font-normal"
                          : "border-slate-300 bg-slate-50 text-slate-600 text-xs font-normal"
                      }
                    >
                      {person.status}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant="outline"
                      className={
                        person.consent_status === "granted"
                          ? "border-sky-300 bg-sky-50 text-sky-700 text-xs font-normal"
                          : "border-amber-300 bg-amber-50 text-amber-700 text-xs font-normal"
                      }
                    >
                      <Shield className="mr-1 size-3" />
                      {person.consent_status}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-xs text-slate-500">
                    {person.created_at ? person.created_at.split("T")[0] : "—"}
                  </TableCell>
                  <TableCell className="text-right">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon" className="size-8">
                          <MoreHorizontal className="size-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end" className="w-48">
                        <DropdownMenuLabel className="text-xs">
                          {t("actions")}
                        </DropdownMenuLabel>
                        <DropdownMenuItem
                          onClick={() => onSelectPerson(person)}
                          className="gap-2 text-xs"
                        >
                          <UserCheck className="size-3.5" />
                          {t("viewProfile")}
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={() => onRequestEnroll(person)}
                          className="gap-2 text-xs"
                        >
                          <Camera className="size-3.5 text-sky-600" />
                          {t("enrollBiometrics")}
                        </DropdownMenuItem>

                        {isSuperAdmin && (
                          <>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem
                              onClick={() => onRequestResetBiometrics(person)}
                              className="gap-2 text-xs text-amber-600 focus:text-amber-700"
                            >
                              <RefreshCw className="size-3.5" />
                              {t("resetBiometrics")}
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              onClick={() => onRequestDelete(person)}
                              className="gap-2 text-xs text-red-600 focus:text-red-700"
                            >
                              <Trash2 className="size-3.5" />
                              {t("delete")}
                            </DropdownMenuItem>
                          </>
                        )}
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
