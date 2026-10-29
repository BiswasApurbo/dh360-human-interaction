import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AppSidebar } from "@/components/layout/app-sidebar";
import { Navbar } from "@/components/layout/navbar";
import { AuthGuard } from "@/features/auth/auth-guard";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <AuthGuard>
      <TooltipProvider>
        <SidebarProvider>
          <AppSidebar />
          <SidebarInset className="h-svh max-h-svh overflow-hidden flex flex-col bg-gradient-to-br from-sky-100 via-sky-50 to-sky-200">
            <Navbar />
            <main className="relative flex min-h-0 flex-1 flex-col overflow-y-auto p-2.5 sm:p-3.5 lg:p-4">
              {/* হালকা সাজানোর বৃত্ত */}
              <div className="pointer-events-none absolute -right-24 top-10 size-72 rounded-full bg-sky-200/40 blur-3xl" />
              <div className="pointer-events-none absolute -bottom-24 left-10 size-72 rounded-full bg-cyan-200/30 blur-3xl" />
              <div className="relative flex h-full min-h-0 flex-1 flex-col">{children}</div>
            </main>
          </SidebarInset>
        </SidebarProvider>
      </TooltipProvider>
    </AuthGuard>
  );
}
