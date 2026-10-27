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
          <SidebarInset className="bg-gradient-to-br from-sky-100 via-sky-50 to-sky-200">
            <Navbar />
            <main className="relative min-h-screen flex-1 overflow-hidden p-4 sm:p-6 lg:p-8">
              {/* হালকা সাজানোর বৃত্ত */}
              <div className="pointer-events-none absolute -right-24 top-10 size-72 rounded-full bg-sky-200/40 blur-3xl" />
              <div className="pointer-events-none absolute -bottom-24 left-10 size-72 rounded-full bg-cyan-200/30 blur-3xl" />
              <div className="relative">{children}</div>
            </main>
          </SidebarInset>
        </SidebarProvider>
      </TooltipProvider>
    </AuthGuard>
  );
}
