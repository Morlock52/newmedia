export const metadata = {
  title: 'Media Dashboard',
  description: 'Unified media server dashboard'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
