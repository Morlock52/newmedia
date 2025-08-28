export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="stylesheet" href="/cyberpunk.css" />
        <title>Cyberpunk Media Dashboard</title>
      </head>
      <body>{children}</body>
    </html>
  )
}