interface WarningsProps {
  warnings: string[]
}

export function WarningBanners({ warnings }: WarningsProps) {
  if (warnings.length === 0) return null
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {warnings.map((w) => (
        <div className="banner banner-warn" key={w}>
          <span>&#9888;</span>
          <span>{w}</span>
        </div>
      ))}
    </div>
  )
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="banner banner-error">
      <span>&#10060;</span>
      <span>{message}</span>
    </div>
  )
}
