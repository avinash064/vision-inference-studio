import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { SystemMetrics } from '../../types'

function Ring({ label, value, detail, color }: { label: string; value: number; detail: string; color: string }) {
  const percent = Math.max(0, Math.min(100, value || 0))
  return <div className="metric" title={`${label}: ${percent.toFixed(0)}% · ${detail}`}>
    <div className="metric-ring" style={{ background: `conic-gradient(${color} ${percent * 3.6}deg, #222a39 0deg)` }}><span>{percent.toFixed(0)}</span></div>
    <div><strong>{label}</strong><small>{detail}</small></div>
  </div>
}

export function SystemMonitor() {
  const [metrics, setMetrics] = useState<SystemMetrics>()
  useEffect(() => {
    let active = true
    const update = () => api.systemMetrics().then((value) => active && setMetrics(value)).catch(() => undefined)
    update(); const timer = window.setInterval(update, 1500)
    return () => { active = false; window.clearInterval(timer) }
  }, [])
  if (!metrics) return <div className="system-monitor loading">SYSTEM METRICS…</div>
  return <div className="system-monitor">
    <Ring label="CPU" value={metrics.cpu.percent} detail={`${metrics.cpu.memory_percent.toFixed(0)}% RAM`} color="#856cff" />
    <Ring label="GPU" value={metrics.gpu?.percent || 0} detail={metrics.gpu ? `${metrics.gpu.memory_percent.toFixed(0)}% VRAM · ${metrics.gpu.temperature_c}°` : 'Unavailable'} color="#43d09a" />
  </div>
}
