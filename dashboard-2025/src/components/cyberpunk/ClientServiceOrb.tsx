'use client'

import dynamic from 'next/dynamic'

const ServiceOrb = dynamic(
  () => import('./ServiceOrb').then((mod) => mod.ServiceOrb),
  { 
    ssr: false,
    loading: () => (
      <div className="w-[150px] h-[150px] flex items-center justify-center">
        <div className="w-32 h-32 rounded-full border-2 border-cyan-500/50 animate-pulse" />
      </div>
    )
  }
)

export default ServiceOrb
