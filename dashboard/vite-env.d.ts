/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string
  readonly VITE_WS_URL: string
  readonly VITE_ENABLE_AI_ASSISTANT: string
  readonly VITE_ENABLE_3D_VISUALIZATION: string
  readonly VITE_ENABLE_VOICE_CONTROL: string
  readonly VITE_ANALYTICS_ID: string
  readonly VITE_ENV: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}