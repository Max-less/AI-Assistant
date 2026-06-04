// Base URL for the web backend. Defaults to same-origin so nginx can reverse-proxy
// /api in production. In dev, Vite's `server.proxy` forwards /api -> :8001.
// Override with VITE_API_BASE when needed (e.g. for split deployments).
export const API_BASE: string = import.meta.env.VITE_API_BASE ?? ""
