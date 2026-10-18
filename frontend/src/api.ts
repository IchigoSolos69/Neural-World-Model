export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: { ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), ...(init?.method && init.method !== 'GET' ? { 'X-NWM-Client': 'local' } : {}), ...init?.headers },
  });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(typeof data?.detail === 'string' ? data.detail : typeof data?.error === 'string' ? data.error : `Request failed (${response.status}).`);
  }
  if (response.status === 204) return undefined as T;
  return response.json();
}
