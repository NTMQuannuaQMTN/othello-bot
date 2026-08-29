export async function api(path, body) {
  const opts =
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        };
  const res = await fetch("/api" + path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

export const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);

export function sanToIdx(san) {
  if (!san || san === "pass") return -1;
  const c = san.charCodeAt(0) - 97;
  const r = parseInt(san[1], 10) - 1;
  return r >= 0 && r < 8 && c >= 0 && c < 8 ? r * 8 + c : -1;
}
