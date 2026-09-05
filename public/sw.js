/* Pomodoro Camera — Service Worker (offline cache, installable PWA) */
const CACHE = "pomodoro-v1";
const ASSETS = [
  "/",
  "/index.html",
  "/manifest.json",
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(ASSETS).catch(() => {}))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  // Network-first for navigations, cache-first for assets
  if (req.mode === "navigate") {
    e.respondWith(
      fetch(req).then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
        return res;
      }).catch(() => caches.match(req).then((c) => c || caches.match("/index.html")))
    );
    return;
  }
  // Cache-first for same-origin GET assets
  if (req.method === "GET" && new URL(req.url).origin === self.location.origin) {
    e.respondWith(
      caches.match(req).then((cached) => {
        if (cached) return cached;
        return fetch(req).then((res) => {
          if (res.ok) {
            const copy = res.clone();
            caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
          }
          return res;
        });
      })
    );
  }
});

// Push notifications (for future server push)
self.addEventListener("push", (e) => {
  const data = e.data ? e.data.json() : { title: "Pomodoro Camera", body: "Time's up!" };
  e.waitUntil(
    self.registration.showNotification(data.title || "Pomodoro Camera", {
      body: data.body || "Session complete",
      icon: "/icon-192.png",
      badge: "/icon-192.png",
      tag: "pomodoro",
      renotify: true,
    })
  );
});

self.addEventListener("notificationclick", (e) => {
  e.notification.close();
  e.waitUntil(
    self.clients.matchAll({ type: "window" }).then((clients) => {
      for (const c of clients) if ("focus" in c) return c.focus();
      return self.clients.openWindow("/");
    })
  );
});
