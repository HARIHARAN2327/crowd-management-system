delete L.Icon.Default.prototype._getIconUrl;

L.Icon.Default.mergeOptions({
  iconRetinaUrl: '/static/leaflet/images/marker-icon-2x.png',
  iconUrl: '/static/leaflet/images/marker-icon.png',
  shadowUrl: '/static/leaflet/images/marker-shadow.png',
});

let _alarmIntervalId = null;
let _alarmAudioCtx = null;
let _alarmIsActive = false;
let _alarmNeedsUserGesture = false;
let _lastCriticalHandledTs = 0;

function dismissAllActiveNotifications() {
  const panel = document.getElementById('alertPanel');
  if (!panel) return;
  const items = panel.querySelectorAll('.alert-toast');
  let any = false;
  items.forEach((el) => {
    try {
      any = true;
      el.classList.add('dismissed');
      setTimeout(() => el.remove(), 350);
    } catch (_e) {
      // ignore
    }
  });

  if (any) {
    setTimeout(() => {
      try {
        panel.querySelectorAll('.alert-toast.dismissed').forEach((el) => el.remove());
      } catch (_e) {
        // ignore
      }
    }, 500);
  }
}

function _ensureAlarmAudioCtx() {
  if (_alarmAudioCtx) return _alarmAudioCtx;
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return null;
  _alarmAudioCtx = new Ctx();
  return _alarmAudioCtx;
}

function _beepOnce() {
  const ctx = _ensureAlarmAudioCtx();
  if (!ctx) return;

  if (ctx.state === 'suspended') {
    ctx.resume().catch(() => {
      _alarmNeedsUserGesture = true;
    });
  }

  const osc = ctx.createOscillator();
  const gain = ctx.createGain();

  osc.type = 'sine';
  osc.frequency.value = 880;
  gain.gain.value = 0.0001;

  osc.connect(gain);
  gain.connect(ctx.destination);

  const now = ctx.currentTime;
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(0.35, now + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.22);

  osc.start(now);
  osc.stop(now + 0.24);
}

function startCriticalAlarm() {
  if (_alarmIsActive) return;
  _alarmIsActive = true;
  _alarmNeedsUserGesture = false;

  const unlock = () => {
    const ctx = _ensureAlarmAudioCtx();
    if (!ctx) return;
    ctx.resume().finally(() => {
      _alarmNeedsUserGesture = false;
      _beepOnce();
    });
  };
  document.addEventListener('pointerdown', unlock, { capture: true, once: true });

  _beepOnce();
  _alarmIntervalId = setInterval(_beepOnce, 750);
}

function stopCriticalAlarm() {
  _alarmIsActive = false;
  _alarmNeedsUserGesture = false;

  window.__toastSuppressCriticalUntil = Number(_lastCriticalHandledTs) || 0;
  dismissAllActiveNotifications();

  if (_alarmIntervalId) {
    clearInterval(_alarmIntervalId);
    _alarmIntervalId = null;
  }

  if (_alarmAudioCtx) {
    try {
      _alarmAudioCtx.close();
    } catch (_e) {
      // ignore
    }
    _alarmAudioCtx = null;
  }
}

function addStopAlertControl() {
  if (!map || !L || !L.Control) return;
  if (map._stopAlertControlAdded) return;
  map._stopAlertControlAdded = true;

  const StopControl = L.Control.extend({
    options: { position: 'topleft' },
    onAdd: function () {
      const btn = L.DomUtil.create('button', 'stop-alert-btn');
      btn.type = 'button';
      btn.textContent = 'Stop Alert';
      btn.title = 'Stop critical alert sound';
      L.DomEvent.disableClickPropagation(btn);
      L.DomEvent.on(btn, 'click', (e) => {
        L.DomEvent.preventDefault(e);
        stopCriticalAlarm();
      });
      return btn;
    }
  });

  map.addControl(new StopControl());
}

const grid = document.getElementById('grid');
const empty = document.getElementById('empty');
const countEl = document.getElementById('count');
const criticalCountEl = document.getElementById('criticalCount');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const refreshBtn = document.getElementById('refreshBtn');

let lastCount = 0;

let map;
let officerMarker;
let officerAccuracyCircle;
let hasCenteredOnOfficer = false;
const zoneLayersById = new Map();
const regionLayersByCameraId = new Map();
const crowdAreaLayersByKey = new Map();

function clsForLevel(lvl) {
  const v = String(lvl || '').toUpperCase();
  if (v === 'CRITICAL') return { card: 'critical', text: 'critical' };
  if (v === 'WARNING') return { card: 'warning', text: 'warning' };
  return { card: 'low', text: 'low' };
}

function fmtTs(ts) {
  if (ts === null || ts === undefined) return '-';
  const n = Number(ts);
  if (!Number.isFinite(n)) return String(ts);
  return new Date(n * 1000).toLocaleString();
}

function initMap() {
  map = L.map('map', {
    zoomControl: true,
  }).setView([20.5937, 78.9629], 5);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);

  addStopAlertControl();
}

function setOfficerLocation(lat, lon, accuracyMeters) {
  if (!map) return;
  const pos = [lat, lon];

  if (!officerMarker) {
    officerMarker = L.marker(pos, { title: 'Officer location' }).addTo(map);
  } else {
    officerMarker.setLatLng(pos);
  }

  if (Number.isFinite(accuracyMeters) && accuracyMeters > 0) {
    if (!officerAccuracyCircle) {
      officerAccuracyCircle = L.circle(pos, {
        radius: accuracyMeters,
        color: '#00d4ff',
        weight: 1,
        fillColor: '#00d4ff',
        fillOpacity: 0.15,
      }).addTo(map);
    } else {
      officerAccuracyCircle.setLatLng(pos);
      officerAccuracyCircle.setRadius(accuracyMeters);
    }
  }

  if (!hasCenteredOnOfficer) {
    map.setView(pos, 16);
    hasCenteredOnOfficer = true;
  }
}

function startGeolocation() {
  if (!('geolocation' in navigator)) {
    statusDot.className = 'dot warn';
    statusText.textContent = 'GPS not supported';
    return;
  }

  navigator.geolocation.watchPosition(
    (p) => {
      const lat = p.coords.latitude;
      const lon = p.coords.longitude;
      const acc = p.coords.accuracy;
      setOfficerLocation(lat, lon, acc);
    },
    (_err) => {
      statusDot.className = 'dot warn';
      statusText.textContent = 'GPS permission denied';
    },
    {
      enableHighAccuracy: true,
      maximumAge: 2000,
      timeout: 10000,
    }
  );
}

function zoneIdFromAlert(a) {
  const lat = a.latitude ?? a.lat;
  const lon = a.longitude ?? a.lon;
  return `${a.timestamp ?? ''}_${lat ?? ''}_${lon ?? ''}`;
}

function upsertCriticalZones(zones) {
  if (!map) return;
  const seen = new Set();

  for (const z of zones) {
    const id = zoneIdFromAlert(z);
    seen.add(id);

    const lat = Number(z.latitude);
    const lon = Number(z.longitude);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;

    const density = Number(z.crowd_density ?? z.density ?? 0);
    const radius = Math.max(40, Math.min(160, 40 + density * 12));

    const html = `
      <div style="font-family: system-ui; font-size: 12px;">
        <div style="font-weight:700; color:#ff2d55;">CRITICAL ALERT AREA</div>
        <div><b>Density</b>: ${Number.isFinite(density) && density > 0 ? density.toFixed(2) : '-'} p/m²</div>
        <div><b>Location</b>: ${lat.toFixed(5)}, ${lon.toFixed(5)}</div>
        <div><b>Timestamp</b>: ${fmtTs(z.timestamp)}</div>
      </div>
    `;

    if (!zoneLayersById.has(id)) {
      const circle = L.circle([lat, lon], {
        radius,
        color: '#ff2d55',
        weight: 2,
        fillColor: '#ff2d55',
        fillOpacity: 0.35,
      });
      const marker = L.circleMarker([lat, lon], {
        radius: 7,
        color: '#ff2d55',
        weight: 2,
        fillColor: '#ff2d55',
        fillOpacity: 0.95,
      });

      circle.bindPopup(html);
      marker.bindPopup(html);

      const group = L.layerGroup([circle, marker]).addTo(map);
      zoneLayersById.set(id, { group, circle, marker });
    } else {
      const obj = zoneLayersById.get(id);
      obj.circle.setLatLng([lat, lon]);
      obj.marker.setLatLng([lat, lon]);
      obj.circle.setRadius(radius);
      obj.circle.bindPopup(html);
      obj.marker.bindPopup(html);
    }
  }

  for (const [id, layer] of zoneLayersById.entries()) {
    if (!seen.has(id)) {
      layer.group.remove();
      zoneLayersById.delete(id);
    }
  }
}

function riskColor(level) {
  const v = String(level || '').toUpperCase();
  if (v === 'CRITICAL') return '#ff2d55';
  if (v === 'WARNING') return '#f5a623';
  return '#00e676';
}

function upsertRegions(regions) {
  if (!map) return;
  const seen = new Set();

  for (const r of regions) {
    const cameraId = String(r.camera_id ?? r.cameraId ?? r.id ?? 'CAM');
    seen.add(cameraId);

    const lat = Number(r.latitude);
    const lon = Number(r.longitude);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;

    const color = riskColor(r.risk_level);
    const radius = 55;

    const html = `
      <div style="font-family: system-ui; font-size: 12px;">
        <div style="font-weight:700;">${cameraId} · Region Status</div>
        <div><b>Risk</b>: <span style="color:${color}; font-weight:700;">${r.risk_level ?? '-'}</span></div>
        <div><b>Density</b>: ${r.crowd_density ?? '-'} p/m²</div>
        <div><b>Timestamp</b>: ${fmtTs(r.timestamp)}</div>
      </div>
    `;

    if (!regionLayersByCameraId.has(cameraId)) {
      const circle = L.circle([lat, lon], {
        radius,
        color,
        weight: 2,
        fillColor: color,
        fillOpacity: 0.18,
      });
      const marker = L.circleMarker([lat, lon], {
        radius: 7,
        color,
        weight: 2,
        fillColor: color,
        fillOpacity: 0.9,
      });
      circle.bindPopup(html);
      marker.bindPopup(html);
      const group = L.layerGroup([circle, marker]).addTo(map);
      regionLayersByCameraId.set(cameraId, { group, circle, marker });
    } else {
      const obj = regionLayersByCameraId.get(cameraId);
      obj.circle.setLatLng([lat, lon]);
      obj.marker.setLatLng([lat, lon]);
      obj.circle.setStyle({ color, fillColor: color });
      obj.marker.setStyle({ color, fillColor: color });
      obj.circle.bindPopup(html);
      obj.marker.bindPopup(html);
    }
  }

  for (const [cameraId, obj] of regionLayersByCameraId.entries()) {
    if (!seen.has(cameraId)) {
      obj.group.remove();
      regionLayersByCameraId.delete(cameraId);
    }
  }
}

function render(alerts) {
  countEl.textContent = String(alerts.length);
  criticalCountEl.textContent = String(alerts.filter(a => String(a.risk_level || '').toUpperCase() === 'CRITICAL').length);

  const latestCriticalTs = alerts
    .filter(a => String(a.risk_level || '').toUpperCase() === 'CRITICAL')
    .reduce((mx, a) => {
      const ts = Number(a.received_at ?? a.timestamp ?? 0);
      return Number.isFinite(ts) ? Math.max(mx, ts) : mx;
    }, 0);

  if (latestCriticalTs > _lastCriticalHandledTs) {
    _lastCriticalHandledTs = latestCriticalTs;
    startCriticalAlarm();
  }

  if (!alerts.length) {
    empty.style.display = 'block';
    grid.innerHTML = '';
  } else {
    empty.style.display = 'none';
    grid.innerHTML = alerts.slice().reverse().map(a => {
      const c = clsForLevel(a.risk_level);
      const toastTs = Number(a.received_at ?? a.timestamp ?? 0);
      return `
        <div class="card ${c.card}" data-toast-ts="${Number.isFinite(toastTs) ? toastTs : 0}">
          <div class="top">
            <div class="loc">${a.location ?? '-'}</div>
            <div class="lvl ${c.text}">${a.risk_level ?? '-'}</div>
          </div>
          <div class="kv">
            <div>
              <div class="k">Crowd count</div>
              <div class="v">${a.crowd_count ?? '-'}</div>
            </div>
            <div>
              <div class="k">Density (p/m²)</div>
              <div class="v">${a.density ?? '-'}</div>
            </div>
            <div>
              <div class="k">Event timestamp</div>
              <div class="v">${fmtTs(a.timestamp)}</div>
            </div>
            <div>
              <div class="k">Received at</div>
              <div class="v">${fmtTs(a.received_at)}</div>
            </div>
          </div>
        </div>
      `;
    }).join('');
  }

  const newCount = alerts.length;
  if (newCount > lastCount) {
    statusDot.className = 'dot crit';
    statusText.textContent = 'New alert received';
    setTimeout(() => {
      statusDot.className = 'dot';
      statusText.textContent = 'Live';
    }, 1200);
  } else {
    statusDot.className = 'dot';
    statusText.textContent = 'Live';
  }
  lastCount = newCount;
}

async function poll() {
  try {
    const res = await fetch('/api/alerts', { cache: 'no-store' });
    const data = await res.json();
    if (!data || !data.ok) return;
    render(Array.isArray(data.alerts) ? data.alerts : []);
  } catch (e) {
    statusDot.className = 'dot warn';
    statusText.textContent = 'Disconnected';
  }
}

async function pollZones() {
  try {
    const res = await fetch('/api/zones', { cache: 'no-store' });
    const data = await res.json();
    if (!data || !data.ok) return;
    const zones = Array.isArray(data.zones) ? data.zones : [];
    upsertCriticalZones(zones);
  } catch (_e) {
    // ignore
  }
}

async function pollRegions() {
  try {
    const res = await fetch('/api/regions', { cache: 'no-store' });
    const data = await res.json();
    if (!data || !data.ok) return;
    const regions = Array.isArray(data.regions) ? data.regions : [];
    upsertRegions(regions);
  } catch (_e) {
    // ignore
  }
}

function upsertCrowdAreas(items) {
  if (!map) return;
  const seen = new Set();

  for (const it of items) {
    const lat = Number(it.lat);
    const lng = Number(it.lng);
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) continue;

    const key = `${lat.toFixed(6)}_${lng.toFixed(6)}`;
    seen.add(key);

    if (!crowdAreaLayersByKey.has(key)) {
      const circle = L.circle([lat, lng], {
        radius: 90,
        color: '#ff2d55',
        weight: 2,
        fillColor: '#ff2d55',
        fillOpacity: 0.22,
      }).addTo(map);
      crowdAreaLayersByKey.set(key, circle);
    } else {
      const circle = crowdAreaLayersByKey.get(key);
      circle.setLatLng([lat, lng]);
    }
  }

  for (const [key, circle] of crowdAreaLayersByKey.entries()) {
    if (!seen.has(key)) {
      circle.remove();
      crowdAreaLayersByKey.delete(key);
    }
  }
}

async function pollCrowdZones() {
  try {
    const res = await fetch('/api/crowd-zones', { cache: 'no-store' });
    const data = await res.json();
    if (!Array.isArray(data)) return;
    upsertCrowdAreas(data);
  } catch (_e) {
    // ignore
  }
}

refreshBtn.addEventListener('click', poll);
initMap();
startGeolocation();
poll();
pollZones();
pollRegions();
pollCrowdZones();
setInterval(poll, 1000);
setInterval(pollZones, 1000);
setInterval(pollRegions, 1000);
setInterval(pollCrowdZones, 1000);
