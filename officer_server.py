from __future__ import annotations

import time
from typing import Any, Dict, List

from flask import Flask, jsonify, request, render_template_string


app = Flask(__name__)


_alerts: List[Dict[str, Any]] = []
_regions_by_camera: Dict[str, Dict[str, Any]] = {}


def _now_ts() -> float:
    return time.time()


@app.get('/')
def index():
    return render_template_string(
        """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Officer Map Dashboard</title>
    <link
      rel=\"stylesheet\"
      href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\"
      integrity=\"sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=\"
      crossorigin=\"\"
    />
    <style>
      :root {
        --bg: #0b1220;
        --card: #101a2e;
        --border: #223055;
        --text: #e7eefc;
        --muted: #8aa0c7;
        --crit: #ff2d55;
        --warn: #f5a623;
        --low: #00e676;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        background: radial-gradient(1200px 800px at 10% 0%, #122042 0%, var(--bg) 60%);
        color: var(--text);
        min-height: 100vh;
      }
      header {
        position: sticky;
        top: 0;
        backdrop-filter: blur(8px);
        background: rgba(11, 18, 32, 0.75);
        border-bottom: 1px solid var(--border);
        padding: 14px 18px;
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 12px;
      }
      h1 { margin: 0; font-size: 16px; letter-spacing: 0.5px; }
      .sub { color: var(--muted); font-size: 12px; }
      main { padding: 16px 18px 32px; max-width: 1150px; margin: 0 auto; }
      .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
      .pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 10px;
        border: 1px solid var(--border);
        background: rgba(16, 26, 46, 0.7);
        border-radius: 999px;
        font-size: 12px;
        color: var(--muted);
      }
      .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--low); }
      .dot.warn { background: var(--warn); }
      .dot.crit { background: var(--crit); }

      #map {
        width: 100%;
        height: 560px;
        border-radius: 12px;
        border: 1px solid var(--border);
        overflow: hidden;
        background: rgba(16, 26, 46, 0.6);
      }

      .grid {
        margin-top: 14px;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 12px;
      }
      .card {
        border: 1px solid var(--border);
        background: rgba(16, 26, 46, 0.8);
        border-radius: 10px;
        padding: 12px;
        overflow: hidden;
      }
      .card.critical { border-color: rgba(255, 45, 85, 0.5); box-shadow: 0 0 0 1px rgba(255, 45, 85, 0.15) inset; }
      .card.warning { border-color: rgba(245, 166, 35, 0.5); box-shadow: 0 0 0 1px rgba(245, 166, 35, 0.15) inset; }
      .top { display: flex; justify-content: space-between; gap: 10px; }
      .loc { font-weight: 650; }
      .lvl { font-weight: 700; }
      .lvl.low { color: var(--low); }
      .lvl.warning { color: var(--warn); }
      .lvl.critical { color: var(--crit); }
      .kv { margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
      .k { color: var(--muted); font-size: 11px; }
      .v { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
      .empty {
        margin-top: 18px;
        border: 1px dashed rgba(138, 160, 199, 0.35);
        border-radius: 10px;
        padding: 18px;
        color: var(--muted);
        text-align: center;
      }
      button {
        cursor: pointer;
        border: 1px solid var(--border);
        background: rgba(16, 26, 46, 0.7);
        color: var(--text);
        padding: 8px 10px;
        border-radius: 8px;
        font-size: 12px;
      }
      button:hover { border-color: rgba(0, 230, 118, 0.4); }

      .section-title {
        margin: 14px 0 10px;
        color: var(--muted);
        font-size: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
      }
    </style>
  </head>
  <body>
    <header>
      <div>
        <h1>Officer Map Dashboard</h1>
        <div class=\"sub\">Live GPS + CRITICAL alert zones from <code>POST /api/alert</code></div>
      </div>
      <div class=\"row\">
        <div class=\"pill\"><span class=\"dot\" id=\"statusDot\"></span><span id=\"statusText\">Waiting</span></div>
        <div class=\"pill\">Total alerts: <span id=\"count\">0</span></div>
        <div class=\"pill\">CRITICAL zones: <span id=\"criticalCount\">0</span></div>
        <button id=\"refreshBtn\" type=\"button\">Refresh</button>
      </div>
    </header>
    <main>
      <div class=\"section-title\">Map</div>
      <div id=\"map\"></div>

      <div class=\"section-title\">Recent Alerts</div>
      <div id=\"empty\" class=\"empty\" style=\"display:none\">No alerts received yet.</div>
      <div id=\"grid\" class=\"grid\"></div>
    </main>

    <script
      src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"
      integrity=\"sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=\"
      crossorigin=\"\"
    ></script>
    <script>
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

        if (!alerts.length) {
          empty.style.display = 'block';
          grid.innerHTML = '';
        } else {
          empty.style.display = 'none';
          grid.innerHTML = alerts.slice().reverse().map(a => {
            const c = clsForLevel(a.risk_level);
            return `
              <div class=\"card ${c.card}\">
                <div class=\"top\">
                  <div class=\"loc\">${a.location ?? '-'}</div>
                  <div class=\"lvl ${c.text}\">${a.risk_level ?? '-'}</div>
                </div>
                <div class=\"kv\">
                  <div>
                    <div class=\"k\">Crowd count</div>
                    <div class=\"v\">${a.crowd_count ?? '-'}</div>
                  </div>
                  <div>
                    <div class=\"k\">Density (p/m²)</div>
                    <div class=\"v\">${a.density ?? '-'}</div>
                  </div>
                  <div>
                    <div class=\"k\">Event timestamp</div>
                    <div class=\"v\">${fmtTs(a.timestamp)}</div>
                  </div>
                  <div>
                    <div class=\"k\">Received at</div>
                    <div class=\"v\">${fmtTs(a.received_at)}</div>
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
    </script>
  </body>
</html>"""
    )


@app.post('/api/alert')
def api_alert():
    data = request.get_json(silent=True) or {}

    location = data.get('location')
    risk_level = data.get('risk_level')
    crowd_count = data.get('crowd_count')
    timestamp = data.get('timestamp')

    latitude = data.get('latitude', data.get('lat'))
    longitude = data.get('longitude', data.get('lon'))
    crowd_density = data.get('crowd_density', data.get('density'))

    required = {
        'location': location,
        'risk_level': risk_level,
        'crowd_count': crowd_count,
        'crowd_density': crowd_density,
        'timestamp': timestamp,
        'latitude': latitude,
        'longitude': longitude,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        return jsonify({
            'ok': False,
            'error': 'missing_fields',
            'missing': missing,
        }), 400

    alert = {
        'received_at': _now_ts(),
        'location': location,
        'risk_level': risk_level,
        'crowd_count': crowd_count,
        'crowd_density': crowd_density,
        'timestamp': timestamp,
        'latitude': latitude,
        'longitude': longitude,
    }
    _alerts.append(alert)

    print("\n[OFFICER ALERT RECEIVED]")
    print(f" location   : {alert['location']}")
    print(f" risk_level : {alert['risk_level']}")
    print(f" crowd_count: {alert['crowd_count']}")
    print(f" crowd_density: {alert['crowd_density']}")
    print(f" latitude   : {alert['latitude']}")
    print(f" longitude  : {alert['longitude']}")
    print(f" timestamp  : {alert['timestamp']}")

    return jsonify({
        'ok': True,
        'count': len(_alerts),
    })


@app.get('/api/alerts')
def api_alerts():
    return jsonify({
        'ok': True,
        'count': len(_alerts),
        'alerts': _alerts,
    })


@app.get('/api/zones')
def api_zones():
    zones = [
        a for a in _alerts
        if str(a.get('risk_level', '')).upper() == 'CRITICAL'
    ]
    return jsonify({
        'ok': True,
        'count': len(zones),
        'zones': zones,
    })


@app.get('/api/crowd-zones')
def api_crowd_zones():
    zones = []
    for a in _alerts:
        if str(a.get('risk_level', '')).upper() != 'CRITICAL':
            continue
        lat = a.get('latitude')
        lng = a.get('longitude')
        if lat is None or lng is None:
            continue
        zones.append({
            'lat': lat,
            'lng': lng,
        })
    return jsonify(zones)


@app.post('/api/region')
def api_region():
    data = request.get_json(silent=True) or {}

    camera_id = data.get('camera_id')
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    crowd_density = data.get('crowd_density')
    risk_level = data.get('risk_level')
    timestamp = data.get('timestamp')

    required = {
        'camera_id': camera_id,
        'latitude': latitude,
        'longitude': longitude,
        'crowd_density': crowd_density,
        'risk_level': risk_level,
        'timestamp': timestamp,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        return jsonify({
            'ok': False,
            'error': 'missing_fields',
            'missing': missing,
        }), 400

    region = {
        'received_at': _now_ts(),
        'camera_id': camera_id,
        'latitude': latitude,
        'longitude': longitude,
        'crowd_density': crowd_density,
        'risk_level': risk_level,
        'timestamp': timestamp,
    }
    _regions_by_camera[str(camera_id)] = region

    return jsonify({
        'ok': True,
        'camera_id': str(camera_id),
    })


@app.get('/api/regions')
def api_regions():
    regions = list(_regions_by_camera.values())
    return jsonify({
        'ok': True,
        'count': len(regions),
        'regions': regions,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, threaded=True)
