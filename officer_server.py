from __future__ import annotations

import time
from typing import Any, Dict, List

from flask import Flask, jsonify, request, render_template


app = Flask(__name__)


_alerts: List[Dict[str, Any]] = []
_regions_by_camera: Dict[str, Dict[str, Any]] = {}


def _now_ts() -> float:
    return time.time()


@app.get('/')
def index():
    return render_template('newui.html')


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
