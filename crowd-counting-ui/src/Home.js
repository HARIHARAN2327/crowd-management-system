import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function Home() {
  const [monitoring, setMonitoring] = useState(false);
  const [metrics, setMetrics] = useState({
    count: '-', rho: '-', v: '-', risk: '-', drho: '-', fps: '-', risk_level: 'LOW'
  });
  const [history, setHistory] = useState([]);
  const API_BASE = 'http://192.168.0.102:8080';

  useEffect(() => {
    let interval;
    if (monitoring) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/api/status`);
          const data = await res.json();
          if (data.ok) {
            setMetrics({
              count: data.metrics.count,
              rho: data.metrics.rho?.toFixed(2) || '-',
              v: data.metrics.v?.toFixed(2) || '-',
              risk: data.metrics.risk?.toFixed(2) || '-',
              drho: data.metrics.drho_dt?.toFixed(2) || '-',
              fps: data.metrics.fps?.toFixed(1) || '-',
              risk_level: data.metrics.risk_level
            });
          }
        } catch (e) {
          console.error('Status fetch error:', e);
        }

        try {
          const res = await fetch(`${API_BASE}/api/history?limit=120`);
          const data = await res.json();
          if (data.ok) {
            setHistory(data.items.map(item => ({
              time: new Date(item.ts * 1000).toLocaleTimeString(),
              risk: item.risk || 0,
              rho: item.rho || 0
            })));
          }
        } catch (e) {
          console.error('History fetch error:', e);
        }
      }, 500);
    }
    return () => clearInterval(interval);
  }, [monitoring]);

  const startMonitoring = () => setMonitoring(true);
  const stopMonitoring = () => setMonitoring(false);

  const getBadgeClass = (level) => {
    if (level === 'CRITICAL') return 'badge-critical';
    if (level === 'WARNING') return 'badge-warning';
    return 'badge-low';
  };

  return (
    <div style={{ background: '#0b0f14', color: '#e6edf3', minHeight: '100vh', padding: '20px' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        <div style={{ marginBottom: '20px' }}>
          <h1>Crowd Management Dashboard</h1>
          <p>Live P2PNet stream + analytics</p>
        </div>

        <div style={{ display: 'flex', gap: '20px', marginBottom: '20px' }}>
          <button onClick={startMonitoring} disabled={monitoring} style={{ padding: '10px 20px', fontSize: '18px', background: monitoring ? '#ccc' : '#007bff', color: 'white', border: 'none', borderRadius: '5px' }}>
            Start Monitoring
          </button>
          <button onClick={stopMonitoring} disabled={!monitoring} style={{ padding: '10px 20px', fontSize: '18px', background: !monitoring ? '#ccc' : '#dc3545', color: 'white', border: 'none', borderRadius: '5px' }}>
            Stop Monitoring
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px' }}>
          <div>
            <div style={{ background: '#111827', padding: '20px', borderRadius: '5px', border: '1px solid #1f2937' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <h3>Live Stream</h3>
                <span className={`badge ${getBadgeClass(metrics.risk_level)}`} style={{ padding: '5px 10px', borderRadius: '3px', color: 'white' }}>
                  {metrics.risk_level}
                </span>
              </div>
              {monitoring && <img src={`${API_BASE}/video_feed?method=p2pnet`} alt="Live Stream" style={{ width: '100%', border: '1px solid #1f2937' }} />}
            </div>
          </div>

          <div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '20px' }}>
              <div style={{ background: '#111827', padding: '15px', borderRadius: '5px', border: '1px solid #1f2937', textAlign: 'center' }}>
                <div style={{ color: '#9ca3af' }}>Count</div>
                <div style={{ fontSize: '28px', fontWeight: 'bold' }}>{metrics.count}</div>
              </div>
              <div style={{ background: '#111827', padding: '15px', borderRadius: '5px', border: '1px solid #1f2937', textAlign: 'center' }}>
                <div style={{ color: '#9ca3af' }}>Density (p/m²)</div>
                <div style={{ fontSize: '28px', fontWeight: 'bold' }}>{metrics.rho}</div>
              </div>
              <div style={{ background: '#111827', padding: '15px', borderRadius: '5px', border: '1px solid #1f2937', textAlign: 'center' }}>
                <div style={{ color: '#9ca3af' }}>Velocity (m/s)</div>
                <div style={{ fontSize: '28px', fontWeight: 'bold' }}>{metrics.v}</div>
              </div>
              <div style={{ background: '#111827', padding: '15px', borderRadius: '5px', border: '1px solid #1f2937', textAlign: 'center' }}>
                <div style={{ color: '#9ca3af' }}>Risk (0-1)</div>
                <div style={{ fontSize: '28px', fontWeight: 'bold' }}>{metrics.risk}</div>
              </div>
              <div style={{ background: '#111827', padding: '15px', borderRadius: '5px', border: '1px solid #1f2937', textAlign: 'center' }}>
                <div style={{ color: '#9ca3af' }}>dρ/dt</div>
                <div style={{ fontSize: '28px', fontWeight: 'bold' }}>{metrics.drho}</div>
              </div>
              <div style={{ background: '#111827', padding: '15px', borderRadius: '5px', border: '1px solid #1f2937', textAlign: 'center' }}>
                <div style={{ color: '#9ca3af' }}>FPS</div>
                <div style={{ fontSize: '28px', fontWeight: 'bold' }}>{metrics.fps}</div>
              </div>
            </div>

            <div style={{ background: '#111827', padding: '20px', borderRadius: '5px', border: '1px solid #1f2937' }}>
              <h3>History (Risk + Density)</h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="rho" stroke="#38bdf8" name="Density" />
                  <Line type="monotone" dataKey="risk" stroke="#f97316" name="Risk" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;