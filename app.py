import os

from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from flask_cors import CORS
from camera_p2pnet import VideoCamera as VideoCameraP2PNet
from inferenceP2PNet import get_prediction as get_prediction_p2pnet


from threading import Thread, Event
from collections import deque
import time

import gps_state


from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './static/'
ALLOWED_EXTENSIONS = set(['.jpg', '.jpeg','mp4'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app)

global event 
event = Event()


method = ''


_p2pnet_camera = None

_metrics_history = deque(maxlen=600) 


def _get_p2pnet_camera(fileName: str):
    global _p2pnet_camera
    if _p2pnet_camera is None:
        _p2pnet_camera = VideoCameraP2PNet(fileName or '')
    return _p2pnet_camera


def _record_metrics_from_camera(camera):
    try:
        m = getattr(camera, 'latest_metrics', None)
        if isinstance(m, dict) and m:
            _metrics_history.append(m)
    except Exception:
        pass

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        method = request.form.get('method')
        
        if request.form.get('upload-file') == 'upload-file':
            # Remove existing images
            files_in_dir = os.listdir(app.config['UPLOAD_FOLDER'])
            filtered_files = [file for file in files_in_dir if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".mp4")]
            for file in filtered_files:
                path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                os.remove(path)

            # Upload new file
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files['file']
            if not file:
                return redirect(request.url)
            
            filename = secure_filename(file.filename)
            file_name1 = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_name1)
            print("GETTING PREDICTION")

            if file_name1.endswith(".mp4"):
                return render_template('result-video.html', File=file_name1, Method=method)
                
            if method == 'p2pnet':
                prediction, density = get_prediction_p2pnet(file_name1)
            else:
                return "Method not supported", 400
            
            if file_name1.endswith(".jpg") or file_name1.endswith(".jpeg"):
                return jsonify({
                    'prediction': prediction,
                    'file': filename,
                    'density': density,
                    'method': method
                })
                
        elif request.form.get('use-webcam') == 'use-webcam':
            return render_template('result-video.html', File='', Method=method)
          
    return render_template('dashboard.html')


@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html')

    
 
    


def gen(camera):
    while True:
        frame = camera.get_frame()
        _record_metrics_from_camera(camera)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen_p2pnet(panel):
    camera = _get_p2pnet_camera('')
    while True:
        frame = camera.get_frame(panel=panel)
        _record_metrics_from_camera(camera)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')






@app.route("/video_feed", methods = ['GET', 'POST'])
def video_feed():
    method = request.args.get('method', 'p2pnet')
    panel = request.args.get('panel', 'grid')
    if method == 'p2pnet':
        return Response(gen_p2pnet(panel), mimetype='multipart/x-mixed-replace; boundary=frame')

    else:
        return "Method not supported", 400


@app.route('/api/status', methods=['GET'])
def api_status():
    """Latest metrics for dashboard."""
    cam = _get_p2pnet_camera('')
    m = getattr(cam, 'latest_metrics', None)
    if not isinstance(m, dict) or not m:
        return jsonify({
            'ok': False,
            'error': 'no_metrics_yet'
        })
    return jsonify({
        'ok': True,
        'metrics': m
    })


@app.route('/api/history', methods=['GET'])
def api_history():
    """Recent metrics time-series for charts."""
    limit = request.args.get('limit', default=300, type=int)
    limit = max(1, min(int(limit), 600))
    items = list(_metrics_history)[-limit:]
    return jsonify({
        'ok': True,
        'count': len(items),
        'items': items
    })


@app.route('/api/gps', methods=['POST'])
def api_gps_post():
    data = request.get_json(silent=True) or {}
    lat = data.get('lat')
    lon = data.get('lon')
    accuracy = data.get('accuracy')
    ts = data.get('ts')

    if lat is None or lon is None:
        return jsonify({
            'ok': False,
            'error': 'missing_lat_lon',
        }), 400

    gps_state.set_latest(lat=lat, lon=lon, accuracy=accuracy, ts=ts)
    return jsonify({
        'ok': True,
    })


@app.route('/api/gps', methods=['GET'])
def api_gps_get():
    return jsonify({
        'ok': True,
        'gps': gps_state.get_latest(),
    })
  
if __name__ == '__main__':
    #app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    app.run(threaded=True,host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
