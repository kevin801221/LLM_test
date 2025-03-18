from flask import Blueprint, request, jsonify
import json
import logging
import os
from datetime import datetime
from queue import Queue

common_bp = Blueprint('common', __name__)

# 設置日誌
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 設置 API 請求日誌
api_logger = logging.getLogger('api_logger')
api_logger.setLevel(logging.INFO)

# 檔案處理器 - 所有 API 請求
all_requests_handler = logging.FileHandler(os.path.join(log_dir, 'all_api_requests.log'))
all_requests_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
api_logger.addHandler(all_requests_handler)

# 檔案處理器 - 只記錄設備註冊
device_register_handler = logging.FileHandler(os.path.join(log_dir, 'device_register.log'))
device_register_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
api_logger.addHandler(device_register_handler)

# 控制台處理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
api_logger.addHandler(console_handler)

# 用於儲存設備 IP 的字典
device_sessions = {}

# 用於處理 RTSP 註冊的事件隊列
rtsp_event_queue = Queue()

@common_bp.route('/api/register-device', methods=['POST'])
def register_device():
    try:
        data = request.get_json()
        
        # 記錄請求詳情
        api_logger.info("\n=== 接收到設備註冊請求 ===")
        api_logger.info(f"請求時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        api_logger.info(f"請求 IP: {request.remote_addr}")
        api_logger.info(f"Headers: {json.dumps(dict(request.headers), indent=2, ensure_ascii=False)}")
        api_logger.info(f"Data: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        device_id = data.get('device_id')
        ip_address = data.get('ip_address')
        rtsp_port = data.get('rtsp_port', '5050')
        
        if not device_id or not ip_address:
            error_msg = '缺少必要參數 device_id 或 ip_address'
            api_logger.error(f"註冊失敗: {error_msg}")
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 400
            
        # 儲存設備 IP
        device_sessions[device_id] = {
            'ip_address': ip_address,
            'rtsp_port': rtsp_port
        }
        
        # 生成 RTSP URL
        rtsp_url = f"rtsp://{ip_address}:{rtsp_port}"
        api_logger.info(f"生成的 RTSP URL: {rtsp_url}")
        
        # 將設備資訊放入事件隊列
        rtsp_event = {
            'type': 'add',
            'device_id': device_id,
            'rtsp_url': rtsp_url
        }
        rtsp_event_queue.put(rtsp_event)
        api_logger.info(f"事件已加入隊列: {rtsp_event}")
        
        success_response = {
            'status': 'success',
            'message': '設備註冊成功',
            'data': {
                'device_id': device_id,
                'ip_address': ip_address,
                'rtsp_url': rtsp_url
            }
        }
        api_logger.info(f"註冊成功: {json.dumps(success_response, indent=2, ensure_ascii=False)}")
        api_logger.info("=== 設備註冊請求處理完成 ===\n")
        
        return jsonify(success_response)
        
    except Exception as e:
        error_msg = f"註冊設備時發生錯誤: {str(e)}"
        api_logger.error(error_msg)
        api_logger.exception("詳細錯誤信息:")
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 500

@common_bp.route('/api/get-device-info/<device_id>', methods=['GET'])
def get_device_info(device_id):
    device_info = device_sessions.get(device_id)
    if device_info:
        return jsonify({
            'status': 'success',
            'data': device_info
        })
    return jsonify({
        'status': 'error',
        'message': '找不到設備資訊'
    }), 404

def get_device_rtsp_url(device_id):
    """
    獲取設備的 RTSP URL
    """
    device_info = device_sessions.get(device_id)
    if device_info:
        return f"rtsp://{device_info['ip_address']}:{device_info['rtsp_port']}"
    return None

# 清除設備 session
@common_bp.route('/api/clear-device/<device_id>', methods=['DELETE'])
def clear_device(device_id):
    if device_id in device_sessions:
        del device_sessions[device_id]
        return jsonify({
            'status': 'success',
            'message': '設備資訊已清除'
        })
    return jsonify({
        'status': 'error',
        'message': '找不到設備資訊'
    }), 404
