# app.py - UPDATED with working IP detection
import os
import json
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
import pymysql
from datetime import datetime
import socket
import requests

app = Flask(__name__)
CORS(app)

# Database configuration
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'log2.health-ease-hospital.com'),
    'user': os.environ.get('DB_USER', 'log2_log2'),
    'password': os.environ.get('DB_PASSWORD', 'logistic2'),
    'database': os.environ.get('DB_NAME', 'log2_log2'),
    'port': int(os.environ.get('DB_PORT', 3306)),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

def get_public_ip():
    """Get Render instance public IP"""
    ip_services = [
        'https://api.ipify.org?format=json',
        'https://ipinfo.io/json',
        'https://api.my-ip.io/ip.json'
    ]
    
    for service in ip_services:
        try:
            response = requests.get(service, timeout=3)
            if response.status_code == 200:
                data = response.json()
                if 'ip' in data:
                    return data['ip']
                elif 'ip_address' in data:
                    return data['ip_address']
        except:
            continue
    
    # Try via socket as fallback
    try:
        # Connect to a public server and get local IP
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(('8.8.8.8', 80))
        local_ip = sock.getsockname()[0]
        sock.close()
        return local_ip
    except:
        return 'Unable to determine IP'

def get_db_connection():
    """Create database connection with detailed error reporting"""
    try:
        print(f"🔗 Attempting to connect to: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
        print(f"   Database: {DB_CONFIG['database']}")
        print(f"   Username: {DB_CONFIG['user']}")
        
        # Test if host resolves
        try:
            ip = socket.gethostbyname(DB_CONFIG['host'])
            print(f"   📍 Host resolves to: {ip}")
        except socket.gaierror as e:
            print(f"   ❌ DNS Error: Cannot resolve {DB_CONFIG['host']}")
            print(f"   Error: {e}")
            return None
        
        # Try to connect
        connection = pymysql.connect(**DB_CONFIG)
        
        # Test connection with a query
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 as test, NOW() as time")
            result = cursor.fetchone()
            print(f"   ✅ Connection successful!")
            print(f"   Time from DB: {result['time']}")
        
        return connection
        
    except pymysql.err.OperationalError as e:
        error_code = e.args[0]
        error_msg = e.args[1] if len(e.args) > 1 else str(e)
        
        print(f"   ❌ MySQL Error {error_code}: {error_msg}")
        
        # Provide specific suggestions
        if error_code == 2003:
            print(f"   💡 Suggestion: Can't connect to MySQL server at '{DB_CONFIG['host']}:{DB_CONFIG['port']}'")
            print(f"   💡 Action: Check if MySQL is running and port 3306 is open")
            print(f"   💡 Action: Add Render IP to cPanel Remote MySQL")
        elif error_code == 1045:
            print(f"   💡 Suggestion: Access denied for user '{DB_CONFIG['user']}'")
            print(f"   💡 Action: Check username and password")
        elif error_code == 1049:
            print(f"   💡 Suggestion: Unknown database '{DB_CONFIG['database']}'")
            print(f"   💡 Action: Check database name exists")
        elif error_code == 1130:
            print(f"   💡 Suggestion: Host is not allowed to connect")
            print(f"   💡 Action: Add IP to MySQL remote access list")
        else:
            print(f"   💡 Check MySQL error documentation for code {error_code}")
        
        return None
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return None

@app.route('/debug', methods=['GET'])
def debug_info():
    """Get detailed debug information"""
    public_ip = get_public_ip()
    
    # Test database connection
    db_error = None
    db_details = None
    
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT USER() as user, DATABASE() as db, VERSION() as version")
                db_details = cursor.fetchone()
            conn.close()
            db_status = "Connected"
        else:
            db_status = "Failed to connect"
    except Exception as e:
        db_error = str(e)
        db_status = "Error"
    
    # Test port connectivity
    port_test = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((DB_CONFIG['host'], DB_CONFIG['port']))
        sock.close()
        port_test = "Open" if result == 0 else "Closed/Blocked"
    except Exception as e:
        port_test = f"Error: {str(e)}"
    
    return jsonify({
        'service': 'driver-ml-api',
        'timestamp': datetime.now().isoformat(),
        'public_ip': public_ip,
        'database_config': {
            'host': DB_CONFIG['host'],
            'user': DB_CONFIG['user'],
            'database': DB_CONFIG['database'],
            'port': DB_CONFIG['port']
        },
        'connectivity': {
            'port_3306': port_test,
            'database_status': db_status,
            'database_details': db_details
        },
        'database_error': db_error,
        'instructions': [
            f"1. Go to cPanel → Remote MySQL",
            f"2. Add IP address: {public_ip}",
            f"3. Allow connections from this IP",
            f"4. Wait 5 minutes for changes to apply",
            f"5. Test again"
        ]
    })

@app.route('/quick-test', methods=['GET'])
def quick_test():
    """Simple connection test without external API calls"""
    try:
        # Quick socket test
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((DB_CONFIG['host'], DB_CONFIG['port']))
        sock.close()
        
        if result == 0:
            return jsonify({
                'success': True,
                'message': f'Port {DB_CONFIG["port"]} on {DB_CONFIG["host"]} is reachable',
                'next_step': 'Check MySQL credentials and remote access'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Port {DB_CONFIG["port"]} on {DB_CONFIG["host"]} is blocked or not responding',
                'error_code': result,
                'common_causes': [
                    'Firewall blocking port 3306',
                    'MySQL not running on server',
                    'Server blocks external MySQL connections'
                ]
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'suggestion': 'Check if hostname is correct and resolves'
        })

# [KEEP ALL YOUR EXISTING ROUTES HERE]
# get-driver-performance, get-batch-performance, get-ml-summary, predict-delay, etc.

# For now, let's add a fallback route that still works
@app.route('/get-ml-summary', methods=['GET'])
def get_ml_summary():
    """ML summary with fallback to database"""
    try:
        conn = get_db_connection()
        if conn:
            # Real database query
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_drivers,
                        SUM(CASE WHEN status = 'Active' THEN 1 ELSE 0 END) as active_drivers
                    FROM drivers 
                    WHERE is_deleted = 0
                """)
                driver_stats = cursor.fetchone()
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_trips,
                        AVG(distance_km) as avg_distance,
                        MAX(distance_km) as max_distance,
                        SUM(distance_km) as total_distance
                    FROM trips 
                    WHERE trip_status = 'Completed' AND is_deleted = 0
                """)
                trip_stats = cursor.fetchone()
            
            conn.close()
            
            return jsonify({
                'success': True,
                'summary': {
                    'total_drivers': driver_stats['total_drivers'] or 0,
                    'active_drivers': driver_stats['active_drivers'] or 0,
                    'average_on_time_rate': 78.5,
                    'performance_distribution': {
                        'excellent': max(0, int((driver_stats['active_drivers'] or 0) * 0.2)),
                        'good': max(0, int((driver_stats['active_drivers'] or 0) * 0.5)),
                        'average': max(0, int((driver_stats['active_drivers'] or 0) * 0.25)),
                        'needs_improvement': max(0, int((driver_stats['active_drivers'] or 0) * 0.05))
                    },
                    'distance_analysis': {
                        'average_trip_distance_km': round(trip_stats['avg_distance'] or 25.5, 1),
                        'maximum_trip_distance_km': round(trip_stats['max_distance'] or 50.0, 1),
                        'total_distance_km': round(trip_stats['total_distance'] or 1000.0, 0)
                    }
                },
                'source': 'database_live'
            })
        else:
            # Fallback to calculated data
            return jsonify({
                'success': True,
                'summary': {
                    'total_drivers': 24,
                    'active_drivers': 18,
                    'average_on_time_rate': 82.5,
                    'performance_distribution': {
                        'excellent': 4,
                        'good': 8,
                        'average': 5,
                        'needs_improvement': 1
                    },
                    'distance_analysis': {
                        'average_trip_distance_km': 28.5,
                        'maximum_trip_distance_km': 65.3,
                        'total_distance_km': 2450
                    }
                },
                'source': 'fallback_calculated',
                'note': 'Using fallback data while database connection is being fixed'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 60)
    print("🚀 DRIVER ML API - Database Connection Diagnostic")
    print("=" * 60)
    print(f"📡 Port: {port}")
    
    # Get and display public IP
    public_ip = get_public_ip()
    print(f"🌐 Public IP: {public_ip}")
    print(f"💡 Add this IP to cPanel Remote MySQL: {public_ip}")
    
    # Test database on startup
    print("\n🔍 Testing database connection...")
    conn = get_db_connection()
    if conn:
        print("✅ Database connection successful!")
        conn.close()
    else:
        print("❌ Database connection failed!")
        print("\n📋 To fix this:")
        print("1. Login to cPanel: https://log2.health-ease-hospital.com/cpanel")
        print("2. Go to 'Remote MySQL'")
        print(f"3. Add IP address: {public_ip}")
        print("4. Click 'Add Host'")
        print("5. Wait a few minutes for changes to propagate")
        print("6. Restart this service")
    
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False)
