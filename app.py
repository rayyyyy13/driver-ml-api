# app.py - DEBUG VERSION
import os
import json
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
import pymysql
from datetime import datetime
import socket

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

def get_db_connection():
    """Create database connection with debug info"""
    try:
        # Test DNS resolution first
        print(f"🔍 Testing DNS resolution for {DB_CONFIG['host']}...")
        try:
            ip_address = socket.gethostbyname(DB_CONFIG['host'])
            print(f"   ✓ Resolved to IP: {ip_address}")
        except socket.gaierror:
            print(f"   ✗ Cannot resolve hostname: {DB_CONFIG['host']}")
            return None
        
        # Try to connect
        print(f"🔗 Attempting MySQL connection to {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
        print(f"   Username: {DB_CONFIG['user']}")
        print(f"   Database: {DB_CONFIG['database']}")
        
        connection = pymysql.connect(**DB_CONFIG)
        print(f"✅ Successfully connected to database!")
        
        # Test a simple query
        with connection.cursor() as cursor:
            cursor.execute("SELECT VERSION() as version")
            result = cursor.fetchone()
            print(f"   MySQL Version: {result['version']}")
        
        return connection
    except pymysql.err.OperationalError as e:
        print(f"❌ MySQL Operational Error: {e}")
        print(f"   Error code: {e.args[0]}")
        print(f"   Error message: {e.args[1]}")
        
        # Common error codes:
        # 2003: Can't connect to MySQL server
        # 1045: Access denied
        # 1049: Unknown database
        # 1130: Host not allowed to connect
        
        if e.args[0] == 2003:
            print("💡 Suggestion: MySQL server is not reachable. Check if port 3306 is open.")
        elif e.args[0] == 1045:
            print("💡 Suggestion: Access denied. Check username/password.")
        elif e.args[0] == 1049:
            print(f"💡 Suggestion: Database '{DB_CONFIG['database']}' doesn't exist.")
        elif e.args[0] == 1130:
            print("💡 Suggestion: Your hosting blocks external connections. Check cPanel Remote MySQL.")
        
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

@app.route('/debug', methods=['GET'])
def debug_info():
    """Get detailed debug information"""
    # Get Render instance info
    render_external_url = os.environ.get('RENDER_EXTERNAL_URL', 'Not set')
    render_service_id = os.environ.get('RENDER_SERVICE_ID', 'Not set')
    
    # Try to get public IP
    try:
        import requests
        ip_response = requests.get('https://api.ipify.org?format=json', timeout=2)
        public_ip = ip_response.json().get('ip', 'Unknown')
    except:
        public_ip = 'Unknown'
    
    # Test database connection
    db_status = "Not tested"
    db_error = None
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT USER() as user, DATABASE() as db")
                db_info = cursor.fetchone()
            conn.close()
            db_status = f"Connected as {db_info['user']} to {db_info['db']}"
        else:
            db_status = "Failed to connect"
    except Exception as e:
        db_error = str(e)
        db_status = "Error"
    
    return jsonify({
        'service': 'driver-ml-api',
        'timestamp': datetime.now().isoformat(),
        'render_info': {
            'external_url': render_external_url,
            'service_id': render_service_id,
            'public_ip': public_ip
        },
        'database_config': {
            'host': DB_CONFIG['host'],
            'user': DB_CONFIG['user'],
            'database': DB_CONFIG['database'],
            'port': DB_CONFIG['port']
        },
        'database_status': db_status,
        'database_error': db_error,
        'instructions': 'Add this IP to your cPanel Remote MySQL: ' + public_ip
    })

# [KEEP ALL YOUR EXISTING ROUTES - get-driver-performance, get-ml-summary, etc.]

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 50)
    print("🚀 Driver ML API - Database Debug Mode")
    print("=" * 50)
    print(f"📡 Port: {port}")
    print(f"🗄️  Database Host: {DB_CONFIG['host']}")
    print(f"👤 Database User: {DB_CONFIG['user']}")
    print(f"📂 Database Name: {DB_CONFIG['database']}")
    print("=" * 50)
    
    # Test connection on startup
    print("\n🔍 Testing database connection on startup...")
    conn = get_db_connection()
    if conn:
        conn.close()
        print("✅ Database connection successful!")
    else:
        print("❌ Database connection failed!")
        print("\n💡 To fix this:")
        print("1. Go to your cPanel → Remote MySQL")
        print("2. Add the Render IP address (shown in /debug endpoint)")
        print("3. Allow connections from that IP")
    
    app.run(host='0.0.0.0', port=port, debug=False)
