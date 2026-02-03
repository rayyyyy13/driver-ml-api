now this is my app.py 

# app.py - SIMPLIFIED with Enhanced ML PHP API Integration
import os
import json
import random
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
PHP_API_KEY = "ML_API_ASFWGKISD"
PHP_API_URL = "https://log2.health-ease-hospital.com/admin/api.php"

# Sample data for demonstration
REAL_STATS = {
    'drivers': {
        'total_drivers': 24,
        'active_drivers': 18
    },
    'trips': {
        'completed_trips': 150,
        'on_time_trips': 120,
        'avg_distance': 28.5,
        'max_distance': 65.3,
        'total_distance': 2450
    },
    'performance_distribution': {
        'excellent': 4,
        'good': 8,
        'average': 5,
        'needs_improvement': 1
    },
    'top_drivers': [
        {'driver_id': 1, 'name': 'Driver 1', 'performance_score': 95.0, 'performance_category': 'Excellent'},
        {'driver_id': 3, 'name': 'Driver 3', 'performance_score': 92.5, 'performance_category': 'Excellent'},
        {'driver_id': 7, 'name': 'Driver 7', 'performance_score': 89.0, 'performance_category': 'Good'}
    ]
}

TRAINING_SUMMARY = {
    'performance_model': {
        'accuracy': '98.7%',
        'algorithm': 'RandomForestRegressor',
        'samples': 100,
        'features': ['completed_trips', 'on_time_rate', 'avg_distance', 'experience']
    },
    'delay_model': {
        'accuracy': '70.6%',
        'algorithm': 'RandomForestClassifier',
        'samples': 150,
        'features': ['distance_km', 'hour_of_day', 'driver_experience', 'day_of_week']
    }
}

print("🤖 Loading trained ML models...")

try:
    # First try to load actual files if they exist
    if os.path.exists('database_stats.json'):
        with open('database_stats.json', 'r') as f:
            REAL_STATS = json.load(f)
        print("✅ Loaded database_stats.json")
    
    if os.path.exists('training_summary.json'):
        with open('training_summary.json', 'r') as f:
            TRAINING_SUMMARY = json.load(f)
        print("✅ Loaded training_summary.json")
    
    MODELS_LOADED = True
    print("✅ Mock models created based on your trained ML results!")
    
except Exception as e:
    print(f"⚠️  Error loading files: {e}")
    print("ℹ️  Using built-in sample data for demonstration")
    MODELS_LOADED = True  # Still loaded with sample data

# Simple prediction functions without numpy
def predict_performance(driver_id):
    """Mimic your 98.7% accurate performance model"""
    base = 75 + (driver_id % 5 * 5)
    
    if driver_id in [1, 3, 7, 11]:
        return min(98, base + 15)  # Top performers
    elif driver_id in [5, 9, 13]:
        return max(55, base - 10)  # Needs improvement
    else:
        # Use random.random() instead of numpy.random.normal
        random_factor = (random.random() * 6) - 3  # Range -3 to 3
        return min(95, max(60, base + random_factor))

def predict_delay_probability(driver_id, distance_km, hour):
    """Mimic your 70.6% accurate delay model"""
    base = 0.3
    
    # Distance factor
    distance_factor = min(0.4, (distance_km / 100) * 0.5)
    
    # Time factor (rush hour)
    time_factor = 0.2 if 7 <= hour <= 9 or 16 <= hour <= 18 else 0
    
    # Driver factor
    driver_factor = (driver_id % 5) * 0.05
    
    return min(0.9, max(0.1, base + distance_factor + time_factor + driver_factor))

# Live PHP API integration functions
def get_live_data_from_php():
    """Fetch live data from your PHP API"""
    try:
        response = requests.get(
            f"{PHP_API_URL}?action=summary&api_key={PHP_API_KEY}",
            timeout=5,
            verify=False  # Disable SSL verification if needed
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                source = data.get('source', 'unknown')
                if source == 'ml_enhanced':
                    print(f"✅ Using ML-enhanced data (generated: {data.get('generated_at', 'N/A')})")
                elif source == 'live_database_basic':
                    print(f"ℹ️ Using basic data - run ml_sync.php for enhanced ML analysis")
                else:
                    print(f"ℹ️ Using data from source: {source}")
                return data
        return None
    except requests.exceptions.Timeout:
        print("⚠️  PHP API timeout - server may be sleeping")
        return None
    except requests.exceptions.ConnectionError:
        print("⚠️  PHP API connection error - server may be offline")
        return None
    except Exception as e:
        print(f"⚠️  Error fetching from PHP API: {e}")
        return None

def get_php_driver_performance(driver_id):
    """Get driver performance from PHP API"""
    try:
        print(f"🔄 Fetching driver performance for ID: {driver_id}")
        
        response = requests.post(
            PHP_API_URL,
            data={'action': 'driver-performance', 'driver_id': driver_id, 'api_key': PHP_API_KEY},
            timeout=3,
            verify=False
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                source = data.get('source', 'unknown')
                if source == 'ml_enhanced':
                    print(f"✅ Using ML-enhanced driver data")
                else:
                    print(f"ℹ️ Using basic driver data")
                return data
            else:
                print(f"⚠️  PHP API returned success: false for driver")
                return None
        else:
            print(f"⚠️  PHP API returned status: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        print("⚠️  PHP API timeout for driver data")
        return None
    except Exception as e:
        print(f"⚠️  Error fetching driver data: {e}")
        return None

def get_php_ml_health():
    """Check ML health status from PHP API"""
    try:
        print("🔄 Checking ML health status...")
        
        response = requests.get(
            f"{PHP_API_URL}?action=ml-health&api_key={PHP_API_KEY}",
            timeout=3,
            verify=False
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ ML health status fetched successfully")
                return data
        return None
    except Exception as e:
        print(f"⚠️  Error checking ML health: {e}")
        return None

@app.route('/')
def home():
    live_data = get_live_data_from_php()
    php_connected = live_data is not None
    ml_enhanced = live_data and live_data.get('source') == 'ml_enhanced'
    
    return jsonify({
        'message': 'Driver ML API 🤖 - Powered by ' + ('ML-ENHANCED' if ml_enhanced else 'LIVE') + ' Database Data',
        'status': 'running',
        'ml_status': 'active',
        'php_api_connected': php_connected,
        'ml_enhanced': ml_enhanced,
        'model_accuracy': {
            'performance': '98.7% (trained locally)',
            'delay_prediction': '70.6% (trained locally)'
        },
        'data_source': 'ML-Enhanced PHP API' if ml_enhanced else 'Basic PHP API',
        'data_source_detail': live_data.get('source', 'unknown') if live_data else 'disconnected',
        'drivers_count': live_data.get('summary', {}).get('total_drivers', 0) if live_data else 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/get-ml-summary', methods=['GET'])
def get_ml_summary():
    """Get ML summary - Try live data first"""
    print("📊 Getting ML summary...")
    
    live_data = get_live_data_from_php()
    
    if live_data:
        summary = live_data.get('summary', {})
        source = live_data.get('source', 'unknown')
        is_ml_enhanced = source == 'ml_enhanced'
        
        # Ensure trip_statistics exists, use default if not
        trip_statistics = summary.get('trip_statistics', {})
        if not trip_statistics:
            # Fallback to calculating from summary data
            trip_statistics = {
                'completed_trips': summary.get('distance_analysis', {}).get('estimated_trips', 0) or 0,
                'estimated_trips': True
            }
        
        # Build response with proper trip statistics
        response_data = {
            'success': True,
            'summary': {
                'total_drivers': summary.get('total_drivers', 0),
                'active_drivers': summary.get('active_drivers', 0),
                'average_on_time_rate': summary.get('average_on_time_rate', 75.0),
                'average_performance_score': summary.get('average_performance_score', 78.5),
                'performance_distribution': summary.get('performance_distribution', {
                    'excellent': 0, 'good': 0, 'average': 0, 'needs_improvement': 0
                }),
                'distance_analysis': summary.get('distance_analysis', {
                    'average_trip_distance_km': 25.0,
                    'maximum_trip_distance_km': 50.0,
                    'total_distance_km': 1000.0
                }),
                'trip_statistics': trip_statistics
            },
            'drivers': live_data.get('drivers', []),
            'source': source,
            'ml_info': {
                'enhanced': is_ml_enhanced,
                'generated_at': live_data.get('generated_at', ''),
                'model_accuracy': live_data.get('model_accuracy', {
                    'performance': '98.7%',
                    'delay_prediction': '70.6%'
                }),
                'data_quality': live_data.get('data_quality', {}),
                'note': 'ML-enhanced predictions active' if is_ml_enhanced else 'Basic data analysis'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if not is_ml_enhanced:
            response_data['ml_info']['recommendation'] = 'Run ml_sync.php for ML enhancement'
        
        print(f"✅ Using {source} data with {trip_statistics.get('completed_trips', 0)} completed trips")
        return jsonify(response_data)
    else:
        # Fallback to trained model data
        print("ℹ️  Using local trained model data as fallback")
        if MODELS_LOADED and REAL_STATS:
            drivers = REAL_STATS.get('drivers', {})
            trips = REAL_STATS.get('trips', {})
            
            completed = trips.get('completed_trips', 0) or 1
            on_time = trips.get('on_time_trips', 0) or 0
            on_time_rate = (on_time / completed) * 100
            
            return jsonify({
                'success': True,
                'summary': {
                    'total_drivers': drivers.get('total_drivers', 0),
                    'active_drivers': drivers.get('active_drivers', 0),
                    'average_on_time_rate': round(on_time_rate, 1),
                    'average_performance_score': 78.5,
                    'performance_distribution': REAL_STATS.get('performance_distribution', {
                        'excellent': 0, 'good': 0, 'average': 0, 'needs_improvement': 0
                    }),
                    'distance_analysis': {
                        'average_trip_distance_km': round(trips.get('avg_distance', 28.5), 1),
                        'maximum_trip_distance_km': round(trips.get('max_distance', 65.3), 1),
                        'total_distance_km': round(trips.get('total_distance', 2450), 0)
                    },
                    'trip_statistics': {
                        'completed_trips': trips.get('completed_trips', 150),
                        'on_time_trips': trips.get('on_time_trips', 120),
                        'total_trips': trips.get('total_trips', 200),
                        'note': 'Mock data - PHP API disconnected'
                    }
                },
                'drivers': REAL_STATS.get('top_drivers', []),
                'source': 'local_trained_models_fallback',
                'ml_info': {
                    'enhanced': False,
                    'model_accuracy': {
                        'performance': '98.7%',
                        'delay_prediction': '70.6%'
                    },
                    'note': 'Local mock models - PHP API disconnected',
                    'recommendation': 'Connect to PHP API for real data'
                },
                'timestamp': datetime.now().isoformat()
            })
        else:
            # Ultimate fallback
            return jsonify({
                'success': True,
                'summary': {
                    'total_drivers': 24,
                    'active_drivers': 18,
                    'average_on_time_rate': 82.5,
                    'average_performance_score': 78.5,
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
                    },
                    'trip_statistics': {
                        'completed_trips': 150,
                        'on_time_trips': 120,
                        'total_trips': 200,
                        'note': 'Static fallback data'
                    }
                },
                'drivers': [
                    {'driver_id': 1, 'name': 'Driver 1', 'performance_category': 'Excellent', 'performance_score': 95.0},
                    {'driver_id': 3, 'name': 'Driver 3', 'performance_category': 'Excellent', 'performance_score': 92.5},
                    {'driver_id': 7, 'name': 'Driver 7', 'performance_category': 'Good', 'performance_score': 89.0}
                ],
                'source': 'static_fallback',
                'ml_info': {
                    'enhanced': False,
                    'model_accuracy': {
                        'performance': '98.7%',
                        'delay_prediction': '70.6%'
                    },
                    'note': 'Static fallback - PHP API disconnected',
                    'recommendation': 'Check PHP API connection'
                },
                'timestamp': datetime.now().isoformat()
            })

@app.route('/get-driver-performance', methods=['POST'])
def get_driver_performance():
    """Get driver performance with enhanced ML data first, fallback to mock"""
    try:
        data = request.json
        driver_id = data.get('driver_id', 1)
        print(f"👤 Getting performance for driver ID: {driver_id}")
        
        # Try to get enhanced data from PHP API first
        php_data = get_php_driver_performance(driver_id)
        
        if php_data and php_data.get('success'):
            source = php_data.get('source', 'unknown')
            is_ml_enhanced = source == 'ml_enhanced'
            
            print(f"✅ Using PHP API data (source: {source})")
            
            # Add timestamp and ML info
            response_data = php_data.copy()
            response_data['timestamp'] = datetime.now().isoformat()
            response_data['ml_info'] = {
                'enhanced': is_ml_enhanced,
                'model_accuracy': '98.7%',
                'generated_at': php_data.get('ml_info', {}).get('generated_at', 'N/A') if is_ml_enhanced else 'N/A'
            }
            
            if not is_ml_enhanced:
                response_data['ml_info']['recommendation'] = 'Run ml_sync.php for ML-enhanced predictions'
            
            return jsonify(response_data)
        
        # Fallback to trained model pattern
        print("ℹ️  Using local trained model pattern as fallback")
        score = predict_performance(driver_id)
        
        # Determine category
        if score >= 85:
            category = "Excellent"
        elif score >= 70:
            category = "Good"
        elif score >= 50:
            category = "Average"
        else:
            category = "Needs Improvement"
        
        return jsonify({
            'success': True,
            'driver': {
                'driver_id': driver_id,
                'name': f'Driver {driver_id}',
                'performance_metrics': {
                    'performance_score': round(score, 1),
                    'performance_category': category,
                    'prediction_source': 'local_trained_model_pattern',
                    'model_accuracy': '98.7%',
                    'training_algorithm': 'RandomForestRegressor',
                    'ml_enhanced': False
                }
            },
            'ml_info': {
                'enhanced': False,
                'accuracy': '98.7%',
                'algorithm': 'Random Forest',
                'features_used': ['completed_trips', 'on_time_rate', 'avg_distance', 'experience'],
                'note': 'Local mock model - PHP API disconnected',
                'recommendation': 'Connect to PHP API for ML-enhanced predictions'
            },
            'source': 'local_model_fallback',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in get-driver-performance: {e}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'source': 'error_fallback',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict-delay', methods=['POST'])
def predict_delay():
    """Predict delay using patterns from your trained model"""
    try:
        data = request.json
        driver_id = data.get('driver_id', 1)
        distance_km = float(data.get('distance_km', 25))
        
        print(f"⏱️  Predicting delay for driver {driver_id}, distance {distance_km}km")
        
        # Get current hour
        hour = datetime.now().hour
        
        # Use pattern from your trained model (70.6% accuracy!)
        delay_probability = predict_delay_probability(driver_id, distance_km, hour)
        predicted_delay = distance_km * 0.3 * delay_probability * 2
        
        return jsonify({
            'success': True,
            'prediction': {
                'delay_probability': round(delay_probability, 3),
                'predicted_delay_minutes': round(predicted_delay, 1),
                'will_be_delayed': delay_probability > 0.5,
                'prediction_source': 'trained_ml_model_pattern',
                'model_accuracy': '70.6%',
                'training_algorithm': 'RandomForestClassifier',
                'ml_enhanced': False
            },
            'ml_info': {
                'enhanced': False,
                'accuracy': '70.6%',
                'algorithm': 'Random Forest Classifier',
                'features_used': ['distance_km', 'hour_of_day', 'driver_experience', 'day_of_week'],
                'note': 'Local mock model - Connect to PHP API for enhanced predictions'
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in predict-delay: {e}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'source': 'error_fallback',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check with ML status"""
    # Test PHP connection and ML status
    live_data = get_live_data_from_php()
    
    ml_status = {
        'ml_enhanced': False,
        'ml_models_valid': False,
        'generated_at': 'N/A'
    }
    
    if live_data and live_data.get('source') == 'ml_enhanced':
        ml_status = {
            'ml_enhanced': True,
            'ml_models_valid': True,
            'generated_at': live_data.get('generated_at', 'N/A'),
            'drivers_analyzed': len(live_data.get('drivers', [])),
            'data_quality': live_data.get('data_quality', {})
        }
    
    return jsonify({
        'status': 'ok',
        'service': 'driver-ml-api',
        'php_api_connected': live_data is not None,
        'php_api_source': live_data.get('source', 'disconnected') if live_data else 'disconnected',
        'ml_status': ml_status,
        'flask_api': {
            'models_loaded': MODELS_LOADED,
            'trained_models_available': os.path.exists('training_summary.json'),
            'database_stats_available': os.path.exists('database_stats.json')
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/ml-health', methods=['GET'])
def ml_health():
    """Detailed ML health check"""
    print("🧪 Running detailed ML health check...")
    
    # Get PHP ML health
    php_ml_health = get_php_ml_health()
    
    # Get current data
    live_data = get_live_data_from_php()
    
    ml_enhanced = live_data and live_data.get('source') == 'ml_enhanced'
    
    response = {
        'success': True,
        'php_api': {
            'connected': live_data is not None,
            'ml_enhanced': ml_enhanced,
            'source': live_data.get('source', 'disconnected') if live_data else 'disconnected',
            'ml_health': php_ml_health.get('ml_status', {}) if php_ml_health else None
        },
        'flask_api': {
            'models_loaded': MODELS_LOADED,
            'trained_models': os.path.exists('training_summary.json'),
            'database_stats': os.path.exists('database_stats.json')
        },
        'recommendations': []
    }
    
    # Add recommendations
    if not live_data:
        response['recommendations'].append('Connect to PHP API at ' + PHP_API_URL)
    elif not ml_enhanced:
        response['recommendations'].append('Run ml_sync.php on your server to generate ML-enhanced data')
    
    if not MODELS_LOADED:
        response['recommendations'].append('Upload training_summary.json and database_stats.json to Render')
    
    response['timestamp'] = datetime.now().isoformat()
    
    return jsonify(response)

@app.route('/get-sample-data', methods=['GET'])
def get_sample_data():
    """Endpoint to view the sample data"""
    return jsonify({
        'success': True,
        'real_stats': REAL_STATS,
        'training_summary': TRAINING_SUMMARY,
        'note': 'This data is used when database_stats.json and training_summary.json are not available',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/test-connection', methods=['GET'])
def test_connection():
    """Test connection to PHP API with ML status"""
    print("🧪 Testing PHP API connection with ML enhancement...")
    
    # Test enhanced data
    live_data = get_live_data_from_php()
    
    # Test driver data
    driver_data = get_php_driver_performance(1)
    
    # Test ML health
    ml_health = get_php_ml_health()
    
    if live_data:
        trip_stats = live_data.get('summary', {}).get('trip_statistics', {})
        completed_trips = trip_stats.get('completed_trips', 0)
        
        return jsonify({
            'success': True,
            'php_api_status': 'connected',
            'data_source': live_data.get('source', 'unknown'),
            'ml_enhanced': live_data.get('source') == 'ml_enhanced',
            'total_drivers': live_data.get('summary', {}).get('total_drivers', 0),
            'completed_trips': completed_trips,
            'driver_data_available': driver_data is not None,
            'ml_health_available': ml_health is not None,
            'generated_at': live_data.get('generated_at', 'N/A'),
            'timestamp': datetime.now().isoformat(),
            'recommendation': 'Run ml_sync.php on your server to generate ML-enhanced data' if live_data.get('source') != 'ml_enhanced' else 'ML enhancement active'
        })
    else:
        return jsonify({
            'success': False,
            'php_api_status': 'disconnected',
            'error': 'Could not connect to PHP API',
            'php_api_url': PHP_API_URL,
            'timestamp': datetime.now().isoformat(),
            'recommendation': 'Check PHP API URL and ensure server is running'
        }), 500

@app.route('/test-full', methods=['GET'])
def test_full():
    """Test all components"""
    results = {
        'flask_api': 'running',
        'timestamp': datetime.now().isoformat()
    }
    
    # Test PHP connection with ML
    live_data = get_live_data_from_php()
    results['php_api'] = 'connected' if live_data else 'disconnected'
    results['php_data_source'] = live_data.get('source') if live_data else 'none'
    results['php_ml_enhanced'] = live_data and live_data.get('source') == 'ml_enhanced'
    
    # Get trip statistics if available
    if live_data:
        trip_stats = live_data.get('summary', {}).get('trip_statistics', {})
        results['completed_trips'] = trip_stats.get('completed_trips', 0)
        results['trip_data_available'] = results['completed_trips'] > 0
    
    # Test ML health
    ml_health = get_php_ml_health()
    results['php_ml_health'] = 'available' if ml_health else 'unavailable'
    
    # Test sample data loading
    results['flask_models'] = 'loaded' if MODELS_LOADED else 'failed'
    
    # Test prediction functions
    try:
        score = predict_performance(1)
        delay = predict_delay_probability(1, 25, 12)
        results['prediction_functions'] = 'working'
        results['sample_score'] = score
        results['sample_delay_prob'] = delay
    except Exception as e:
        results['prediction_functions'] = f'error: {str(e)}'
    
    return jsonify({
        'success': True,
        'results': results,
        'configuration': {
            'php_api_url': PHP_API_URL,
            'api_key_masked': PHP_API_KEY[:8] + '...' if PHP_API_KEY else 'not_set'
        },
        'recommendations': [
            'Ensure ml_models.json exists in your server cache directory',
            'Run ml_sync.php to generate ML-enhanced data',
            'Check PHP API is accessible from Render'
        ] if live_data and live_data.get('source') != 'ml_enhanced' else []
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print("="*80)
    print("🚀 DRIVER ML API - WITH ENHANCED ML PHP API INTEGRATION")
    print("="*80)
    print("✅ ML Status: ACTIVE")
    print("🔗 Enhanced PHP API: ENABLED")
    print(f"🔑 API Key: {PHP_API_KEY[:8]}...")
    print(f"🌐 PHP API URL: {PHP_API_URL}")
    print("🤖 No numpy/scikit-learn dependency")
    print("⚡ Lightweight deployment with ML enhancement")
    print("="*80)
    print(f"🌐 Flask API running on port {port}")
    print("="*80)
    print("Available endpoints:")
    print("  GET  / - API home with ML status")
    print("  GET  /get-ml-summary - ML-enhanced summary")
    print("  POST /get-driver-performance - Driver performance (ML-enhanced)")
    print("  POST /predict-delay - Delay prediction")
    print("  GET  /health - Health check with ML status")
    print("  GET  /ml-health - Detailed ML health check")
    print("  GET  /get-sample-data - View sample data")
    print("  GET  /test-connection - Test PHP API with ML")
    print("  GET  /test-full - Test all components")
    print("="*80)
    print("📡 Testing PHP API connection with ML enhancement...")
    
    # Test connection on startup
    test_result = get_live_data_from_php()
    if test_result:
        source = test_result.get('source', 'unknown')
        trip_stats = test_result.get('summary', {}).get('trip_statistics', {})
        completed_trips = trip_stats.get('completed_trips', 0)
        
        print(f"✅ PHP API connection successful!")
        print(f"✅ Data source: {source}")
        print(f"✅ Completed trips: {completed_trips}")
        
        if source == 'ml_enhanced':
            print("✅ ML enhancement: ACTIVE")
            print(f"✅ Generated: {test_result.get('generated_at', 'N/A')}")
        else:
            print("⚠️  ML enhancement: NOT ACTIVE")
            print("⚠️  Run ml_sync.php on your server to generate enhanced ML data")
    else:
        print("⚠️  PHP API connection failed - will use fallback data")
        print("⚠️  Check your PHP API is running at: " + PHP_API_URL)
    
    print("="*80)
    
    app.run(host='0.0.0.0', port=port, debug=False)

Fix the "150 trips" issue in app.py:
Update your app.py file on Render:

python
# In the get_ml_summary function, update this section:
@app.route('/get-ml-summary', methods=['GET'])
def get_ml_summary():
    """Get ML summary - Try live data first"""
    live_data = get_live_data_from_php()
    
    if live_data:
        summary = live_data['summary']
        
        # Use the actual completed trips count from your PHP API
        completed_trips = summary.get('trip_statistics', {}).get('completed_trips', 1)
        
        return jsonify({
            'success': True,
            'summary': {
                'total_drivers': summary.get('total_drivers', 0),
                'active_drivers': summary.get('active_drivers', 0),
                'average_on_time_rate': summary.get('average_on_time_rate', 50.0),
                'average_performance_score': summary.get('average_performance_score', 78.5),
                'performance_distribution': summary.get('performance_distribution', {
                    'excellent': 0, 'good': 0, 'average': 0, 'needs_improvement': 0
                }),
                'distance_analysis': summary.get('distance_analysis', {
                    'average_trip_distance_km': 25.0,
                    'maximum_trip_distance_km': 50.0,
                    'total_distance_km': 1000.0
                }),
                'trip_statistics': {
                    'completed_trips': completed_trips  # Use real value
                }
            },
            'drivers': live_data.get('drivers', []),
            'source': 'ml_enhanced',
            'ml_training': {
                'performance_accuracy': '98.7%',
                'delay_accuracy': '70.6%',
                'algorithm': 'Random Forest (scikit-learn)',
                'training_data': 'Your actual database'
            },
            'note': f'Live data from PHP API - {completed_trips} trips analyzed'
        })
    else:
        # Fallback to basic data
        return jsonify({
            'success': True,
            'summary': {
                'total_drivers': 17,
                'active_drivers': 15,
                'average_on_time_rate': 100.0,
                'average_performance_score': 73.3,
                'performance_distribution': {
                    'excellent': 0,
                    'good': 1,
                    'average': 14,
                    'needs_improvement': 0
                },
                'distance_analysis': {
                    'average_trip_distance_km': 36.5,
                    'maximum_trip_distance_km': 36.5,
                    'total_distance_km': 36
                },
                'trip_statistics': {
                    'completed_trips': 1  # Your actual trip count
                }
            },
            'drivers': [
                {
                    'driver_id': '1',
                    'name': 'Juan Dela Cruz',
                    'performance_score': 73.3,
                    'performance_category': 'Good',
                    'avg_distance_km': 24.3
                }
            ],
            'source': 'mock_fallback_with_real_data',
            'note': 'Using cached data - PHP API not reachable'
        })


write the complete codes with the changes
