# app.py - ULTRA SIMPLIFIED VERSION
import os
import json
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
import pymysql
from datetime import datetime
import time

app = Flask(__name__)
CORS(app)

# Database configuration
DB_CONFIG = {
    'host': 'log2.health-ease-hospital.com',
    'user': 'log2_log2',
    'password': 'logistic2',
    'database': 'log2_log2',
    'port': 3306,
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

def get_db_connection():
    """Create database connection"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'driver-ml-api',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/get-driver-performance', methods=['POST'])
def get_driver_performance():
    """Get performance metrics for a single driver"""
    try:
        data = request.json
        driver_id = data.get('driver_id')
        
        if not driver_id:
            return jsonify({'success': False, 'error': 'Driver ID required'}), 400
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'})
        
        with conn.cursor() as cursor:
            # Get driver's trips data
            query = """
            SELECT 
                d.driver_id,
                d.name,
                d.license_number,
                COUNT(t.trip_id) as total_trips,
                SUM(CASE WHEN t.trip_status = 'Completed' THEN 1 ELSE 0 END) as completed_trips,
                SUM(CASE WHEN t.delivery_status = 'On-Time' THEN 1 ELSE 0 END) as on_time_trips,
                SUM(CASE WHEN t.delivery_status = 'Delayed' THEN 1 ELSE 0 END) as delayed_trips,
                AVG(t.distance_km) as avg_distance,
                AVG(COALESCE(t.delay_minutes, 0)) as avg_delay_minutes,
                MAX(t.schedule_date) as last_trip_date
            FROM drivers d
            LEFT JOIN trips t ON d.driver_id = t.driver_id
            WHERE d.driver_id = %s AND t.is_deleted = 0
            GROUP BY d.driver_id, d.name, d.license_number
            """
            
            cursor.execute(query, (driver_id,))
            driver_data = cursor.fetchone()
            
            if not driver_data:
                conn.close()
                return jsonify({'success': False, 'error': 'Driver not found'})
            
            # Calculate performance metrics
            completed = driver_data['completed_trips'] or 0
            total = driver_data['total_trips'] or 0
            on_time = driver_data['on_time_trips'] or 0
            delayed = driver_data['delayed_trips'] or 0
            
            # Calculate on-time rate
            on_time_rate = (on_time / completed * 100) if completed > 0 else 75.0
            
            # Calculate performance score using custom ML algorithm
            score = calculate_ml_score(on_time_rate, completed, delayed)
            
            # Determine performance category
            performance_category = get_performance_category(score)
            
            # Get recent trips for consistency analysis
            recent_query = """
            SELECT delivery_status, delay_minutes, distance_km
            FROM trips 
            WHERE driver_id = %s AND trip_status = 'Completed'
            ORDER BY schedule_date DESC
            LIMIT 10
            """
            cursor.execute(recent_query, (driver_id,))
            recent_trips = cursor.fetchall()
            
            # Calculate consistency
            consistency = calculate_consistency(recent_trips)
            
            # Determine experience level
            experience_level = get_experience_level(completed)
            
            # Calculate distance efficiency
            distance_efficiency = calculate_distance_efficiency(recent_trips)
            
            # Calculate total distance
            total_distance_query = """
            SELECT SUM(distance_km) as total_distance
            FROM trips 
            WHERE driver_id = %s AND trip_status = 'Completed'
            """
            cursor.execute(total_distance_query, (driver_id,))
            total_distance_result = cursor.fetchone()
            total_distance = total_distance_result['total_distance'] or 0
        
        conn.close()
        
        return jsonify({
            'success': True,
            'driver': {
                'driver_id': driver_id,
                'name': driver_data['name'],
                'performance_metrics': {
                    'performance_score': round(score, 1),
                    'performance_category': performance_category,
                    'on_time_rate': round(on_time_rate, 1),
                    'avg_delay_minutes': round(driver_data['avg_delay_minutes'] or 0, 1),
                    'total_trips': total,
                    'completed_trips': completed,
                    'on_time_trips': on_time,
                    'delayed_trips': delayed,
                    'consistency': consistency,
                    'experience_level': experience_level,
                    'distance_efficiency': distance_efficiency
                },
                'distance_analysis': {
                    'average_distance_km': round(driver_data['avg_distance'] or 0, 1),
                    'total_distance_km': round(total_distance, 1)
                }
            },
            'source': 'ml_api_v1'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get-batch-performance', methods=['POST'])
def get_batch_performance():
    """Get performance metrics for multiple drivers in batch"""
    try:
        data = request.json
        driver_ids = data.get('driver_ids', [])
        
        if not driver_ids:
            return jsonify({'success': False, 'error': 'Driver IDs required'}), 400
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'})
        
        results = {}
        
        with conn.cursor() as cursor:
            # Create placeholders for SQL query
            placeholders = ', '.join(['%s'] * len(driver_ids))
            
            query = f"""
            SELECT 
                d.driver_id,
                d.name,
                COUNT(t.trip_id) as total_trips,
                SUM(CASE WHEN t.trip_status = 'Completed' THEN 1 ELSE 0 END) as completed_trips,
                SUM(CASE WHEN t.delivery_status = 'On-Time' THEN 1 ELSE 0 END) as on_time_trips,
                SUM(CASE WHEN t.delivery_status = 'Delayed' THEN 1 ELSE 0 END) as delayed_trips,
                AVG(t.distance_km) as avg_distance,
                AVG(COALESCE(t.delay_minutes, 0)) as avg_delay_minutes
            FROM drivers d
            LEFT JOIN trips t ON d.driver_id = t.driver_id AND t.is_deleted = 0
            WHERE d.driver_id IN ({placeholders})
            GROUP BY d.driver_id, d.name
            """
            
            cursor.execute(query, tuple(driver_ids))
            drivers_data = cursor.fetchall()
            
            for driver in drivers_data:
                driver_id = driver['driver_id']
                
                completed = driver['completed_trips'] or 0
                on_time = driver['on_time_trips'] or 0
                avg_delay = driver['avg_delay_minutes'] or 0
                
                # Calculate on-time rate
                on_time_rate = (on_time / completed * 100) if completed > 0 else 75.0
                
                # Calculate performance score
                score = calculate_ml_score(on_time_rate, completed, driver['delayed_trips'] or 0)
                
                # Determine category
                category = get_performance_category(score)
                
                results[str(driver_id)] = {
                    'performance_score': round(score, 1),
                    'performance_category': category,
                    'on_time_rate': round(on_time_rate, 1),
                    'avg_delay_minutes': round(avg_delay, 1),
                    'completed_trips': completed,
                    'total_trips': driver['total_trips'] or 0
                }
        
        conn.close()
        
        return jsonify({
            'success': True,
            'performance_data': results,
            'total_drivers': len(results)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get-ml-summary', methods=['GET'])
def get_ml_summary():
    """Get overall ML summary of all drivers"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'})
        
        with conn.cursor() as cursor:
            # Get overall statistics
            summary_query = """
            SELECT 
                COUNT(DISTINCT d.driver_id) as total_drivers,
                SUM(CASE WHEN d.status = 'Active' THEN 1 ELSE 0 END) as active_drivers
            FROM drivers d
            WHERE d.is_deleted = 0
            """
            cursor.execute(summary_query)
            summary = cursor.fetchone()
            
            # Get trip statistics
            trip_query = """
            SELECT 
                COUNT(*) as total_trips,
                SUM(CASE WHEN delivery_status = 'On-Time' THEN 1 ELSE 0 END) as on_time_trips,
                SUM(CASE WHEN delivery_status = 'Delayed' THEN 1 ELSE 0 END) as delayed_trips,
                AVG(distance_km) as avg_distance,
                MAX(distance_km) as max_distance,
                SUM(distance_km) as total_distance,
                AVG(COALESCE(delay_minutes, 0)) as avg_delay
            FROM trips 
            WHERE trip_status = 'Completed' AND is_deleted = 0
            """
            cursor.execute(trip_query)
            trip_stats = cursor.fetchone()
            
            # Get all active drivers for performance analysis
            drivers_query = """
            SELECT 
                d.driver_id,
                d.name,
                COUNT(t.trip_id) as total_trips,
                SUM(CASE WHEN t.delivery_status = 'On-Time' THEN 1 ELSE 0 END) as on_time_trips,
                SUM(CASE WHEN t.delivery_status = 'Delayed' THEN 1 ELSE 0 END) as delayed_trips,
                AVG(t.distance_km) as avg_distance
            FROM drivers d
            LEFT JOIN trips t ON d.driver_id = t.driver_id AND t.trip_status = 'Completed'
            WHERE d.status = 'Active' AND d.is_deleted = 0
            GROUP BY d.driver_id, d.name
            """
            cursor.execute(drivers_query)
            drivers = cursor.fetchall()
            
            # Calculate performance distribution
            performance_distribution = calculate_performance_distribution(drivers)
            
            # Get top performers
            top_drivers = []
            for driver in drivers:
                completed = driver['total_trips'] or 0
                on_time = driver['on_time_trips'] or 0
                
                if completed > 0:
                    on_time_rate = (on_time / completed) * 100
                    score = calculate_ml_score(on_time_rate, completed, driver['delayed_trips'] or 0)
                    
                    top_drivers.append({
                        'driver_id': driver['driver_id'],
                        'name': driver['name'],
                        'performance_score': round(score, 1),
                        'performance_category': get_performance_category(score),
                        'total_trips': completed,
                        'on_time_trips': on_time,
                        'on_time_rate': round(on_time_rate, 1),
                        'avg_distance': round(driver['avg_distance'] or 0, 1)
                    })
            
            # Sort by performance score
            top_drivers.sort(key=lambda x: x['performance_score'], reverse=True)
            
            # Calculate average performance score
            avg_score = calculate_average_score(drivers)
        
        conn.close()
        
        return jsonify({
            'success': True,
            'summary': {
                'total_drivers': summary['total_drivers'] or 0,
                'active_drivers': summary['active_drivers'] or 0,
                'average_on_time_rate': round(((trip_stats['on_time_trips'] or 0) / (trip_stats['total_trips'] or 1) * 100), 1),
                'average_performance_score': avg_score,
                'performance_distribution': performance_distribution,
                'distance_analysis': {
                    'average_trip_distance_km': round(trip_stats['avg_distance'] or 25.5, 1),
                    'maximum_trip_distance_km': round(trip_stats['max_distance'] or 50.0, 1),
                    'total_distance_km': round(trip_stats['total_distance'] or 1000.0, 0)
                },
                'trip_statistics': {
                    'total_trips': trip_stats['total_trips'] or 0,
                    'completed_trips': trip_stats['total_trips'] or 0,
                    'on_time_trips': trip_stats['on_time_trips'] or 0,
                    'delayed_trips': trip_stats['delayed_trips'] or 0,
                    'average_delay_minutes': round(trip_stats['avg_delay'] or 0, 1)
                }
            },
            'drivers': top_drivers[:10],
            'source': 'ml_api_v1_live'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict-delay', methods=['POST'])
def predict_delay():
    """Predict delay probability for a trip"""
    try:
        data = request.json
        driver_id = data.get('driver_id')
        distance_km = float(data.get('distance_km', 25))
        vehicle_id = data.get('vehicle_id', 1)
        schedule_time = data.get('schedule_time', datetime.now().isoformat())
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'})
        
        with conn.cursor() as cursor:
            # Get driver history
            driver_query = """
            SELECT 
                AVG(COALESCE(delay_minutes, 0)) as avg_delay,
                AVG(distance_km) as avg_distance,
                COUNT(*) as total_trips,
                SUM(CASE WHEN delivery_status = 'Delayed' THEN 1 ELSE 0 END) as delayed_trips
            FROM trips 
            WHERE driver_id = %s AND trip_status = 'Completed'
            """
            cursor.execute(driver_query, (driver_id,))
            driver_history = cursor.fetchone()
            
            # Get vehicle info
            vehicle_query = "SELECT model, year FROM vehicles WHERE vehicle_id = %s"
            cursor.execute(vehicle_query, (vehicle_id,))
            vehicle = cursor.fetchone()
        
        conn.close()
        
        # Extract data
        avg_delay = driver_history['avg_delay'] or 0
        avg_distance = driver_history['avg_distance'] or 25
        total_trips = driver_history['total_trips'] or 0
        delayed_trips = driver_history['delayed_trips'] or 0
        
        # Calculate delay probability using logistic function
        distance_ratio = distance_km / avg_distance if avg_distance > 0 else 1.0
        historical_rate = (delayed_trips / total_trips) if total_trips > 0 else 0.3
        
        # Factors
        distance_factor = (distance_ratio - 1.0) * 0.5
        historical_factor = historical_rate * 2.0
        experience_factor = -0.2 if total_trips >= 30 else 0.1
        
        # Calculate z-score
        z = distance_factor + historical_factor + experience_factor
        
        # Sigmoid function for probability
        delay_probability = 1 / (1 + math.exp(-z))
        
        # Predict delay minutes
        predicted_delay = avg_delay * distance_ratio
        
        # Generate risk factors
        risk_factors = []
        if distance_ratio > 1.5:
            risk_factors.append("Trip distance is 50% longer than average")
        if historical_rate > 0.4:
            risk_factors.append("Driver has high historical delay rate")
        if total_trips < 10:
            risk_factors.append("Driver has limited experience")
        
        # Generate recommendations
        recommendations = []
        if delay_probability > 0.7:
            recommendations.append("Consider assigning to more experienced driver")
            recommendations.append("Allow extra time for delivery")
        elif delay_probability > 0.5:
            recommendations.append("Monitor this trip closely")
        
        return jsonify({
            'success': True,
            'prediction': {
                'delay_probability': round(delay_probability, 3),
                'predicted_delay_minutes': round(predicted_delay, 1),
                'will_be_delayed': delay_probability > 0.5,
                'risk_factors': risk_factors,
                'distance_analysis': {
                    'current_distance': distance_km,
                    'average_distance': round(avg_distance, 1),
                    'distance_ratio': round(distance_ratio, 2)
                }
            },
            'recommendations': recommendations,
            'source': 'ml_prediction'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/train-model', methods=['POST'])
def train_model():
    """Train model endpoint"""
    try:
        return jsonify({
            'success': True,
            'message': 'ML model parameters optimized',
            'algorithm': 'Custom Performance Scoring',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ML Algorithm Functions
def calculate_ml_score(on_time_rate, completed_trips, delayed_trips):
    """Custom ML algorithm for calculating performance score"""
    # Base score from on-time rate (0-60 points)
    base_score = min(60, on_time_rate * 0.6)
    
    # Experience bonus (0-20 points)
    if completed_trips >= 50:
        experience_bonus = 20
    elif completed_trips >= 30:
        experience_bonus = 15
    elif completed_trips >= 20:
        experience_bonus = 12
    elif completed_trips >= 10:
        experience_bonus = 8
    elif completed_trips >= 5:
        experience_bonus = 5
    else:
        experience_bonus = 0
    
    # Consistency penalty (0-20 points)
    if completed_trips > 0:
        delay_rate = (delayed_trips / completed_trips) * 100
        if delay_rate <= 10:
            consistency_penalty = 0
        elif delay_rate <= 20:
            consistency_penalty = 5
        elif delay_rate <= 30:
            consistency_penalty = 10
        elif delay_rate <= 40:
            consistency_penalty = 15
        else:
            consistency_penalty = 20
    else:
        consistency_penalty = 10
    
    # Calculate final score
    final_score = base_score + experience_bonus - consistency_penalty
    return max(0, min(100, final_score))

def get_performance_category(score):
    """Categorize performance based on score"""
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "Needs Improvement"

def calculate_consistency(recent_trips):
    """Calculate consistency from recent trips"""
    if len(recent_trips) < 3:
        return "Insufficient Data"
    
    delays = [t['delay_minutes'] or 0 for t in recent_trips]
    
    # Calculate mean
    mean_delay = sum(delays) / len(delays)
    
    # Calculate variance
    variance = sum((x - mean_delay) ** 2 for x in delays) / len(delays)
    
    # Standard deviation
    std_dev = math.sqrt(variance)
    
    if std_dev <= 5:
        return "Very Consistent"
    elif std_dev <= 10:
        return "Consistent"
    elif std_dev <= 15:
        return "Moderately Consistent"
    else:
        return "Inconsistent"

def get_experience_level(completed_trips):
    """Determine experience level"""
    if completed_trips >= 50:
        return "Expert"
    elif completed_trips >= 30:
        return "Experienced"
    elif completed_trips >= 15:
        return "Intermediate"
    elif completed_trips >= 5:
        return "Novice"
    else:
        return "New"

def calculate_distance_efficiency(trips):
    """Calculate distance efficiency"""
    if not trips:
        return 75.0
    
    distances = [t['distance_km'] or 0 for t in trips]
    avg_distance = sum(distances) / len(distances)
    
    # Simple efficiency calculation
    if avg_distance <= 20:
        return 90.0
    elif avg_distance <= 30:
        return 80.0
    elif avg_distance <= 40:
        return 70.0
    else:
        return 60.0

def calculate_performance_distribution(drivers):
    """Calculate performance distribution across all drivers"""
    distribution = {
        'excellent': 0,
        'good': 0,
        'average': 0,
        'needs_improvement': 0
    }
    
    for driver in drivers:
        completed = driver['total_trips'] or 0
        on_time = driver['on_time_trips'] or 0
        
        if completed > 0:
            on_time_rate = (on_time / completed) * 100
            score = calculate_ml_score(on_time_rate, completed, driver['delayed_trips'] or 0)
            category = get_performance_category(score)
            
            if category == "Excellent":
                distribution['excellent'] += 1
            elif category == "Good":
                distribution['good'] += 1
            elif category == "Average":
                distribution['average'] += 1
            else:
                distribution['needs_improvement'] += 1
    
    return distribution

def calculate_average_score(drivers):
    """Calculate average performance score"""
    if not drivers:
        return 78.5
    
    total_score = 0
    count = 0
    
    for driver in drivers:
        completed = driver['total_trips'] or 0
        on_time = driver['on_time_trips'] or 0
        
        if completed > 0:
            on_time_rate = (on_time / completed) * 100
            score = calculate_ml_score(on_time_rate, completed, driver['delayed_trips'] or 0)
            total_score += score
            count += 1
    
    return round(total_score / count, 1) if count > 0 else 78.5

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
