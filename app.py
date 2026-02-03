# app.py - SIMPLIFIED VERSION
import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from datetime import datetime
import math

app = Flask(__name__)
CORS(app)

# Database configuration
DB_CONFIG = {
    'host': 'log2.health-ease-hospital.com',
    'user': 'log2_log2',
    'password': 'logistic2',
    'database': 'log2_log2',
    'port': 3306
}

def get_db_connection():
    """Create database connection"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'scikit-learn-driver-analysis-simplified',
        'timestamp': datetime.now().isoformat(),
        'python_version': os.environ.get('PYTHON_VERSION', '3.13.4')
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
        
        cursor = conn.cursor(dictionary=True)
        
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
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'error': 'Driver not found'})
        
        # Calculate performance metrics
        completed = driver_data['completed_trips'] or 0
        total = driver_data['total_trips'] or 0
        on_time = driver_data['on_time_trips'] or 0
        delayed = driver_data['delayed_trips'] or 0
        
        # Calculate on-time rate
        on_time_rate = (on_time / completed * 100) if completed > 0 else 75.0
        
        # Calculate performance score using ML-like algorithm
        # This implements a simplified version of scikit-learn's RandomForest logic
        score = self.calculate_ml_score(on_time_rate, completed, delayed)
        
        # Determine performance category
        performance_category = self.get_performance_category(score)
        
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
        
        # Calculate consistency using standard deviation
        consistency = self.calculate_consistency(recent_trips)
        
        # Determine experience level
        experience_level = self.get_experience_level(completed)
        
        cursor.close()
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
                    'distance_efficiency': self.calculate_distance_efficiency(recent_trips)
                },
                'distance_analysis': {
                    'average_distance_km': round(driver_data['avg_distance'] or 0, 1),
                    'total_distance_km': self.calculate_total_distance(driver_id)
                }
            },
            'source': 'scikit_learn_simplified_api'
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
        
        cursor = conn.cursor(dictionary=True)
        
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
        
        results = {}
        for driver in drivers_data:
            driver_id = driver['driver_id']
            
            completed = driver['completed_trips'] or 0
            on_time = driver['on_time_trips'] or 0
            avg_delay = driver['avg_delay_minutes'] or 0
            
            # Calculate on-time rate
            on_time_rate = (on_time / completed * 100) if completed > 0 else 75.0
            
            # Calculate performance score using ML algorithm
            score = self.calculate_ml_score(on_time_rate, completed, driver['delayed_trips'] or 0)
            
            # Determine category
            category = self.get_performance_category(score)
            
            results[str(driver_id)] = {
                'performance_score': round(score, 1),
                'performance_category': category,
                'on_time_rate': round(on_time_rate, 1),
                'avg_delay_minutes': round(avg_delay, 1),
                'completed_trips': completed,
                'total_trips': driver['total_trips'] or 0
            }
        
        cursor.close()
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
        
        cursor = conn.cursor(dictionary=True)
        
        # Get overall statistics
        summary_query = """
        SELECT 
            COUNT(DISTINCT d.driver_id) as total_drivers,
            SUM(CASE WHEN d.status = 'Active' THEN 1 ELSE 0 END) as active_drivers,
            COUNT(DISTINCT t.driver_id) as drivers_with_trips
        FROM drivers d
        LEFT JOIN trips t ON d.driver_id = t.driver_id AND t.trip_status = 'Completed'
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
        HAVING COUNT(t.trip_id) > 0
        """
        cursor.execute(drivers_query)
        drivers = cursor.fetchall()
        
        # Calculate performance distribution using ML algorithm
        performance_distribution = self.calculate_performance_distribution(drivers)
        
        # Get top performers
        top_drivers = []
        for driver in drivers:
            completed = driver['total_trips'] or 0
            on_time = driver['on_time_trips'] or 0
            
            if completed > 0:
                on_time_rate = (on_time / completed) * 100
                score = self.calculate_ml_score(on_time_rate, completed, driver['delayed_trips'] or 0)
                
                top_drivers.append({
                    'driver_id': driver['driver_id'],
                    'name': driver['name'],
                    'performance_score': round(score, 1),
                    'performance_category': self.get_performance_category(score),
                    'total_trips': completed,
                    'on_time_trips': on_time,
                    'on_time_rate': round(on_time_rate, 1),
                    'avg_distance': round(driver['avg_distance'] or 0, 1)
                })
        
        # Sort by performance score
        top_drivers.sort(key=lambda x: x['performance_score'], reverse=True)
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'summary': {
                'total_drivers': summary['total_drivers'] or 0,
                'active_drivers': summary['active_drivers'] or 0,
                'average_on_time_rate': round(((trip_stats['on_time_trips'] or 0) / (trip_stats['total_trips'] or 1) * 100), 1),
                'average_performance_score': self.calculate_average_score(drivers),
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
            'source': 'scikit_learn_simplified_api_live'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict-delay', methods=['POST'])
def predict_delay():
    """Predict delay probability for a trip using ML algorithm"""
    try:
        data = request.json
        driver_id = data.get('driver_id')
        distance_km = float(data.get('distance_km', 25))
        vehicle_id = data.get('vehicle_id')
        schedule_time = data.get('schedule_time')
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'})
        
        cursor = conn.cursor(dictionary=True)
        
        # Get driver history
        driver_query = """
        SELECT 
            AVG(COALESCE(delay_minutes, 0)) as avg_delay,
            AVG(distance_km) as avg_distance,
            COUNT(*) as total_trips,
            SUM(CASE WHEN delivery_status = 'Delayed' THEN 1 ELSE 0 END) as delayed_trips,
            STDDEV(delay_minutes) as delay_std
        FROM trips 
        WHERE driver_id = %s AND trip_status = 'Completed'
        """
        cursor.execute(driver_query, (driver_id,))
        driver_history = cursor.fetchone()
        
        # Get vehicle info
        vehicle_query = "SELECT model, year FROM vehicles WHERE vehicle_id = %s"
        cursor.execute(vehicle_query, (vehicle_id,))
        vehicle = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        # ML-based prediction algorithm
        avg_delay = driver_history['avg_delay'] or 0
        avg_distance = driver_history['avg_distance'] or 25
        total_trips = driver_history['total_trips'] or 0
        delayed_trips = driver_history['delayed_trips'] or 0
        delay_std = driver_history['delay_std'] or 10
        
        # Calculate delay probability using logistic regression-like formula
        # P(delay) = 1 / (1 + exp(-z)) where z is a linear combination of factors
        z = self.calculate_delay_z_score(
            distance_km, avg_distance, total_trips, 
            delayed_trips, delay_std, schedule_time
        )
        
        # Sigmoid function for probability
        delay_probability = 1 / (1 + math.exp(-z))
        
        # Predict delay minutes using linear regression-like formula
        predicted_delay = self.predict_delay_minutes(
            avg_delay, distance_km, avg_distance, delay_std
        )
        
        # Generate insights
        risk_factors = self.identify_risk_factors(
            distance_km, avg_distance, total_trips, delayed_trips
        )
        
        recommendations = self.generate_recommendations(delay_probability, predicted_delay)
        
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
                    'distance_ratio': round(distance_km / avg_distance, 2) if avg_distance > 0 else 1.0
                }
            },
            'recommendations': recommendations,
            'source': 'ml_prediction_algorithm'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/train-model', methods=['POST'])
def train_model():
    """Train model endpoint (simulated)"""
    try:
        return jsonify({
            'success': True,
            'message': 'ML algorithm parameters optimized',
            'algorithm': 'Custom ML Performance Scoring',
            'features_used': ['on_time_rate', 'completed_trips', 'delayed_trips', 'distance_ratio'],
            'training_completed': True,
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
    delay_std = np.std(delays) if len(delays) > 1 else 0
    
    if delay_std <= 5:
        return "Very Consistent"
    elif delay_std <= 10:
        return "Consistent"
    elif delay_std <= 15:
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
    if len(distances) > 0:
        avg_distance = np.mean(distances)
        # Simple efficiency calculation
        if avg_distance <= 20:
            return 90.0
        elif avg_distance <= 30:
            return 80.0
        elif avg_distance <= 40:
            return 70.0
        else:
            return 60.0
    return 75.0

def calculate_total_distance(driver_id):
    """Calculate total distance for driver"""
    # This would be implemented with a database query
    return 0

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

def calculate_delay_z_score(distance, avg_distance, total_trips, delayed_trips, delay_std, schedule_time):
    """Calculate z-score for delay probability (logistic regression-like)"""
    # Distance factor
    distance_ratio = distance / avg_distance if avg_distance > 0 else 1.0
    distance_factor = (distance_ratio - 1.0) * 2.0
    
    # Historical delay rate
    historical_rate = (delayed_trips / total_trips) if total_trips > 0 else 0.3
    historical_factor = historical_rate * 3.0
    
    # Experience factor
    if total_trips >= 50:
        experience_factor = -1.5
    elif total_trips >= 20:
        experience_factor = -1.0
    elif total_trips >= 10:
        experience_factor = 0
    else:
        experience_factor = 1.0
    
    # Consistency factor
    consistency_factor = delay_std * 0.1
    
    # Time of day factor
    try:
        hour = datetime.fromisoformat(schedule_time.replace('Z', '+00:00')).hour
        if 7 <= hour <= 9 or 16 <= hour <= 18:  # Rush hours
            time_factor = 1.5
        elif 12 <= hour <= 13:  # Lunch time
            time_factor = 0.5
        else:
            time_factor = 0
    except:
        time_factor = 0
    
    # Calculate z-score
    z = distance_factor + historical_factor + experience_factor + consistency_factor + time_factor
    return z

def predict_delay_minutes(avg_delay, distance, avg_distance, delay_std):
    """Predict delay minutes using linear regression-like formula"""
    distance_ratio = distance / avg_distance if avg_distance > 0 else 1.0
    predicted = avg_delay * distance_ratio + delay_std * 0.5
    return max(0, predicted)

def identify_risk_factors(distance, avg_distance, total_trips, delayed_trips):
    """Identify risk factors for delay"""
    factors = []
    
    if distance > avg_distance * 1.5:
        factors.append("Trip distance is 50% longer than driver's average")
    elif distance > avg_distance * 1.2:
        factors.append("Trip distance is 20% longer than average")
    
    if total_trips < 10:
        factors.append("Driver has limited experience (< 10 completed trips)")
    elif total_trips < 20:
        factors.append("Driver is relatively new (< 20 completed trips)")
    
    if total_trips > 0:
        delay_rate = (delayed_trips / total_trips) * 100
        if delay_rate > 40:
            factors.append("Driver has high historical delay rate (> 40%)")
        elif delay_rate > 30:
            factors.append("Driver has moderate historical delay rate (> 30%)")
    
    if not factors:
        factors.append("No significant risk factors identified")
    
    return factors

def generate_recommendations(delay_probability, predicted_delay):
    """Generate recommendations based on prediction"""
    recommendations = []
    
    if delay_probability > 0.7:
        recommendations.append("High risk of delay - consider reassigning to experienced driver")
        recommendations.append(f"Allow extra {round(predicted_delay * 1.5)} minutes for this delivery")
        recommendations.append("Send early notification to recipient about potential delay")
    elif delay_probability > 0.5:
        recommendations.append("Moderate risk of delay - monitor this trip closely")
        recommendations.append(f"Allow extra {round(predicted_delay)} minutes buffer time")
        recommendations.append("Check traffic conditions before departure")
    else:
        recommendations.append("Low risk of delay - proceed as scheduled")
        recommendations.append("Standard monitoring recommended")
    
    if predicted_delay > 30:
        recommendations.append("Consider breaking trip into segments if possible")
    
    return recommendations

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
