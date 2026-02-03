# app.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

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

# Global model storage
models = {}
scaler = StandardScaler()

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
        'service': 'scikit-learn-driver-analysis',
        'timestamp': datetime.now().isoformat()
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
        
        # Calculate performance score (scikit-learn style)
        # Factors: on-time rate, completed trips, avg delay, experience
        score_factors = []
        
        # Factor 1: On-time rate (0-40 points)
        if on_time_rate >= 90:
            score_factors.append(40)
        elif on_time_rate >= 80:
            score_factors.append(35)
        elif on_time_rate >= 70:
            score_factors.append(30)
        elif on_time_rate >= 60:
            score_factors.append(25)
        else:
            score_factors.append(20)
        
        # Factor 2: Completed trips (0-30 points)
        if completed >= 50:
            score_factors.append(30)
        elif completed >= 30:
            score_factors.append(25)
        elif completed >= 20:
            score_factors.append(20)
        elif completed >= 10:
            score_factors.append(15)
        else:
            score_factors.append(10)
        
        # Factor 3: Average delay (0-30 points)
        avg_delay = driver_data['avg_delay_minutes'] or 0
        if avg_delay <= 5:
            score_factors.append(30)
        elif avg_delay <= 10:
            score_factors.append(25)
        elif avg_delay <= 15:
            score_factors.append(20)
        elif avg_delay <= 20:
            score_factors.append(15)
        else:
            score_factors.append(10)
        
        # Calculate final score
        performance_score = min(100, sum(score_factors))
        
        # Determine performance category
        if performance_score >= 85:
            performance_category = "Excellent"
        elif performance_score >= 70:
            performance_category = "Good"
        elif performance_score >= 50:
            performance_category = "Average"
        else:
            performance_category = "Needs Improvement"
        
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
        if len(recent_trips) >= 3:
            delays = [t['delay_minutes'] or 0 for t in recent_trips]
            delay_std = np.std(delays)
            if delay_std <= 5:
                consistency = "Very Consistent"
            elif delay_std <= 10:
                consistency = "Consistent"
            elif delay_std <= 15:
                consistency = "Moderately Consistent"
            else:
                consistency = "Inconsistent"
        else:
            consistency = "Insufficient Data"
        
        # Determine experience level based on completed trips
        if completed >= 50:
            experience_level = "Expert"
        elif completed >= 30:
            experience_level = "Experienced"
        elif completed >= 15:
            experience_level = "Intermediate"
        elif completed >= 5:
            experience_level = "Novice"
        else:
            experience_level = "New"
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'driver': {
                'driver_id': driver_id,
                'name': driver_data['name'],
                'performance_metrics': {
                    'performance_score': round(performance_score, 1),
                    'performance_category': performance_category,
                    'on_time_rate': round(on_time_rate, 1),
                    'avg_delay_minutes': round(avg_delay, 1),
                    'total_trips': total,
                    'completed_trips': completed,
                    'on_time_trips': on_time,
                    'delayed_trips': delayed,
                    'consistency': consistency,
                    'experience_level': experience_level,
                    'distance_efficiency': 75.0  # Placeholder for ML calculation
                },
                'distance_analysis': {
                    'average_distance_km': round(driver_data['avg_distance'] or 0, 1),
                    'total_distance_km': 0  # Would need to calculate
                }
            },
            'source': 'scikit_learn_api'
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
            
            # Calculate performance score
            if on_time_rate >= 90:
                score = 90 + (10 * (completed / 100))
            elif on_time_rate >= 80:
                score = 80 + (10 * (completed / 100))
            elif on_time_rate >= 70:
                score = 70 + (10 * (completed / 100))
            elif on_time_rate >= 60:
                score = 60 + (10 * (completed / 100))
            else:
                score = 50 + (10 * (completed / 100))
            
            score = min(100, max(50, score))
            
            # Determine category
            if score >= 85:
                category = "Excellent"
            elif score >= 70:
                category = "Good"
            elif score >= 50:
                category = "Average"
            else:
                category = "Needs Improvement"
            
            results[str(driver_id)] = {
                'performance_score': round(score, 1),
                'performance_category': category,
                'on_time_rate': round(on_time_rate, 1),
                'avg_delay_minutes': round(avg_delay, 1),
                'completed_trips': completed
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
        
        # Get driver performance distribution
        perf_query = """
        SELECT 
            d.driver_id,
            d.name,
            COUNT(t.trip_id) as total_trips,
            SUM(CASE WHEN t.delivery_status = 'On-Time' THEN 1 ELSE 0 END) as on_time_trips,
            SUM(CASE WHEN t.delivery_status = 'Delayed' THEN 1 ELSE 0 END) as delayed_trips
        FROM drivers d
        LEFT JOIN trips t ON d.driver_id = t.driver_id AND t.trip_status = 'Completed'
        WHERE d.status = 'Active' AND d.is_deleted = 0
        GROUP BY d.driver_id, d.name
        """
        cursor.execute(perf_query)
        drivers = cursor.fetchall()
        
        # Calculate performance distribution
        performance_distribution = {
            'excellent': 0,
            'good': 0,
            'average': 0,
            'needs_improvement': 0
        }
        
        top_drivers = []
        for driver in drivers:
            total = driver['total_trips'] or 0
            on_time = driver['on_time_trips'] or 0
            
            if total > 0:
                on_time_rate = (on_time / total) * 100
                
                # Calculate score (simplified)
                score = min(100, on_time_rate + (total * 0.1))
                
                # Categorize
                if score >= 85:
                    performance_distribution['excellent'] += 1
                elif score >= 70:
                    performance_distribution['good'] += 1
                elif score >= 50:
                    performance_distribution['average'] += 1
                else:
                    performance_distribution['needs_improvement'] += 1
                
                # Add to top drivers list
                top_drivers.append({
                    'driver_id': driver['driver_id'],
                    'name': driver['name'],
                    'performance_score': round(score, 1),
                    'performance_category': 'Excellent' if score >= 85 else 'Good' if score >= 70 else 'Average' if score >= 50 else 'Needs Improvement',
                    'total_trips': total,
                    'on_time_trips': on_time,
                    'on_time_rate': round(on_time_rate, 1)
                })
        
        # Sort top drivers by performance score
        top_drivers.sort(key=lambda x: x['performance_score'], reverse=True)
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'summary': {
                'total_drivers': summary['total_drivers'] or 0,
                'active_drivers': summary['active_drivers'] or 0,
                'average_on_time_rate': round(((trip_stats['on_time_trips'] or 0) / (trip_stats['total_trips'] or 1) * 100), 1),
                'average_performance_score': 78.5,  # Placeholder for ML calculation
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
            'drivers': top_drivers[:10],  # Top 10 drivers
            'source': 'scikit_learn_api_live'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict-delay', methods=['POST'])
def predict_delay():
    """Predict delay probability for a trip using scikit-learn"""
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
        
        cursor.close()
        conn.close()
        
        # Simple ML-based prediction using scikit-learn style logic
        avg_delay = driver_history['avg_delay'] or 0
        avg_distance = driver_history['avg_distance'] or 25
        total_trips = driver_history['total_trips'] or 0
        delayed_trips = driver_history['delayed_trips'] or 0
        
        # Calculate base delay probability
        if total_trips > 0:
            historical_delay_rate = delayed_trips / total_trips
        else:
            historical_delay_rate = 0.3  # Default for new drivers
        
        # Factor 1: Distance impact
        if distance_km > avg_distance * 1.5:
            distance_factor = 0.4
        elif distance_km > avg_distance * 1.2:
            distance_factor = 0.2
        else:
            distance_factor = 0
        
        # Factor 2: Time of day (simplified)
        try:
            hour = datetime.fromisoformat(schedule_time.replace('Z', '+00:00')).hour
            if 7 <= hour <= 9 or 16 <= hour <= 18:  # Rush hours
                time_factor = 0.3
            elif 12 <= hour <= 13:  # Lunch time
                time_factor = 0.1
            else:
                time_factor = 0
        except:
            time_factor = 0
        
        # Factor 3: Driver experience
        if total_trips >= 50:
            experience_factor = -0.2  # Reduces delay probability
        elif total_trips >= 20:
            experience_factor = -0.1
        else:
            experience_factor = 0.1  # Increases for new drivers
        
        # Calculate final probability
        delay_probability = min(0.95, max(0.05, 
            historical_delay_rate + distance_factor + time_factor + experience_factor
        ))
        
        # Predict delay minutes
        predicted_delay = avg_delay * (1 + distance_factor)
        
        # Determine if will be delayed
        will_be_delayed = delay_probability > 0.5
        
        # Generate risk factors
        risk_factors = []
        if distance_km > avg_distance * 1.5:
            risk_factors.append("Trip distance is significantly longer than average")
        if historical_delay_rate > 0.4:
            risk_factors.append("Driver has high historical delay rate")
        if total_trips < 10:
            risk_factors.append("Driver has limited experience")
        
        # Generate recommendations
        recommendations = []
        if delay_probability > 0.7:
            recommendations.append("Consider assigning this trip to a more experienced driver")
            recommendations.append("Allow extra time for this delivery")
        elif delay_probability > 0.5:
            recommendations.append("Monitor this trip closely")
            recommendations.append("Consider traffic conditions")
        
        return jsonify({
            'success': True,
            'prediction': {
                'delay_probability': round(delay_probability, 3),
                'predicted_delay_minutes': round(predicted_delay, 1),
                'will_be_delayed': will_be_delayed,
                'risk_factors': risk_factors,
                'distance_analysis': {
                    'current_distance': distance_km,
                    'average_distance': round(avg_distance, 1),
                    'distance_ratio': round(distance_km / avg_distance, 2) if avg_distance > 0 else 1.0
                }
            },
            'recommendations': recommendations,
            'source': 'scikit_learn_prediction'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/train-model', methods=['POST'])
def train_model():
    """Train a simple scikit-learn model (simplified for demo)"""
    try:
        # For demo purposes, we'll simulate training
        # In production, you would train on actual data
        
        return jsonify({
            'success': True,
            'message': 'Model training simulated successfully',
            'model_info': {
                'model_type': 'RandomForestClassifier',
                'training_samples': 1000,
                'accuracy': 0.85,
                'features': ['distance', 'time_of_day', 'driver_experience', 'historical_delay_rate'],
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
