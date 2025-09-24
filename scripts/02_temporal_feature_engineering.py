#!/usr/bin/env python3
"""
Step 2: Advanced Temporal Feature Engineering
============================================

This script creates sophisticated temporal features from payment history strings,
leveraging the discovered LEFT-TO-RIGHT chronological ordering where the RIGHT side
represents the most recent payments (critical for prediction).
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class TemporalFeatureEngineer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        
        # Payment severity mapping (discovered in Phase 1)
        self.payment_severity = {
            '0': 0,    # On-time (excellent)
            '1': 1,    # 1-29 days late (mild)
            '2': 2,    # 30-59 days late (moderate)
            '3': 3,    # 60-89 days late (serious)
            '4': 4,    # 90-119 days late (severe)
            '5': 5,    # 120+ days late (write-off)
            '6': 6,    # Unknown - assume worse than 5
            '7': 7,    # Unknown - assume worse than 6
            '8': 8,    # Unknown - assume worse than 7
            '9': 9,    # Unknown - assume worst
        }
    
    def extract_temporal_features(self, is_train=True):
        """Extract temporal features from payment histories"""
        
        suffix = "train" if is_train else "test"
        accounts_file = self.data_dir / f"accounts_data_{suffix}.json"
        
        print(f"🚀 EXTRACTING TEMPORAL FEATURES FROM {suffix.upper()} DATA")
        print("="*60)
        print(f"🔄 Processing {accounts_file}...")
        
        with open(accounts_file, 'r') as f:
            accounts_raw = json.load(f)
        
        temporal_features = []
        processed = 0
        
        for record_list in accounts_raw:
            if isinstance(record_list, list):
                for account in record_list:
                    if isinstance(account, dict) and len(account) > 0:
                        uid = account.get('uid', '')
                        payment_hist = str(account.get('payment_hist_string', ''))
                        
                        if uid:
                            if payment_hist and payment_hist != 'nan':
                                features = self.extract_single_temporal_features(uid, payment_hist)
                            else:
                                features = self.create_no_history_features(uid)
                            
                            temporal_features.append(features)
                            processed += 1
                            
                            if processed % 100000 == 0:
                                print(f"   Processed {processed:,} accounts...")
        
        temporal_df = pd.DataFrame(temporal_features)
        print(f"✅ Created temporal features for {len(temporal_df):,} accounts")
        
        return temporal_df
    
    def extract_single_temporal_features(self, uid, payment_hist):
        """Extract comprehensive temporal features from a single payment history"""
        
        features = {'uid': uid, 'has_payment_history': True}
        
        # Convert to severity scores (LEFT-TO-RIGHT = old-to-new)
        severity_scores = [self.payment_severity.get(char, 0) for char in payment_hist]
        
        # FEATURE GROUP 1: RECENCY-WEIGHTED FEATURES
        # Recent payments (right side) matter more
        features.update(self.create_recency_weighted_features(severity_scores))
        
        # FEATURE GROUP 2: RECENT BEHAVIOR FEATURES
        # Focus on last 3, 6, 12 months
        features.update(self.create_recent_behavior_features(severity_scores))
        
        # FEATURE GROUP 3: TREND FEATURES
        # Payment behavior trends over time
        features.update(self.create_trend_features(severity_scores))
        
        # FEATURE GROUP 4: RISK PATTERN FEATURES
        # Critical risk indicators
        features.update(self.create_risk_pattern_features(payment_hist, severity_scores))
        
        # FEATURE GROUP 5: SEQUENCE FEATURES
        # Payment transitions and patterns
        features.update(self.create_sequence_features(payment_hist))
        
        return features
    
    def create_recency_weighted_features(self, severity_scores):
        """Create features with exponential decay weighting (recent = more important)"""
        
        if not severity_scores:
            return {'recency_weighted_score': 0, 'recency_weighted_bad_ratio': 0}
        
        # Exponential decay weights (most recent = highest weight)
        weights = [np.exp(-0.1 * i) for i in range(len(severity_scores))]
        weights.reverse()  # Reverse so recent (end) gets highest weight
        
        # Weighted severity score
        weighted_score = sum(s * w for s, w in zip(severity_scores, weights)) / sum(weights)
        
        # Weighted bad payment ratio
        bad_payments = [1 if s > 0 else 0 for s in severity_scores]
        weighted_bad_ratio = sum(b * w for b, w in zip(bad_payments, weights)) / sum(weights)
        
        return {
            'recency_weighted_score': weighted_score,
            'recency_weighted_bad_ratio': weighted_bad_ratio
        }
    
    def create_recent_behavior_features(self, severity_scores):
        """Create features focusing on recent payment behavior"""
        
        features = {}
        
        # Recent windows (3, 6, 12 months)
        for window in [3, 6, 12]:
            if len(severity_scores) >= window:
                recent_scores = severity_scores[-window:]  # Most recent N months
                
                features.update({
                    f'recent_{window}m_avg_severity': np.mean(recent_scores),
                    f'recent_{window}m_max_severity': max(recent_scores),
                    f'recent_{window}m_bad_count': sum(1 for s in recent_scores if s > 0),
                    f'recent_{window}m_severe_count': sum(1 for s in recent_scores if s >= 4)
                })
            else:
                # Not enough history - use what we have
                if len(severity_scores) > 0:
                    features.update({
                        f'recent_{window}m_avg_severity': np.mean(severity_scores),
                        f'recent_{window}m_max_severity': max(severity_scores),
                        f'recent_{window}m_bad_count': sum(1 for s in severity_scores if s > 0),
                        f'recent_{window}m_severe_count': sum(1 for s in severity_scores if s >= 4)
                    })
                else:
                    features.update({
                        f'recent_{window}m_avg_severity': 0,
                        f'recent_{window}m_max_severity': 0,
                        f'recent_{window}m_bad_count': 0,
                        f'recent_{window}m_severe_count': 0
                    })
        
        return features
    
    def create_trend_features(self, severity_scores):
        """Create trend features showing payment behavior over time"""
        
        if len(severity_scores) < 3:
            return {
                'payment_trend_slope': 0,
                'trend_direction': 0,
                'deterioration_velocity': 0
            }
        
        # Linear trend (positive = getting worse over time)
        x = np.arange(len(severity_scores))
        if len(severity_scores) > 1:
            slope = np.corrcoef(x, severity_scores)[0, 1]
            slope = slope if not np.isnan(slope) else 0
        else:
            slope = 0
        
        # Trend direction (recent vs historical)
        mid_point = len(severity_scores) // 2
        recent_avg = np.mean(severity_scores[mid_point:])
        historical_avg = np.mean(severity_scores[:mid_point])
        trend_direction = recent_avg - historical_avg
        
        # Deterioration velocity (how fast someone gets into trouble)
        max_severity = max(severity_scores)
        first_bad_index = next((i for i, s in enumerate(severity_scores) if s > 0), None)
        max_severity_index = next((i for i, s in enumerate(severity_scores) if s == max_severity), None)
        
        if first_bad_index is not None and max_severity_index is not None and max_severity_index > first_bad_index:
            deterioration_velocity = max_severity / max(max_severity_index - first_bad_index, 1)
        else:
            deterioration_velocity = 0
        
        return {
            'payment_trend_slope': slope,
            'trend_direction': trend_direction,
            'deterioration_velocity': deterioration_velocity
        }
    
    def create_risk_pattern_features(self, payment_hist, severity_scores):
        """Create features for critical risk patterns"""
        
        features = {}
        
        # Consecutive bad payments (death spiral indicator)
        max_consecutive_bad = 0
        current_bad_streak = 0
        
        for score in severity_scores:
            if score > 0:
                current_bad_streak += 1
                max_consecutive_bad = max(max_consecutive_bad, current_bad_streak)
            else:
                current_bad_streak = 0
        
        # Severe delinquency patterns
        severe_delinquency_months = sum(1 for s in severity_scores if s >= 4)
        write_off_months = sum(1 for s in severity_scores if s >= 5)
        
        # Unknown code analysis (codes 6-9)
        unknown_code_months = sum(1 for char in payment_hist if char in '6789')
        
        # Recent severe problems (last 6 months)
        recent_6m = severity_scores[-6:] if len(severity_scores) >= 6 else severity_scores
        recent_severe_flag = 1 if any(s >= 4 for s in recent_6m) else 0
        
        # Death spiral risk (consecutive bad + recent severe)
        death_spiral_risk = 1 if max_consecutive_bad >= 3 and recent_severe_flag else 0
        
        features.update({
            'max_consecutive_bad_payments': max_consecutive_bad,
            'severe_delinquency_months': severe_delinquency_months,
            'write_off_months': write_off_months,
            'unknown_code_months': unknown_code_months,
            'recent_severe_flag': recent_severe_flag,
            'death_spiral_risk': death_spiral_risk
        })
        
        return features
    
    def create_sequence_features(self, payment_hist):
        """Create sequence pattern features"""
        
        features = {}
        
        # Payment transitions (bigrams)
        if len(payment_hist) >= 2:
            bigrams = [payment_hist[i:i+2] for i in range(len(payment_hist)-1)]
            
            # Critical transitions
            good_to_bad = sum(1 for bg in bigrams if bg[0] == '0' and bg[1] in '123456789')
            bad_to_good = sum(1 for bg in bigrams if bg[0] in '123456789' and bg[1] == '0')
            
            features.update({
                'good_to_bad_transitions': good_to_bad,
                'bad_to_good_transitions': bad_to_good,
                'transition_ratio': good_to_bad / max(bad_to_good, 1)
            })
        else:
            features.update({
                'good_to_bad_transitions': 0,
                'bad_to_good_transitions': 0,
                'transition_ratio': 0
            })
        
        # Payment volatility
        if len(payment_hist) >= 2:
            severity_scores = [self.payment_severity.get(char, 0) for char in payment_hist]
            volatility = np.std(severity_scores)
            features['payment_volatility'] = volatility
        else:
            features['payment_volatility'] = 0
        
        return features
    
    def create_no_history_features(self, uid):
        """Create default features for accounts with no payment history"""
        
        features = {
            'uid': uid,
            'has_payment_history': False,
            'recency_weighted_score': 0,
            'recency_weighted_bad_ratio': 0,
            'payment_trend_slope': 0,
            'trend_direction': 0,
            'deterioration_velocity': 0,
            'max_consecutive_bad_payments': 0,
            'severe_delinquency_months': 0,
            'write_off_months': 0,
            'unknown_code_months': 0,
            'recent_severe_flag': 0,
            'death_spiral_risk': 0,
            'good_to_bad_transitions': 0,
            'bad_to_good_transitions': 0,
            'transition_ratio': 0,
            'payment_volatility': 0
        }
        
        # Recent behavior features
        for window in [3, 6, 12]:
            features.update({
                f'recent_{window}m_avg_severity': 0,
                f'recent_{window}m_max_severity': 0,
                f'recent_{window}m_bad_count': 0,
                f'recent_{window}m_severe_count': 0
            })
        
        return features

def main():
    # Set data directory
    data_dir = "/home/miso/Documents/WINDOWS/monsoon/senior_ds_test/data/train"
    
    # Create temporal feature engineer
    engineer = TemporalFeatureEngineer(data_dir)
    
    # Extract temporal features for training data
    temporal_df = engineer.extract_temporal_features(is_train=True)
    
    # Save temporal features
    output_path = "/home/miso/Documents/WINDOWS/monsoon/solution/final_working_scripts/temporal_features.csv"
    temporal_df.to_csv(output_path, index=False)
    print(f"\n💾 Temporal features saved: {output_path}")
    
    # Feature summary
    print(f"\n📊 TEMPORAL FEATURES SUMMARY:")
    print(f"   Total records: {len(temporal_df):,}")
    print(f"   With payment history: {temporal_df['has_payment_history'].sum():,}")
    print(f"   Without payment history: {(~temporal_df['has_payment_history']).sum():,}")
    print(f"   Total temporal features: {len(temporal_df.columns) - 1}")  # Subtract uid
    
    return temporal_df

if __name__ == "__main__":
    temporal_df = main()
    print(f"\n✅ STEP 2 COMPLETE!")
    print(f"🎯 Temporal features ready: {temporal_df.shape}")


