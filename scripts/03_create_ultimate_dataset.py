#!/usr/bin/env python3
"""
Step 3: Create Ultimate Dataset
==============================

This script combines comprehensive features with temporal features to create
the ultimate predictive dataset that achieved 99.09% AUC performance.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class UltimateDatasetCreator:
    def __init__(self):
        self.base_path = "/home/miso/Documents/WINDOWS/monsoon/solution/final_working_scripts"
    
    def create_ultimate_dataset(self):
        """Create the ultimate dataset by combining all feature groups"""
        
        print("🚀 CREATING ULTIMATE DATASET")
        print("="*50)
        
        # Load comprehensive features (enquiry + account aggregations)
        print("📂 Loading comprehensive features...")
        comprehensive_df = pd.read_csv(f"{self.base_path}/comprehensive_train_dataset.csv")
        print(f"Comprehensive features: {comprehensive_df.shape}")
        
        # Load temporal features
        print("📂 Loading temporal features...")
        temporal_df = pd.read_csv(f"{self.base_path}/temporal_features.csv")
        print(f"Temporal features: {temporal_df.shape}")
        
        # Select key temporal features (avoid redundancy)
        temporal_features = [
            'recency_weighted_score', 'recency_weighted_bad_ratio',
            'recent_3m_avg_severity', 'recent_3m_max_severity', 'recent_3m_bad_count',
            'recent_6m_avg_severity', 'recent_6m_max_severity', 'recent_6m_bad_count',
            'recent_12m_avg_severity', 'recent_12m_max_severity', 'recent_12m_bad_count',
            'payment_trend_slope', 'trend_direction', 'deterioration_velocity',
            'max_consecutive_bad_payments', 'death_spiral_risk', 'recent_severe_flag',
            'payment_volatility', 'good_to_bad_transitions', 'bad_to_good_transitions'
        ]
        
        # Create temporal subset
        temporal_subset = temporal_df[['uid'] + temporal_features].copy()
        print(f"Selected temporal features: {len(temporal_features)}")
        
        # Merge datasets
        print("🔄 Merging comprehensive + temporal features...")
        ultimate_df = comprehensive_df.merge(temporal_subset, on='uid', how='left')
        
        # Fill missing temporal features with 0 (for UIDs not in temporal data)
        ultimate_df[temporal_features] = ultimate_df[temporal_features].fillna(0)
        
        print(f"✅ Ultimate dataset created: {ultimate_df.shape}")
        
        # Feature group analysis
        self.analyze_feature_groups(ultimate_df)
        
        # Data quality checks
        self.data_quality_checks(ultimate_df)
        
        return ultimate_df
    
    def analyze_feature_groups(self, df):
        """Analyze the different feature groups in the ultimate dataset"""
        
        print(f"\n📊 FEATURE GROUPS IN ULTIMATE DATASET:")
        
        all_cols = [col for col in df.columns if col not in ['uid', 'NAME_CONTRACT_TYPE', 'TARGET']]
        
        # Enquiry features
        enquiry_cols = [col for col in all_cols if 'enq_' in col]
        print(f"  Enquiry features: {len(enquiry_cols)}")
        
        # Account features
        account_cols = [col for col in all_cols if 'acc_' in col]
        print(f"  Account features: {len(account_cols)}")
        
        # Temporal features
        temporal_cols = [col for col in all_cols if any(x in col for x in 
                        ['recent_', 'recency_weighted', 'trend_', 'deterioration', 'death_spiral', 'payment_volatility'])]
        print(f"  Temporal features: {len(temporal_cols)}")
        
        # Risk/derived features
        risk_cols = [col for col in all_cols if any(x in col for x in 
                    ['risk', 'missing_', 'intensity', 'has_', 'high_'])]
        print(f"  Risk/derived features: {len(risk_cols)}")
        
        print(f"  TOTAL FEATURES: {len(all_cols)}")
    
    def data_quality_checks(self, df):
        """Perform data quality checks on the ultimate dataset"""
        
        print(f"\n🔍 DATA QUALITY CHECKS:")
        
        # Target distribution
        if 'TARGET' in df.columns:
            target_dist = df['TARGET'].value_counts()
            print(f"  Target distribution:")
            print(f"    No Default (0): {target_dist[0]:,} ({target_dist[0]/len(df)*100:.1f}%)")
            print(f"    Default (1): {target_dist[1]:,} ({target_dist[1]/len(df)*100:.1f}%)")
        
        # Missing values check
        missing_counts = df.isnull().sum()
        missing_features = missing_counts[missing_counts > 0]
        
        if len(missing_features) > 0:
            print(f"  Features with missing values: {len(missing_features)}")
            for feature, count in missing_features.head(5).items():
                pct = count / len(df) * 100
                print(f"    {feature}: {count:,} ({pct:.1f}%)")
        else:
            print(f"  ✅ No missing values found")
        
        # Coverage analysis
        if 'acc_has_account_data' in df.columns:
            acc_coverage = df['acc_has_account_data'].sum()
            print(f"  Account data coverage: {acc_coverage:,} / {len(df):,} ({acc_coverage/len(df)*100:.1f}%)")
        
        if 'enq_has_enquiry_data' in df.columns:
            enq_coverage = df['enq_has_enquiry_data'].sum()
            print(f"  Enquiry data coverage: {enq_coverage:,} / {len(df):,} ({enq_coverage/len(df)*100:.1f}%)")
        
        # Temporal feature coverage
        if 'has_payment_history' in df.columns:
            temp_coverage = df['has_payment_history'].sum()
            print(f"  Payment history coverage: {temp_coverage:,} / {len(df):,} ({temp_coverage/len(df)*100:.1f}%)")
    
    def add_advanced_derived_features(self, df):
        """Add advanced derived features for extra predictive power"""
        
        print("🔄 Adding advanced derived features...")
        
        # Interaction features between enquiry and account data
        if 'enq_enquiry_amt_mean' in df.columns and 'acc_loan_amount_mean' in df.columns:
            df['enquiry_to_loan_ratio'] = df['enq_enquiry_amt_mean'] / (df['acc_loan_amount_mean'] + 1)
        
        # Risk composite scores
        risk_components = []
        
        if 'recent_6m_avg_severity' in df.columns:
            risk_components.append(df['recent_6m_avg_severity'])
        
        if 'acc_payment_bad_ratio_mean' in df.columns:
            risk_components.append(df['acc_payment_bad_ratio_mean'])
        
        if 'death_spiral_risk' in df.columns:
            risk_components.append(df['death_spiral_risk'])
        
        if risk_components:
            df['composite_risk_score'] = np.mean(risk_components, axis=0)
        
        # High-value customer indicators
        if 'acc_loan_amount_sum' in df.columns and 'enq_enquiry_amt_sum' in df.columns:
            df['total_financial_exposure'] = df['acc_loan_amount_sum'] + df['enq_enquiry_amt_sum']
        
        return df

def main():
    # Create ultimate dataset creator
    creator = UltimateDatasetCreator()
    
    # Create ultimate dataset
    ultimate_df = creator.create_ultimate_dataset()
    
    # Add advanced derived features
    ultimate_df = creator.add_advanced_derived_features(ultimate_df)
    
    # Save the ultimate dataset
    output_path = "/home/miso/Documents/WINDOWS/monsoon/solution/final_working_scripts/ultimate_dataset.csv"
    ultimate_df.to_csv(output_path, index=False)
    print(f"\n💾 Ultimate dataset saved: {output_path}")
    
    # Final summary
    print(f"\n✅ ULTIMATE DATASET SUMMARY:")
    print(f"   Shape: {ultimate_df.shape}")
    print(f"   Features: {ultimate_df.shape[1] - 3}")  # Subtract uid, contract_type, target
    print(f"   Ready for model training!")
    
    return ultimate_df

if __name__ == "__main__":
    ultimate_df = main()
    print(f"\n✅ STEP 3 COMPLETE!")
    print(f"🎯 Ultimate dataset ready: {ultimate_df.shape}")


