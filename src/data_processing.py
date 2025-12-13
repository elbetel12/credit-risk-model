import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineer additional features with error handling"""
    
    def __init__(self):
        self.feature_columns = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Transform data with error handling"""
        try:
            X = X.copy()
            logger.info(f"Starting feature engineering on {len(X)} rows")
            
            # Check required columns
            required_cols = ['TransactionStartTime', 'Amount']
            missing_cols = [col for col in required_cols if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Time-based features with error handling
            try:
                X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'], errors='coerce')
                X['Transaction_Hour'] = X['TransactionStartTime'].dt.hour
                X['Transaction_Day'] = X['TransactionStartTime'].dt.day
                X['Transaction_Month'] = X['TransactionStartTime'].dt.month
                X['Transaction_Year'] = X['TransactionStartTime'].dt.year
                X['Transaction_DayOfWeek'] = X['TransactionStartTime'].dt.dayofweek
            except Exception as e:
                logger.warning(f"Error processing datetime features: {e}")
                # Create placeholder columns if datetime fails
                X['Transaction_Hour'] = 0
                X['Transaction_Day'] = 1
                X['Transaction_Month'] = 1
                X['Transaction_Year'] = 2023
            
            # Transaction patterns
            X['Is_Weekend'] = X['Transaction_DayOfWeek'].isin([5, 6]).astype(int)
            X['Is_BusinessHours'] = ((X['Transaction_Hour'] >= 9) & 
                                    (X['Transaction_Hour'] <= 17)).astype(int)
            
            # Amount-based features
            X['Amount_Abs'] = abs(X['Amount'])
            X['Is_Credit'] = (X['Amount'] < 0).astype(int)
            
            logger.info(f"Feature engineering completed. Output shape: {X.shape}")
            self.feature_columns = X.columns.tolist()
            return X
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise

def load_and_process_data(filepath: str) -> Tuple[pd.DataFrame, list, Pipeline]:
    """Load and process data with comprehensive error handling"""
    try:
        logger.info(f"Loading data from {filepath}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Load data with error handling
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded {len(df)} rows from {filepath}")
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            # Try alternative formats
            try:
                df = pd.read_parquet(filepath.replace('.csv', '.parquet'))
                logger.info("Loaded from parquet file instead")
            except:
                raise
        
        # Check data quality
        if df.empty:
            logger.warning("Loaded dataframe is empty")
            return df, [], None
        
        # Initialize pipeline
        pipeline = create_feature_pipeline()
        
        # Fit and transform
        logger.info("Starting pipeline transformation")
        processed_features = pipeline.fit_transform(df)
        
        # Get feature names
        feature_names = []
        numerical_features = ['Amount', 'Value', 'Transaction_Hour', 
                            'Transaction_Day', 'Transaction_Month']
        feature_names.extend(numerical_features)
        
        logger.info(f"Processing completed. Features shape: {processed_features.shape}")
        return processed_features, feature_names, pipeline
        
    except Exception as e:
        logger.error(f"Data processing pipeline failed: {e}")
        raise