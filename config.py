# config.py - Configuration settings for your poetry emotion system

class Config:
    """Configuration settings"""
    
    # Model Configuration
    MODEL_NAME = 'bert-base-chinese'          # Base model for Chinese text
    MAX_LENGTH = 512                          # Maximum sequence length
    NUM_EMOTIONS = 4                          # Number of emotion categories
    EMOTION_NAMES = ['哀伤', '思念', '怨恨', '喜悦']  # Emotion labels
    
    # Training Configuration
    BATCH_SIZE = 8                            # Batch size (conservative for memory)
    LEARNING_RATE = 2e-5                      # Learning rate
    NUM_EPOCHS = 5                            # Number of training epochs
    WARMUP_STEPS = 100                        # Warmup steps for learning rate
    WEIGHT_DECAY = 0.01                       # Weight decay for regularization
    
    # Data Configuration
    TEST_SIZE = 0.2                           # Test set proportion
    VAL_SIZE = 0.1                            # Validation set proportion
    RANDOM_STATE = 42                         # Random seed for reproducibility
    
    # File Paths
    DATA_DIR = './data'                       # Data directory
    MODEL_DIR = './results/models'            # Model save directory
    VIZ_DIR = './results/visualizations'      # Visualization save directory
    PRETRAINED_DIR = './results/models/pretrained'     # Pretrained model directory
    FINAL_MODEL_DIR = './results/models/final_model'   # Final model directory
    
    # Data Files
    CONSOLIDATED_CSV = 'data/poem_emotions_consolidated.csv'    # Main labeled dataset
    MULTILABEL_CSV = 'data/poem_emotions_multilabel.csv'       # Original dataset
    UNLABELED_TXT = 'data/unlabeled_poems.txt'                 # Unlabeled poems
    
    # Visualization Configuration
    FIGURE_SIZE = (12, 8)                     # Default figure size
    DPI = 300                                 # Image resolution
    FONT_SIZE = 12                            # Default font size
    
    # Chinese Font Configuration (for matplotlib)
    CHINESE_FONT = 'SimHei'                   # Chinese font name
    
    # Model Interpretation
    TOP_K_WORDS = 10                          # Top-K important words to show
    ATTENTION_THRESHOLD = 0.3                 # Minimum attention score to consider
    
    # Pretraining Configuration
    MASK_PROBABILITY = 0.15                   # Probability for masking tokens
    PRETRAIN_EPOCHS = 2                       # Pretraining epochs
    MAX_UNLABELED_SAMPLES = 1000              # Maximum unlabeled samples to use
    
    # Teaching/Demo Configuration
    SAMPLE_POEMS = [                          # Sample poems for demonstration
        "春花秋月何时了，往事知多少",
        "独在异乡为异客，每逢佳节倍思亲", 
        "怒发冲冠，凭栏处",
        "春风得意马蹄疾，一日看尽长安花"
    ]
    
    # Output Files
    EMOTION_DIST_PLOT = 'emotion_distribution.png'
    CONFUSION_MATRIX_PLOT = 'confusion_matrices.png'
    ATTENTION_HEATMAP = 'attention_heatmap_demo.png'
    EMOTION_RADAR = 'emotion_radar_demo.html'
    
    @classmethod
    def get_model_config(cls):
        """Get model-related configuration"""
        return {
            'model_name': cls.MODEL_NAME,
            'max_length': cls.MAX_LENGTH,
            'num_emotions': cls.NUM_EMOTIONS,
            'emotion_names': cls.EMOTION_NAMES
        }
    
    @classmethod
    def get_training_config(cls):
        """Get training-related configuration"""
        return {
            'batch_size': cls.BATCH_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'num_epochs': cls.NUM_EPOCHS,
            'warmup_steps': cls.WARMUP_STEPS,
            'weight_decay': cls.WEIGHT_DECAY
        }
    
    @classmethod
    def get_data_config(cls):
        """Get data-related configuration"""
        return {
            'test_size': cls.TEST_SIZE,
            'val_size': cls.VAL_SIZE,
            'random_state': cls.RANDOM_STATE,
            'data_dir': cls.DATA_DIR
        }
    
    @classmethod
    def get_paths(cls):
        """Get all file paths"""
        return {
            'data_dir': cls.DATA_DIR,
            'model_dir': cls.MODEL_DIR,
            'viz_dir': cls.VIZ_DIR,
            'consolidated_csv': cls.CONSOLIDATED_CSV,
            'unlabeled_txt': cls.UNLABELED_TXT
        }

# You can also create different configurations for different scenarios
class QuickDemoConfig(Config):
    """Configuration for quick demo (faster, less resource-intensive)"""
    BATCH_SIZE = 4
    NUM_EPOCHS = 2
    MAX_UNLABELED_SAMPLES = 100

class ProductionConfig(Config):
    """Configuration for production use (higher quality)"""
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    MAX_UNLABELED_SAMPLES = 5000
    LEARNING_RATE = 1e-5

# Easy access to current config
current_config = Config