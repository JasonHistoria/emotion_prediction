"""
Complete training script for the Poetry Emotion Classification System
This script handles the entire pipeline from data loading to model deployment
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from poetry_classifier_optimized import (
    OptimizedPoetryTrainer, 
    PoetryEmotionAnalyzer,
    train_complete_system
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def create_directory_structure():
    """Create necessary directories for the project"""
    directories = [
        'data',
        'models',
        'models/pretrained',
        'models/final',
        'visualizations',
        'logs',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def analyze_data_distribution(csv_path: str, output_dir: str = './visualizations'):
    """Analyze and visualize the data distribution"""
    logger.info("Analyzing data distribution...")
    
    # Load data
    df = pd.read_csv(csv_path, encoding='utf-8')
    emotion_names = ['哀伤', '思念', '怨恨', '喜悦']
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    # 1. Emotion distribution
    emotion_counts = []
    for emotion in emotion_names:
        if emotion in df.columns:
            emotion_counts.append(df[emotion].sum())
    
    ax = axes[0]
    bars = ax.bar(emotion_names, emotion_counts, color=['#667eea', '#f093fb', '#fa709a', '#30cfd0'])
    ax.set_title('情感分布统计', fontsize=14, fontweight='bold')
    ax.set_xlabel('情感类别')
    ax.set_ylabel('诗歌数量')
    
    # Add value labels on bars
    for bar, count in zip(bars, emotion_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom')
    
    # 2. Multi-label distribution
    ax = axes[1]
    emotion_columns = [col for col in emotion_names if col in df.columns]
    multi_label_counts = df[emotion_columns].sum(axis=1).value_counts().sort_index()
    
    ax.bar(multi_label_counts.index, multi_label_counts.values, color='#4ECDC4')
    ax.set_title('多标签分布', fontsize=14, fontweight='bold')
    ax.set_xlabel('情感标签数量')
    ax.set_ylabel('诗歌数量')
    ax.set_xticks(multi_label_counts.index)
    
    # 3. Poem length distribution
    ax = axes[2]
    poem_lengths = df['content'].str.len()
    ax.hist(poem_lengths, bins=30, color='#96CEB4', edgecolor='black', alpha=0.7)
    ax.set_title('诗歌长度分布', fontsize=14, fontweight='bold')
    ax.set_xlabel('字符数')
    ax.set_ylabel('诗歌数量')
    ax.axvline(poem_lengths.mean(), color='red', linestyle='--', 
               label=f'平均长度: {poem_lengths.mean():.1f}')
    ax.legend()
    
    # 4. Emotion co-occurrence heatmap
    ax = axes[3]
    if len(emotion_columns) > 1:
        cooccurrence = df[emotion_columns].T.dot(df[emotion_columns])
        sns.heatmap(cooccurrence, annot=True, fmt='d', cmap='YlOrRd', 
                    xticklabels=emotion_columns, yticklabels=emotion_columns, ax=ax)
        ax.set_title('情感共现矩阵', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = os.path.join(output_dir, 'data_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Data analysis visualization saved to {output_path}")
    
    # Print statistics
    logger.info(f"\n数据集统计:")
    logger.info(f"总诗歌数量: {len(df)}")
    logger.info(f"平均诗歌长度: {poem_lengths.mean():.1f} 字符")
    logger.info(f"最短诗歌: {poem_lengths.min()} 字符")
    logger.info(f"最长诗歌: {poem_lengths.max()} 字符")
    
    for emotion in emotion_names:
        if emotion in df.columns:
            count = df[emotion].sum()
            percentage = (count / len(df)) * 100
            logger.info(f"{emotion}: {count} 首 ({percentage:.1f}%)")
    
    plt.close()
    return df


def train_and_evaluate(args):
    """Main training and evaluation function"""
    logger.info("Starting Poetry Emotion Classification System Training")
    logger.info(f"Configuration: {args}")
    
    # Create directory structure
    create_directory_structure()
    
    # Analyze data distribution
    if os.path.exists(args.labeled_data):
        analyze_data_distribution(args.labeled_data)
    else:
        logger.error(f"Labeled data not found: {args.labeled_data}")
        return
    
    # Check for unlabeled data
    if not os.path.exists(args.unlabeled_data):
        logger.warning(f"Unlabeled data not found: {args.unlabeled_data}")
        logger.warning("Proceeding without pretraining...")
        args.skip_pretrain = True
    
    # Train the model
    try:
        if args.skip_pretrain:
            logger.info("Skipping pretraining phase...")
            trainer = OptimizedPoetryTrainer()
            
            # Load labeled data
            texts, labels = trainer.prepare_labeled_data(args.labeled_data)
            
            # Fine-tune directly on labeled data
            model, test_results = trainer.fine_tune_model(
                trainer.base_model_name,  # Use base model without pretraining
                texts,
                labels,
                output_dir=args.output_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            
            final_model_dir = args.output_dir
        else:
            # Full training pipeline with pretraining
            # Ensure we use the models parent directory
            models_dir = os.path.dirname(args.output_dir)
            final_model_dir = train_complete_system(
                args.unlabeled_data,
                args.labeled_data,
                output_dir=models_dir
            )
        
        logger.info("Training completed successfully!")
        
        # Print detailed results
        logger.info("\n" + "="*60)
        logger.info("TRAINING RESULTS")
        logger.info("="*60)
        
        # Load config for results
        config_path = os.path.join(final_model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if 'test_results' in config:
                    test_results = config['test_results']
                    logger.info(f"Overall Test F1 Score: {test_results['overall_f1']:.4f}")
                    logger.info("\nPer-Emotion Performance:")
                    for emotion in ['哀伤', '思念', '怨恨', '喜悦']:
                        if emotion in test_results:
                            metrics = test_results[emotion]
                            logger.info(f"{emotion}:")
                            logger.info(f"  Precision: {metrics['precision']:.4f}")
                            logger.info(f"  Recall: {metrics['recall']:.4f}")
                            logger.info(f"  F1-Score: {metrics['f1-score']:.4f}")
        
        # Test the trained model
        if args.test_examples:
            logger.info("\nTesting trained model with examples...")
            test_model_with_examples(final_model_dir)
            
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise


def test_model_with_examples(model_dir: str):
    """Test the trained model with example poems"""
    analyzer = PoetryEmotionAnalyzer(model_dir)
    
    test_poems = [
        {
            'title': '九月九日忆山东兄弟',
            'text': '独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。',
            'expected': ['思念']
        },
        {
            'title': '虞美人',
            'text': '春花秋月何时了，往事知多少。小楼昨夜又东风，故国不堪回首月明中。',
            'expected': ['哀伤', '思念']
        },
        {
            'title': '满江红',
            'text': '怒发冲冠，凭栏处、潇潇雨歇。抬望眼，仰天长啸，壮怀激烈。',
            'expected': ['怨恨']
        },
        {
            'title': '登科后',
            'text': '春风得意马蹄疾，一日看尽长安花。',
            'expected': ['喜悦']
        }
    ]
    
    logger.info("\n" + "=" * 60)
    logger.info("模型测试结果")
    logger.info("=" * 60)
    
    for poem_info in test_poems:
        results = analyzer.analyze_poem(poem_info['text'])
        
        logger.info(f"\n诗歌: 《{poem_info['title']}》")
        logger.info(f"内容: {poem_info['text']}")
        logger.info(f"期望情感: {poem_info['expected']}")
        logger.info(f"预测情感: {results['predicted_emotions']}")
        logger.info("情感分数:")
        for emotion, score in results['emotions'].items():
            logger.info(f"  {emotion}: {score:.3f}")
        
        logger.info("重要词汇 (前5个):")
        for token, importance in results['token_importance'][:5]:
            logger.info(f"  {token}: {importance:.3f}")
        
        # Visualize attention for the first poem
        if poem_info == test_poems[0]:
            analyzer.visualize_attention(results, 
                save_path=f'./visualizations/attention_example.png')


def main():
    parser = argparse.ArgumentParser(
        description='Train Poetry Emotion Classification Model'
    )
    
    parser.add_argument(
        '--unlabeled-data',
        type=str,
        default='./data/unlabeled_poems.txt',
        help='Path to unlabeled poems for pretraining'
    )
    
    parser.add_argument(
        '--labeled-data',
        type=str,
        default='./data/poem_emotions_consolidated.csv',
        help='Path to labeled poems CSV file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./models/final',
        help='Directory to save the trained model'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--skip-pretrain',
        action='store_true',
        help='Skip pretraining phase'
    )
    
    parser.add_argument(
        '--test-examples',
        action='store_true',
        default=True,
        help='Test model with example poems after training'
    )
    
    args = parser.parse_args()
    
    try:
        train_and_evaluate(args)
        logger.info("\n✅ Training completed successfully!")
        logger.info("You can now run the web server with: python poetry_web_server.py")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()