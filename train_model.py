"""
Complete training script for the Poetry Emotion Classification System
Updated to automatically handle multi-label emotions
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
import numpy as np
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
    """Analyze and visualize the data distribution with automatic emotion detection"""
    logger.info("Analyzing data distribution...")
    
    # Load data
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Automatically detect emotion columns
    trainer = OptimizedPoetryTrainer()
    emotion_names = trainer.detect_emotions_from_csv(csv_path)
    
    logger.info(f"Detected {len(emotion_names)} emotions: {emotion_names}")
    
    # Determine grid size for subplots
    n_emotions = len(emotion_names)
    if n_emotions <= 4:
        fig_rows, fig_cols = 2, 2
    elif n_emotions <= 6:
        fig_rows, fig_cols = 2, 3
    elif n_emotions <= 9:
        fig_rows, fig_cols = 3, 3
    else:
        fig_rows, fig_cols = 4, 4
    
    # Create visualizations
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(18, 15))
    axes = axes.ravel() if hasattr(axes, 'ravel') else [axes]
    
    # 1. Emotion distribution
    emotion_counts = []
    for emotion in emotion_names:
        if emotion in df.columns:
            emotion_counts.append(df[emotion].sum())
        else:
            emotion_counts.append(0)
    
    ax = axes[0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(emotion_names)))
    bars = ax.bar(range(len(emotion_names)), emotion_counts, color=colors)
    ax.set_title('情感分布统计', fontsize=14, fontweight='bold')
    ax.set_xlabel('情感类别')
    ax.set_ylabel('诗歌数量')
    ax.set_xticks(range(len(emotion_names)))
    ax.set_xticklabels(emotion_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, emotion_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(emotion_counts)*0.01,
                str(count), ha='center', va='bottom', fontsize=10)
    
    # 2. Multi-label distribution
    ax = axes[1]
    emotion_columns = [col for col in emotion_names if col in df.columns]
    if emotion_columns:
        multi_label_counts = df[emotion_columns].sum(axis=1).value_counts().sort_index()
        
        ax.bar(multi_label_counts.index, multi_label_counts.values, color='#4ECDC4')
        ax.set_title('多标签分布', fontsize=14, fontweight='bold')
        ax.set_xlabel('情感标签数量')
        ax.set_ylabel('诗歌数量')
        ax.set_xticks(multi_label_counts.index)
    
    # 3. Poem length distribution
    ax = axes[2]
    text_column = None
    for col in ['content', 'poem', 'text']:
        if col in df.columns:
            text_column = col
            break
    
    if text_column:
        poem_lengths = df[text_column].str.len()
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
        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # 5-8. Individual emotion percentage charts (if more than 4 emotions)
    if len(emotion_names) > 4:
        for i, emotion in enumerate(emotion_names[:4]):  # Show top 4 emotions in detail
            if i + 4 < len(axes) and emotion in df.columns:
                ax = axes[i + 4]
                emotion_data = df[emotion].value_counts()
                ax.pie(emotion_data.values, labels=[f'无{emotion}', f'有{emotion}'], 
                       autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightblue'])
                ax.set_title(f'{emotion} 分布', fontsize=12, fontweight='bold')
    
    # Hide unused subplots
    for i in range(len(emotion_names) + 4, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = os.path.join(output_dir, 'data_analysis_multilabel.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Data analysis visualization saved to {output_path}")
    
    # Print detailed statistics
    logger.info(f"\n数据集统计:")
    logger.info(f"总诗歌数量: {len(df)}")
    
    if text_column:
        logger.info(f"平均诗歌长度: {poem_lengths.mean():.1f} 字符")
        logger.info(f"最短诗歌: {poem_lengths.min()} 字符")
        logger.info(f"最长诗歌: {poem_lengths.max()} 字符")
    
    logger.info(f"\n情感分布:")
    for emotion in emotion_names:
        if emotion in df.columns:
            count = df[emotion].sum()
            percentage = (count / len(df)) * 100
            logger.info(f"  {emotion}: {count} 首 ({percentage:.1f}%)")
    
    # Multi-label statistics
    if emotion_columns:
        total_labels = df[emotion_columns].sum().sum()
        avg_labels_per_poem = total_labels / len(df)
        logger.info(f"\n多标签统计:")
        logger.info(f"  总标签数: {total_labels}")
        logger.info(f"  平均每首诗标签数: {avg_labels_per_poem:.2f}")
        
        # Most common emotion combinations
        df['emotion_combination'] = df[emotion_columns].apply(
            lambda row: '+'.join([emotion for emotion in emotion_columns if row[emotion] == 1]), 
            axis=1
        )
        top_combinations = df['emotion_combination'].value_counts().head(10)
        logger.info(f"\n最常见的情感组合:")
        for combo, count in top_combinations.items():
            if combo:  # Skip empty combinations
                logger.info(f"  {combo}: {count} 首")
    
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
        df = analyze_data_distribution(args.labeled_data)
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
            
            # Load labeled data (this will automatically detect emotions)
            texts, labels, emotion_names = trainer.prepare_labeled_data(args.labeled_data)
            
            # Fine-tune directly on labeled data
            model, test_results = trainer.fine_tune_model(
                trainer.base_model_name,  # Use base model without pretraining
                texts,
                labels,
                emotion_names,
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
                    
                    # Sort emotions by F1 score for better readability
                    emotion_results = [(emotion, metrics) for emotion, metrics in test_results.items() 
                                     if emotion != 'overall_f1']
                    emotion_results.sort(key=lambda x: x[1]['f1-score'], reverse=True)
                    
                    for emotion, metrics in emotion_results:
                        logger.info(f"{emotion}:")
                        logger.info(f"  Precision: {metrics['precision']:.4f}")
                        logger.info(f"  Recall: {metrics['recall']:.4f}")
                        logger.info(f"  F1-Score: {metrics['f1-score']:.4f}")
                        logger.info(f"  Support: {metrics['support']}")
        
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
            'expected': ['思念', '想家', '孤独']
        },
        {
            'title': '虞美人',
            'text': '春花秋月何时了，往事知多少。小楼昨夜又东风，故国不堪回首月明中。',
            'expected': ['哀伤', '思念', '愁绪']
        },
        {
            'title': '满江红',
            'text': '怒发冲冠，凭栏处、潇潇雨歇。抬望眼，仰天长啸，壮怀激烈。',
            'expected': ['愤怒', '怨恨']
        },
        {
            'title': '登科后',
            'text': '春风得意马蹄疾，一日看尽长安花。',
            'expected': ['喜悦']
        },
        {
            'title': '江雪',
            'text': '千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。',
            'expected': ['孤独', '失意']
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
        logger.info("情感分数 (>0.3):")
        
        # Show emotions with score > 0.3 for better insight
        significant_emotions = [(emotion, score) for emotion, score in results['emotions'].items() 
                              if score > 0.3]
        significant_emotions.sort(key=lambda x: x[1], reverse=True)
        
        for emotion, score in significant_emotions:
            logger.info(f"  {emotion}: {score:.3f}")
        
        logger.info("重要词汇 (前5个):")
        for token, importance in results['token_importance'][:5]:
            logger.info(f"  {token}: {importance:.3f}")
        
        # Check if any expected emotions were predicted
        expected_set = set(poem_info['expected'])
        predicted_set = set(results['predicted_emotions'])
        overlap = expected_set.intersection(predicted_set)
        
        if overlap:
            logger.info(f"✅ 正确预测的情感: {list(overlap)}")
        else:
            logger.info(f"❌ 未能预测期望的情感")
    
    # Visualize attention for the first poem
    if test_poems:
        first_poem_results = analyzer.analyze_poem(test_poems[0]['text'])
        try:
            analyzer.visualize_attention(first_poem_results, 
                save_path=f'./visualizations/attention_example_multilabel.png')
            logger.info(f"注意力可视化已保存到: ./visualizations/attention_example_multilabel.png")
        except Exception as e:
            logger.warning(f"注意力可视化失败: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='Train Poetry Emotion Classification Model with Multi-label Support'
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
        default='./data/poem_emotions_multilabel.csv',  # Updated default
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