# main.py - Updated for your exact file structure

import os
import sys
import pandas as pd

# Import from your files
from poetry_emotion_classifier import *
from visualization.attention_maps import *

def simple_setup():
    """Create necessary directories"""
    directories = [
        'results',
        'results/models', 
        'results/models/pretrained',
        'results/models/final_model',
        'results/visualizations',
        'results/reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ 目录结构已创建")

def check_data_files():
    """Check if required data files exist"""
    required_files = {
        'data/poem_emotions_consolidated.csv': '整合后的情感数据',
        'data/unlabeled_poems.txt': '无标签诗歌数据'
    }
    
    missing_files = []
    existing_files = {}
    
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            existing_files[file_path] = description
            print(f"✅ 找到文件: {file_path} ({description})")
        else:
            missing_files.append((file_path, description))
            print(f"❌ 缺少文件: {file_path} ({description})")
    
    return existing_files, missing_files

def run_quick_demo():
    """快速演示 - 不进行完整训练"""
    print("=" * 50)
    print("🚀 诗歌情感分析系统 - 快速演示")
    print("=" * 50)
    
    # Setup
    simple_setup()
    
    # Check data
    existing_files, missing_files = check_data_files()
    
    if 'data/poem_emotions_consolidated.csv' not in existing_files:
        print("\n❌ 错误: 找不到主要数据文件")
        print("请确保 data/poem_emotions_consolidated.csv 文件存在")
        return None, None
    
    try:
        # Load and analyze data
        print(f"\n📊 数据分析...")
        df = pd.read_csv('data/poem_emotions_consolidated.csv', encoding='utf-8')
        
        emotion_names = ['哀伤', '思念', '怨恨', '喜悦']
        
        print(f"📝 总诗歌数量: {len(df)}")
        print(f"🎭 情感类别: {emotion_names}")
        
        print(f"\n📈 情感分布:")
        for emotion in emotion_names:
            if emotion in df.columns:
                count = df[emotion].sum()
                percentage = (count / len(df)) * 100
                print(f"  {emotion}: {count} 首 ({percentage:.1f}%)")
        
        # Create basic visualization
        print(f"\n🎨 生成基础可视化...")
        viz = VisualizationTools()
        viz.plot_emotion_distribution(
            df, emotion_names, 
            'results/visualizations/emotion_distribution.png'
        )
        
        # Sample analysis without training
        print(f"\n🤖 模拟情感分析示例:")
        sample_poems = [
            "春花秋月何时了，往事知多少",
            "独在异乡为异客，每逢佳节倍思亲", 
            "怒发冲冠，凭栏处",
            "春风得意马蹄疾，一日看尽长安花"
        ]
        
        for i, poem in enumerate(sample_poems):
            print(f"\n--- 示例 {i+1} ---")
            print(f"诗句: {poem}")
            # This is just a demo - in real training, these would be model predictions
            print("模拟情感预测:")
            if "春花秋月" in poem:
                print("  哀伤: 0.85, 思念: 0.72")
            elif "异乡" in poem:
                print("  思念: 0.91, 哀伤: 0.43")
            elif "怒发冲冠" in poem:
                print("  怨恨: 0.88, 愤怒: 0.76")
            else:
                print("  喜悦: 0.82, 思念: 0.24")
        
        print(f"\n✅ 快速演示完成!")
        print("运行 'python main.py train' 进行完整模型训练")
        
        return df, emotion_names
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {str(e)}")
        return None, None

def run_full_training():
    """完整训练流程"""
    print("=" * 50)
    print("🚀 诗歌情感分析系统 - 完整训练")
    print("=" * 50)
    
    # Setup
    simple_setup()
    
    # Check data
    existing_files, missing_files = check_data_files()
    
    if 'data/poem_emotions_consolidated.csv' not in existing_files:
        print("\n❌ 错误: 找不到主要数据文件")
        return None, None
    
    try:
        # Load unlabeled data for pretraining
        unlabeled_texts = []
        if 'data/unlabeled_poems.txt' in existing_files:
            print(f"\n📚 加载无标签数据...")
            with open('data/unlabeled_poems.txt', 'r', encoding='utf-8') as f:
                unlabeled_texts = [line.strip() for line in f if line.strip()]
            print(f"✅ 加载了 {len(unlabeled_texts)} 首无标签诗歌")
        else:
            print(f"\n⚠️  未找到无标签数据，跳过预训练步骤")
        
        # Step 1: Optional Pretraining
        pretrained_model_path = 'bert-base-chinese'  # Default
        
        if len(unlabeled_texts) > 100:
            print(f"\n🔧 步骤1: 预训练准备...")
            try:
                pretrainer = PoetryPretrainer()
                # Use simplified pretraining approach
                pretrained_model_path = pretrainer.pretrain_model(
                    unlabeled_texts[:500],  # Use subset for speed
                    output_dir='results/models/pretrained',
                    epochs=2
                )
                print(f"✅ 预训练准备完成，使用模型: {pretrained_model_path}")
            except Exception as e:
                print(f"⚠️  预训练失败: {e}")
                print("将使用默认BERT模型")
                pretrained_model_path = 'bert-base-chinese'
        else:
            print(f"\n🔧 步骤1: 使用预训练BERT模型...")
            pretrained_model_path = 'bert-base-chinese'
        
        # Step 2: Load labeled data
        print(f"\n📊 步骤2: 加载标注数据...")
        trainer = PoetryEmotionTrainer(model_name=pretrained_model_path)
        texts, labels = trainer.load_data('data/poem_emotions_consolidated.csv')
        print(f"✅ 加载了 {len(texts)} 首标注诗歌")
        
        # Step 3: Prepare datasets
        print(f"\n🔄 步骤3: 准备训练数据...")
        train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(texts, labels)
        
        # Step 4: Train model (Use Simple Trainer directly to avoid dependency issues)
        print(f"\n🎯 步骤4: 训练模型...")
        print("⏳ 这可能需要几分钟...")
        
        try:
            # Use the simple trainer (more reliable)
            print("🔄 使用简化训练器...")
            simple_trainer = SimplePoetryTrainer(
                model_name=pretrained_model_path,
                emotion_names=trainer.emotion_names
            )
            
            trained_model = simple_trainer.train_model(
                train_dataset,
                val_dataset,
                output_dir='results/models/final_model',
                epochs=3,  # Good balance of speed and quality
                batch_size=4,  # Conservative for memory
                learning_rate=2e-5
            )
            
            # Update trainer reference for later use
            trainer.model = simple_trainer.model
            trainer.tokenizer = simple_trainer.tokenizer
            
            print(f"✅ 模型训练完成")
            
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            print("跳过训练步骤，使用演示模式...")
            return None, None
        
        # Step 5: Evaluate
        print(f"\n📈 步骤5: 评估模型...")
        predictions, true_labels = trainer.evaluate_model(test_dataset)
        
        # Step 6: Generate visualizations
        print(f"\n🎨 步骤6: 生成可视化...")
        viz = VisualizationTools()
        
        # Load data for visualization
        df = pd.read_csv('data/poem_emotions_consolidated.csv', encoding='utf-8')
        
        # Create visualizations
        viz.plot_emotion_distribution(
            df, trainer.emotion_names, 
            'results/visualizations/emotion_distribution.png'
        )
        
        viz.plot_confusion_matrix(
            true_labels, predictions, trainer.emotion_names,
            'results/visualizations/confusion_matrices.png'
        )
        
        print(f"✅ 基础可视化完成")
        
        # Step 7: Model interpretation
        print(f"\n🔍 步骤7: 模型解释...")
        interpreter = PoetryInterpreter(trainer.model, trainer.tokenizer, trainer.emotion_names)
        
        # Test samples
        sample_poems = [
            "春花秋月何时了，往事知多少",
            "独在异乡为异客，每逢佳节倍思亲", 
            "怒发冲冠，凭栏处",
            "春风得意马蹄疾，一日看尽长安花"
        ]
        
        print(f"\n🎭 情感分析示例:")
        for i, poem in enumerate(sample_poems):
            print(f"\n--- 示例 {i+1} ---")
            print(f"诗句: {poem}")
            
            try:
                explanation = interpreter.explain_prediction(poem)
                
                print("情感预测:")
                for emotion, score in explanation['predictions'].items():
                    if score > 0.3:  # Only show confident predictions
                        print(f"  {emotion}: {score:.3f}")
                
                print("关键词分析:")
                for emotion, word_importance in explanation['word_importance'].items():
                    if word_importance:
                        top_words = word_importance[:3]
                        print(f"  {emotion}: {[f'{word}({score:.3f})' for word, score in top_words]}")
                        
            except Exception as e:
                print(f"  ⚠️ 分析失败: {e}")
        
        # Step 8: Create teaching materials
        print(f"\n🎓 步骤8: 生成教学材料...")
        
        try:
            attention_viz = AttentionVisualizer(trainer.model, trainer.tokenizer, trainer.emotion_names)
            
            # Use first sample for demonstration
            sample_poem = sample_poems[0]
            sample_explanation = interpreter.explain_prediction(sample_poem)
            
            # Create attention heatmap
            print("  生成注意力热力图...")
            attention_viz.create_attention_heatmap(
                sample_poem,
                save_path='results/visualizations/attention_heatmap_demo.png'
            )
            
            # Create emotion radar
            print("  生成情感雷达图...")
            attention_viz.create_interactive_emotion_radar(
                sample_explanation['predictions'],
                sample_poem,
                save_path='results/visualizations/emotion_radar_demo.html'
            )
            
            print(f"✅ 教学材料生成完成")
            
        except Exception as e:
            print(f"⚠️ 教学材料生成部分失败: {e}")
        
        # Final summary
        print(f"\n" + "=" * 50)
        print("🎉 系统训练完成!")
        print("=" * 50)
        print("📁 结果文件位置:")
        print("  🤖 训练模型: results/models/final_model/")
        print("  📊 可视化图表: results/visualizations/")
        print("  📈 情感分布图: results/visualizations/emotion_distribution.png")
        print("  🔥 注意力热力图: results/visualizations/attention_heatmap_demo.png")
        print("  🎯 情感雷达图: results/visualizations/emotion_radar_demo.html")
        print("=" * 50)
        
        return trainer, interpreter
        
    except Exception as e:
        print(f"\n❌ 训练过程中出错: {str(e)}")
        print("请检查数据文件和依赖是否正确安装")
        return None, None

def main():
    """主函数"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'demo':
            run_quick_demo()
        elif command == 'train':
            run_full_training()
        else:
            print("用法:")
            print("  python main.py demo    # 快速演示")  
            print("  python main.py train   # 完整训练")
    else:
        print("🚀 诗歌情感分析系统")
        print("\n选择运行模式:")
        print("1. 快速演示 (不训练模型)")
        print("2. 完整训练 (训练并保存模型)")
        
        choice = input("\n请输入选择 (1 或 2): ").strip()
        
        if choice == '1':
            run_quick_demo()
        elif choice == '2':
            run_full_training()
        else:
            print("无效选择，运行快速演示...")
            run_quick_demo()

if __name__ == "__main__":
    main()