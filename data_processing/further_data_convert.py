import pandas as pd

def consolidate_emotions(input_csv_path: str, output_csv_path: str = None):
    """
    Consolidate similar emotions into 4 main categories:
    - 哀伤: {哀伤, 哭泣, 流泪, 失意}
    - 思念: {孤独, 思念, 愁绪, 想家}
    - 怨恨: {怨恨, 恐惧, 愤怒}
    - 喜悦: {喜悦} (keeps the same)
    
    Args:
        input_csv_path (str): Path to the input CSV file
        output_csv_path (str, optional): Path to save the consolidated CSV
    
    Returns:
        pandas.DataFrame: Consolidated dataset
    """
    
    # Define emotion mapping
    emotion_mapping = {
        # 哀伤 group
        '哀伤': '哀伤',
        '哭泣': '哀伤',
        '流泪': '哀伤',
        '失意': '哀伤',
        
        # 思念 group
        '孤独': '思念',
        '思念': '思念',
        '愁绪': '思念',
        '想家': '思念',
        
        # 怨恨 group
        '怨恨': '怨恨',
        '恐惧': '怨恨',
        '愤怒': '怨恨',
        
        # 喜悦 group (unchanged)
        '喜悦': '喜悦',
        
        # Handle potential other emotions (惊讶 not mentioned in your grouping)
        '惊讶': '惊讶'  # Keep as is for now, or you can assign to a group
    }
    
    # Read the original CSV
    df = pd.read_csv(input_csv_path, encoding='utf-8')
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Get content column
    content_column = df['content']
    
    # Get emotion columns (all except 'content')
    emotion_columns = [col for col in df.columns if col != 'content']
    
    print(f"Original emotions found: {emotion_columns}")
    
    # Create new consolidated emotion columns
    consolidated_emotions = ['哀伤', '思念', '怨恨', '喜悦']
    
    # Initialize the new dataset with content
    new_data = {'content': content_column}
    
    # Initialize consolidated emotion columns with zeros
    for emotion in consolidated_emotions:
        new_data[emotion] = [0] * len(df)
    
    # Process each original emotion column
    for original_emotion in emotion_columns:
        if original_emotion in emotion_mapping:
            target_emotion = emotion_mapping[original_emotion]
            
            # If target emotion is in our consolidated list
            if target_emotion in consolidated_emotions:
                # For each row, if original emotion is 1, set target emotion to 1
                for i in range(len(df)):
                    if df.iloc[i][original_emotion] == 1:
                        new_data[target_emotion][i] = 1
                        
                print(f"Mapped '{original_emotion}' -> '{target_emotion}'")
            else:
                print(f"Warning: '{original_emotion}' mapped to '{target_emotion}' which is not in consolidated list")
        else:
            print(f"Warning: Emotion '{original_emotion}' not found in mapping, skipping")
    
    # Create new DataFrame
    consolidated_df = pd.DataFrame(new_data)
    
    # Reorder columns: content first, then emotions
    column_order = ['content'] + consolidated_emotions
    consolidated_df = consolidated_df[column_order]
    
    print(f"\nConsolidated dataset shape: {consolidated_df.shape}")
    print(f"New emotion columns: {consolidated_emotions}")
    
    # Show statistics
    print("\nEmotion distribution in consolidated dataset:")
    for emotion in consolidated_emotions:
        count = consolidated_df[emotion].sum()
        percentage = (count / len(consolidated_df)) * 100
        print(f"  {emotion}: {count} poems ({percentage:.1f}%)")
    
    # Save if output path provided
    if output_csv_path:
        consolidated_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\nConsolidated dataset saved to: {output_csv_path}")
    
    return consolidated_df

def analyze_consolidation_impact(original_df, consolidated_df):
    """
    Analyze the impact of emotion consolidation
    """
    print("\n" + "="*60)
    print("CONSOLIDATION ANALYSIS")
    print("="*60)
    
    # Original emotion counts
    original_emotions = [col for col in original_df.columns if col != 'content']
    consolidated_emotions = [col for col in consolidated_df.columns if col != 'content']
    
    print(f"Original emotions ({len(original_emotions)}): {original_emotions}")
    print(f"Consolidated emotions ({len(consolidated_emotions)}): {consolidated_emotions}")
    
    print("\nOriginal emotion distribution:")
    for emotion in original_emotions:
        count = original_df[emotion].sum()
        percentage = (count / len(original_df)) * 100
        print(f"  {emotion}: {count} poems ({percentage:.1f}%)")
    
    print("\nConsolidated emotion distribution:")
    for emotion in consolidated_emotions:
        count = consolidated_df[emotion].sum()
        percentage = (count / len(consolidated_df)) * 100
        print(f"  {emotion}: {count} poems ({percentage:.1f}%)")
    
    # Check multi-label distribution
    print(f"\nMulti-label analysis:")
    
    # Original multi-label count
    original_multi_count = 0
    for i in range(len(original_df)):
        emotion_count = sum(original_df.iloc[i][col] for col in original_emotions)
        if emotion_count > 1:
            original_multi_count += 1
    
    # Consolidated multi-label count
    consolidated_multi_count = 0
    for i in range(len(consolidated_df)):
        emotion_count = sum(consolidated_df.iloc[i][col] for col in consolidated_emotions)
        if emotion_count > 1:
            consolidated_multi_count += 1
    
    print(f"  Original: {original_multi_count} poems with multiple emotions ({(original_multi_count/len(original_df)*100):.1f}%)")
    print(f"  Consolidated: {consolidated_multi_count} poems with multiple emotions ({(consolidated_multi_count/len(consolidated_df)*100):.1f}%)")

def preview_consolidation(df, num_examples=3):
    """
    Preview some examples from the consolidated dataset
    """
    print(f"\n" + "="*60)
    print("PREVIEW OF CONSOLIDATED DATASET")
    print("="*60)
    
    emotion_cols = [col for col in df.columns if col != 'content']
    
    for i in range(min(num_examples, len(df))):
        print(f"\nExample {i+1}:")
        print(f"Content: {df.iloc[i]['content'][:80]}...")
        
        # Show active emotions
        active_emotions = [col for col in emotion_cols if df.iloc[i][col] == 1]
        if active_emotions:
            print(f"Emotions: {', '.join(active_emotions)}")
        else:
            print("Emotions: None")

# Example usage
if __name__ == "__main__":
    input_file = 'poem_emotions_multilabel.csv'
    output_file = 'poem_emotions_consolidated.csv'
    
    try:
        # Read original data for comparison
        original_df = pd.read_csv(input_file, encoding='utf-8')
        
        # Consolidate emotions
        consolidated_df = consolidate_emotions(input_file, output_file)
        
        # Analyze the impact
        analyze_consolidation_impact(original_df, consolidated_df)
        
        # Preview results
        preview_consolidation(consolidated_df)
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found. Please check the file path.")
    except Exception as e:
        print(f"Error: {str(e)}")

# Alternative function for custom emotion mapping
def consolidate_emotions_custom_mapping(input_csv_path: str, 
                                      emotion_mapping: dict, 
                                      target_emotions: list,
                                      output_csv_path: str = None):
    """
    Consolidate emotions with custom mapping
    
    Args:
        input_csv_path (str): Path to input CSV
        emotion_mapping (dict): Dictionary mapping original emotions to target emotions
        target_emotions (list): List of target emotion categories
        output_csv_path (str, optional): Path to save output
    
    Returns:
        pandas.DataFrame: Consolidated dataset
    """
    df = pd.read_csv(input_csv_path, encoding='utf-8')
    
    content_column = df['content']
    emotion_columns = [col for col in df.columns if col != 'content']
    
    # Initialize new data
    new_data = {'content': content_column}
    for emotion in target_emotions:
        new_data[emotion] = [0] * len(df)
    
    # Apply mapping
    for original_emotion in emotion_columns:
        if original_emotion in emotion_mapping:
            target_emotion = emotion_mapping[original_emotion]
            if target_emotion in target_emotions:
                for i in range(len(df)):
                    if df.iloc[i][original_emotion] == 1:
                        new_data[target_emotion][i] = 1
    
    consolidated_df = pd.DataFrame(new_data)
    column_order = ['content'] + target_emotions
    consolidated_df = consolidated_df[column_order]
    
    if output_csv_path:
        consolidated_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    return consolidated_df