import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def parse_multilabel_poetry(file_path: str):
    """
    Parse Chinese poetry dataset with correct format:
    topic, emotion1#emotion2#emotion3, title, content
    
    Args:
        file_path (str): Path to the input text file
    
    Returns:
        tuple: (DataFrame with content and emotion columns, list of emotion classes)
    """
    poems = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Split by comma, but be careful with the structure
                parts = line.split(',')
                if len(parts) < 4:
                    print(f"Skipping line {line_num}: not enough parts")
                    continue
                
                topic = parts[0].strip()
                emotions_str = parts[1].strip()
                title = parts[2].strip()
                # Join remaining parts as content (in case content contains commas)
                content = ','.join(parts[3:]).strip()
                
                # Parse emotions (split by #)
                emotions = [emotion.strip() for emotion in emotions_str.split('#') if emotion.strip()]
                
                if not emotions:
                    print(f"Skipping line {line_num}: no emotions found")
                    continue
                
                poems.append((content, emotions))
                
            except Exception as e:
                print(f"Skipping line {line_num} due to error: {line[:50]}...\nError: {e}")
    
    if not poems:
        raise ValueError("No valid poems found in the file")
    
    # Create DataFrame
    df = pd.DataFrame(poems, columns=['content', 'emotions'])
    
    # Multi-label binarizer
    mlb = MultiLabelBinarizer()
    emotion_df = pd.DataFrame(mlb.fit_transform(df['emotions']), columns=mlb.classes_)
    
    # Combine content and one-hot emotion vectors
    result = pd.concat([df['content'], emotion_df], axis=1)
    
    print(f"Successfully parsed {len(result)} poems")
    print(f"Found {len(mlb.classes_)} unique emotions: {list(mlb.classes_)}")
    
    return result, mlb.classes_

def parse_multilabel_poetry_alternative(file_path: str):
    """
    Alternative parsing method with more robust comma handling
    """
    poems = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Find the pattern: topic, emotions, title, content
                # Split into maximum 4 parts to handle commas in content
                parts = line.split(',', 3)
                
                if len(parts) != 4:
                    print(f"Skipping line {line_num}: incorrect format")
                    continue
                
                topic, emotions_str, title, content = parts
                
                # Parse emotions
                emotions = [emotion.strip() for emotion in emotions_str.split('#') if emotion.strip()]
                
                if not emotions:
                    print(f"Skipping line {line_num}: no emotions found")
                    continue
                
                poems.append((content.strip(), emotions))
                
            except Exception as e:
                print(f"Skipping line {line_num} due to error: {line[:50]}...\nError: {e}")
    
    if not poems:
        raise ValueError("No valid poems found in the file")
    
    # Create DataFrame
    df = pd.DataFrame(poems, columns=['content', 'emotions'])
    
    # Multi-label binarizer
    mlb = MultiLabelBinarizer()
    emotion_df = pd.DataFrame(mlb.fit_transform(df['emotions']), columns=mlb.classes_)
    
    # Combine content and one-hot emotion vectors
    result = pd.concat([df['content'], emotion_df], axis=1)
    
    print(f"Successfully parsed {len(result)} poems")
    print(f"Found {len(mlb.classes_)} unique emotions: {list(mlb.classes_)}")
    
    return result, mlb.classes_

def save_results(df, output_path='poem_emotions_multilabel.csv'):
    """
    Save the results to CSV file
    """
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Results saved to: {output_path}")

def preview_results(df, num_rows=3):
    """
    Preview the first few rows of results
    """
    print("\nPreview of results:")
    print("=" * 80)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    
    # Show content and a few emotion columns
    emotion_cols = [col for col in df.columns if col != 'content']
    preview_cols = ['content'] + emotion_cols[:5]  # Show first 5 emotions
    
    for i in range(min(num_rows, len(df))):
        print(f"\nRow {i+1}:")
        print(f"Content: {df.iloc[i]['content'][:100]}...")
        
        # Show active emotions for this poem
        active_emotions = [col for col in emotion_cols if df.iloc[i][col] == 1]
        print(f"Emotions: {', '.join(active_emotions)}")

# Example usage
if __name__ == "__main__":
    file_path = './Classical Chinese poetry_with_labels.txt'
    
    try:
        # Parse the dataset
        df_multilabel, emotion_classes = parse_multilabel_poetry_alternative(file_path)
        
        # Preview results
        preview_results(df_multilabel)
        
        # Save to CSV
        save_results(df_multilabel, 'poem_emotions_multilabel.csv')
        
        print(f"\nAll emotions found: {list(emotion_classes)}")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check the file path.")
    except Exception as e:
        print(f"Error: {str(e)}")

# For testing with your sample data
def test_with_sample():
    """
    Test with the sample data you provided
    """
    sample_data = """思乡,愁绪#想家,其四,北堂天未晓,游子归来早。堂前一夜风,开遍宜男草。
怀人,哀伤#想家,昊体寄张素长,九月十月天雨霜,江南剑南远路长。平生故人阳羊手,万里一书空断肠。人生避健已难得,怀人,哀伤#思念,多慧忆隐直,多劳清长饮,心事如波澜。高卧老将至,相思天正寒。浮云纷蔽雪,唳雀独酶等。里饮非无益"""
    
    # Save sample to temp file
    with open('temp_sample.txt', 'w', encoding='utf-8') as f:
        f.write(sample_data)
    
    try:
        df, emotions = parse_multilabel_poetry_alternative('temp_sample.txt')
        preview_results(df)
        return df, emotions
    except Exception as e:
        print(f"Test error: {e}")
        return None, None