import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from pathlib import Path
from data_load import load_dataset, load_prediction
from model import PlantClassifier

def main():
    np.random.seed(114514)
    
    # 1. 数据加载
    print("\n1. 正在加载数据...")
    # 加载训练数据
    train_paths, train_labels, label_map = load_dataset()
    print(f"✓ 训练数据加载完成：{len(train_paths)} 张图片，{len(label_map)} 个类别")
    # 加载测试数据
    prediction_paths = load_prediction()
    print(f"✓ 预测数据加载完成：{len(prediction_paths)} 张图片")
    # 2. 数据分割（训练集和验证集）
    print("\n2. 正在分割训练集和验证集...")
    train_paths_split, val_paths_split, train_labels_split, val_labels_split = train_test_split(
        train_paths, train_labels, 
        test_size=0.2,  # 20%作为验证集
        random_state=42, 
        stratify=train_labels  # 保持类别比例
    )
    print(f"✓ 数据分割完成：训练集 {len(train_paths_split)} 张，验证集 {len(val_paths_split)} 张")
    
    # 3. 创建和训练模型
    print("\n3. 正在创建和训练模型...")
    try:
        # 创建分类器
        classifier = PlantClassifier(
            n_estimators=100,      # 树的数量
            max_depth=6,          # 树的最大深度
            learning_rate=0.1     # 学习率
        )
        
        # 训练模型
        classifier.fit(train_paths_split, train_labels_split, label_map)
        print("✓ 模型训练完成")

        # 4. 验证集评估
        print("\n4. 正在评估验证集性能...")
        try:
            val_predictions = classifier.predict(val_paths_split)
            # val_labels_split是数字标签，需要转换为类别名称
            reverse_label_map = {v: k for k, v in label_map.items()}
            val_true_labels = [reverse_label_map[label] for label in val_labels_split]

            # 计算准确率
            val_accuracy = accuracy_score(val_true_labels, val_predictions)
            print(f"验证集准确率: {val_accuracy:.3f}")
            # 打印详细分类报告
            print("\n验证集分类报告：")
            print(classification_report(val_true_labels, val_predictions,
                            target_names=sorted(label_map.keys())))

        except Exception as e:
            print(f"❌ 验证集评估失败：{e}")
            return

    except Exception as e:
        print(f"❌ 模型训练失败：{e}")
        return

    # 5. 对测试数据进行预测
    print("\n5. 正在对测试数据进行预测...")
    try:
        predictions = classifier.predict(prediction_paths)
        print("✓ 测试数据预测完成")
        
    except Exception as e:
        print(f"❌ 测试预测失败：{e}")
        return
    
    # 6. 保存结果
    print("\n6. 正在保存结果...")
    try:
        # 创建结果目录
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        # 保存预测结果
        # 提取文件名（只保留*.png格式）
        image_ids = [Path(path).name for path in prediction_paths]

        results_df = pd.DataFrame({
            'ID': image_ids,
            'Category': predictions
        })
        results_path = results_dir / "predictions.csv"
        results_df.to_csv(results_path, index=False, encoding='utf-8')
        print(f"✓ 预测结果已保存到：{results_path}")

        # 保存模型
        model_path = results_dir / "plant_classifier.pkl"
        classifier.save(model_path)

        # 保存标签映射
        label_map_path = results_dir / "label_map.json"
        import json
        with open(label_map_path, 'w', encoding='utf-8') as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
        print(f"✓ 标签映射已保存到：{label_map_path}")

        # 显示预测结果统计
        prediction_counts = results_df['Category'].value_counts()
        print("\n预测结果统计：")
        for category_name, count in prediction_counts.items():
            print(f"  {category_name}: {count} 张")
        
    except Exception as e:
        print(f"❌ 结果保存失败：{e}")
        return

if __name__ == "__main__":
    # 运行主程序
    main()