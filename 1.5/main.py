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
    print("\n" + "="*60)
    print("1. 正在加载数据...")
    print("="*60)
    # 加载训练数据
    train_paths, train_labels, label_map = load_dataset()
    print(f"✓ 训练数据加载完成：{len(train_paths)} 张图片，{len(label_map)} 个类别")
    
    # 显示每个类别的详细统计
    from collections import Counter
    label_counts = Counter(train_labels)
    reverse_label_map_temp = {v: k for k, v in label_map.items()}
    print("\n训练数据类别分布：")
    for label_idx, count in sorted(label_counts.items()):
        class_name = reverse_label_map_temp[label_idx]
        print(f"  {class_name}: {count} 张")
    
    # 加载测试数据
    prediction_paths = load_prediction()
    print(f"\n✓ 预测数据加载完成：{len(prediction_paths)} 张图片")
    # 2. 数据分割（训练集和验证集）
    print("\n" + "="*60)
    print("2. 正在分割训练集和验证集...")
    print("="*60)
    train_paths_split, val_paths_split, train_labels_split, val_labels_split = train_test_split(
        train_paths, train_labels, 
        test_size=0.2,  # 20%作为验证集
        random_state=42, 
        stratify=train_labels  # 保持类别比例
    )
    print(f"✓ 数据分割完成：训练集 {len(train_paths_split)} 张，验证集 {len(val_paths_split)} 张")
    
    # 显示分割后的类别分布
    train_label_counts = Counter(train_labels_split)
    val_label_counts = Counter(val_labels_split)
    print("\n训练集类别分布：")
    for label_idx, count in sorted(train_label_counts.items()):
        class_name = reverse_label_map_temp[label_idx]
        print(f"  {class_name}: {count} 张")
    print("\n验证集类别分布：")
    for label_idx, count in sorted(val_label_counts.items()):
        class_name = reverse_label_map_temp[label_idx]
        print(f"  {class_name}: {count} 张")
    
    # 3. 创建和训练模型
    print("\n" + "="*60)
    print("3. 正在创建和训练模型...")
    print("="*60)
    try:
        # 创建分类器
        print("\n3.1 创建分类器...")
        n_estimators = 100
        max_depth = 6
        learning_rate = 0.1
        print(f"  超参数配置：")
        print(f"    - 树的数量 (n_estimators): {n_estimators}")
        print(f"    - 树的最大深度 (max_depth): {max_depth}")
        print(f"    - 学习率 (learning_rate): {learning_rate}")
        
        classifier = PlantClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
        )
        print("  ✓ 分类器创建完成")
        
        # 训练模型
        print("\n3.2 开始训练模型...")
        print("  - 特征提取中...")
        classifier.fit(train_paths_split, train_labels_split, label_map)
        print("  ✓ 模型训练完成")

        # 4. 验证集评估
        print("\n" + "="*60)
        print("4. 正在评估验证集性能...")
        print("="*60)
        try:
            print("\n4.1 对验证集进行预测...")
            val_predictions = classifier.predict(val_paths_split)
            print(f"  ✓ 预测完成：{len(val_predictions)} 个预测结果")
            
            # val_labels_split是数字标签，需要转换为类别名称
            reverse_label_map = {v: k for k, v in label_map.items()}
            val_true_labels = [reverse_label_map[label] for label in val_labels_split]

            # 计算准确率
            print("\n4.2 计算评估指标...")
            val_accuracy = accuracy_score(val_true_labels, val_predictions)
            print(f"  ✓ 验证集准确率: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
            
            # 打印详细分类报告
            print("\n4.3 详细分类报告：")
            print("-" * 60)
            print(classification_report(val_true_labels, val_predictions,
                            target_names=sorted(label_map.keys())))
            print("-" * 60)
            
            # 显示混淆矩阵（可选，更直观）
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(val_true_labels, val_predictions, labels=sorted(label_map.keys()))
            print("\n4.4 混淆矩阵（行=真实标签，列=预测标签）：")
            print("     ", end="")
            for name in sorted(label_map.keys()):
                print(f"{name[:8]:>8}", end="")
            print()
            for i, name in enumerate(sorted(label_map.keys())):
                print(f"{name[:8]:>8}", end="")
                for j in range(len(sorted(label_map.keys()))):
                    print(f"{cm[i,j]:>8}", end="")
                print()

        except Exception as e:
            print(f"❌ 验证集评估失败：{e}")
            return

    except Exception as e:
        print(f"❌ 模型训练失败：{e}")
        return

    # 5. 对测试数据进行预测
    print("\n" + "="*60)
    print("5. 正在对测试数据进行预测...")
    print("="*60)
    try:
        print(f"\n5.1 提取测试集特征（{len(prediction_paths)} 张图片）...")
        predictions = classifier.predict(prediction_paths)
        print(f"  ✓ 预测完成：{len(predictions)} 个预测结果")
        
        # 显示预测结果分布
        print("\n5.2 预测结果分布：")
        prediction_counts = pd.Series(predictions).value_counts()
        for category_name, count in prediction_counts.items():
            percentage = count / len(predictions) * 100
            print(f"  {category_name}: {count} 张 ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"❌ 测试预测失败：{e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. 保存结果
    print("\n" + "="*60)
    print("6. 正在保存结果...")
    print("="*60)
    try:
        # 创建结果目录
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        print(f"\n6.1 创建结果目录：{results_dir.absolute()}")
        
        # 保存预测结果
        print("\n6.2 保存预测结果到CSV...")
        image_ids = [Path(path).name for path in prediction_paths]
        results_df = pd.DataFrame({
            'ID': image_ids,
            'Category': predictions
        })
        results_path = results_dir / "predictions.csv"
        results_df.to_csv(results_path, index=False, encoding='utf-8')
        print(f"  ✓ 预测结果已保存到：{results_path}")
        print(f"    共 {len(results_df)} 条记录")

        # 保存模型
        print("\n6.3 保存训练好的模型...")
        model_path = results_dir / "plant_classifier.pkl"
        classifier.save(model_path)
        print(f"  ✓ 模型已保存到：{model_path}")
        
        # 显示模型信息
        if hasattr(classifier, 'features_dim_'):
            print(f"    特征维度：{classifier.features_dim_}")
        if hasattr(classifier, 'selected_features') and classifier.selected_features is not None:
            print(f"    特征选择后维度：{len(classifier.selected_features)}")

        # 保存标签映射
        print("\n6.4 保存标签映射...")
        label_map_path = results_dir / "label_map.json"
        import json
        with open(label_map_path, 'w', encoding='utf-8') as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 标签映射已保存到：{label_map_path}")
        print(f"    共 {len(label_map)} 个类别")

        # 显示预测结果统计
        print("\n6.5 最终预测结果统计：")
        prediction_counts = results_df['Category'].value_counts()
        for category_name, count in prediction_counts.items():
            percentage = count / len(predictions) * 100
            print(f"  {category_name}: {count} 张 ({percentage:.1f}%)")
        
        print("\n" + "="*60)
        print("✓ 所有结果已成功保存！")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 结果保存失败：{e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    # 运行主程序
    main()