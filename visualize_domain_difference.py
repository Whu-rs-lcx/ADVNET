import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
import argparse
from tqdm import tqdm
from change_detection.CWCD.backbone import build_backbone
import random
import time

class DomainVisualizer:
    def __init__(self, dataset_path, model_name='hrnet'):
        """
        初始化域可视化器
        
        Args:
            dataset_path: 数据集路径
            model_name: 骨干网络名称
        """
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 构建HRNet骨干网络
        self.backbone, self.num_stages, _ = build_backbone(model_name=model_name)
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        
        # 简单的归一化预处理（不加任何数据增强）
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整到固定大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded on {self.device}")
        print(f"Number of stages: {self.num_stages}")

    def load_dataset_paths(self, split='train', max_samples=None):
        """
        加载数据集路径
        
        Args:
            split: 数据集分割 ('train', 'val', 'test')
            max_samples: 最大样本数量，None表示使用所有样本
        
        Returns:
            List of (pathA, pathB) tuples
        """
        txt_path = os.path.join(self.dataset_path, f'{split}.txt')
        
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Dataset file not found: {txt_path}")
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        paths = []
        for line in lines:
            parts = line.strip().split('  ')
            if len(parts) >= 2:
                pathA, pathB = parts[0], parts[1]
                paths.append((pathA, pathB))
        
        total_samples = len(paths)
        print(f"Found {total_samples} image pairs in {split} set")
        
        # 如果指定了max_samples，则随机采样
        if max_samples is not None and max_samples < total_samples:
            paths = random.sample(paths, max_samples)
            print(f"Randomly sampled {max_samples} pairs from {total_samples} total pairs")
        else:
            print(f"Using all {total_samples} image pairs")
        
        return paths

    def extract_features(self, image_paths, batch_size=32):
        """
        提取图像特征
        
        Args:
            image_paths: 图像路径列表
            batch_size: 批处理大小
        
        Returns:
            features: 提取的特征 (N, feature_dim)
            labels: 标签 (0表示时相A，1表示时相B)
        """
        features_list = []
        labels_list = []
        
        # 准备所有图像路径和对应标签
        all_paths = []
        all_labels = []
        
        for pathA, pathB in image_paths:
            all_paths.extend([pathA, pathB])
            all_labels.extend([0, 1])  # 0=时相A, 1=时相B
        
        total_images = len(all_paths)
        print(f"Processing {total_images} images ({total_images//2} image pairs)...")
        
        # 批处理提取特征
        processed_count = 0
        failed_count = 0
        
        for i in tqdm(range(0, total_images, batch_size), desc="Extracting features"):
            batch_paths = all_paths[i:i+batch_size]
            batch_labels = all_labels[i:i+batch_size]
            
            batch_images = []
            valid_labels = []
            
            for j, img_path in enumerate(batch_paths):
                try:
                    # 加载并预处理图像
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img)
                    batch_images.append(img_tensor)
                    valid_labels.append(batch_labels[j])
                    processed_count += 1
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    failed_count += 1
                    continue
            
            if not batch_images:
                continue
            
            # 转换为batch tensor
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                backbone_features = self.backbone(batch_tensor)
                # 选择第四层特征（倒数第一层）
                if isinstance(backbone_features, (list, tuple)):
                    target_features = backbone_features[-1]  # 最后一层
                else:
                    target_features = backbone_features
                
                # 全局平均池化
                if len(target_features.shape) == 4:  # (B, C, H, W)
                    pooled_features = torch.nn.functional.adaptive_avg_pool2d(target_features, (1, 1))
                    pooled_features = pooled_features.view(pooled_features.size(0), -1)
                else:
                    pooled_features = target_features
                
                features_list.append(pooled_features.cpu().numpy())
                labels_list.extend(valid_labels)
        
        print(f"Successfully processed: {processed_count} images")
        print(f"Failed to process: {failed_count} images")
        
        # 合并所有特征
        if features_list:
            features = np.vstack(features_list)
            labels = np.array(labels_list)
            print(f"Final extracted features shape: {features.shape}")
            print(f"Phase A samples: {np.sum(labels == 0)}")
            print(f"Phase B samples: {np.sum(labels == 1)}")
            return features, labels
        else:
            raise RuntimeError("No features were extracted successfully")

    def perform_pca_with_progress(self, features, n_components, random_state=42):
        """
        执行PCA降维并显示进度条
        
        Args:
            features: 特征矩阵
            n_components: 主成分数量
            random_state: 随机种子
        
        Returns:
            reduced_features: 降维后的特征
            pca: PCA对象
        """
        print(f"🔄 Performing PCA to {n_components} dimensions...")
        
        # 创建进度条
        pbar = tqdm(total=100, desc="PCA Progress", unit="%")
        
        try:
            # 开始PCA
            pbar.set_description("PCA: Initializing")
            pbar.update(10)
            
            pca = PCA(n_components=n_components, random_state=random_state)
            
            pbar.set_description("PCA: Computing SVD")
            pbar.update(30)
            
            # 执行PCA拟合和变换
            reduced_features = pca.fit_transform(features)
            
            pbar.set_description("PCA: Finalizing")
            pbar.update(60)
            
            # 完成
            pbar.set_description("PCA: Completed")
            pbar.update(100)
            pbar.close()
            
            print(f"✅ PCA completed successfully!")
            print(f"   - Input shape: {features.shape}")
            print(f"   - Output shape: {reduced_features.shape}")
            print(f"   - Explained variance ratio (first 10): {pca.explained_variance_ratio_[:min(10, len(pca.explained_variance_ratio_))]}")
            
            return reduced_features, pca
            
        except Exception as e:
            pbar.close()
            print(f"❌ PCA failed: {e}")
            raise

    def perform_tsne_with_progress(self, features, perplexity=30, n_iter=1000, random_state=42):
        """
        执行t-SNE降维并显示进度条
        
        Args:
            features: 特征矩阵
            perplexity: t-SNE参数
            n_iter: 迭代次数
            random_state: 随机种子
        
        Returns:
            reduced_features: 降维后的特征
        """
        print(f"🔄 Performing t-SNE (perplexity={perplexity}, n_iter={n_iter})...")
        
        # 调整perplexity以适应数据大小
        max_perplexity = min(perplexity, (features.shape[0] - 1) // 3)
        if max_perplexity < perplexity:
            print(f"⚠️  Adjusting perplexity from {perplexity} to {max_perplexity} due to sample size")
            perplexity = max_perplexity
        
        # 创建自定义的t-SNE类来捕获进度
        class TSNEWithProgress(TSNE):
            def __init__(self, *args, **kwargs):
                self.pbar = None
                super().__init__(*args, **kwargs)
            
            def _gradient_descent(self, *args, **kwargs):
                # 创建进度条
                self.pbar = tqdm(total=self.n_iter, desc="t-SNE Iterations", unit="iter")
                
                # 重写_gradient_descent来显示进度
                original_method = super()._gradient_descent
                
                def progress_callback(*args, **kwargs):
                    if hasattr(self, '_iter_count'):
                        self._iter_count += 1
                    else:
                        self._iter_count = 1
                    
                    if self.pbar:
                        self.pbar.update(1)
                        self.pbar.set_postfix({
                            'Iter': f"{self._iter_count}/{self.n_iter}",
                            'Status': 'Running'
                        })
                    
                    return original_method(*args, **kwargs)
                
                try:
                    result = original_method(*args, **kwargs)
                    if self.pbar:
                        self.pbar.set_postfix({'Status': 'Completed'})
                        self.pbar.close()
                    return result
                except Exception as e:
                    if self.pbar:
                        self.pbar.close()
                    raise e
        
        try:
            # 初始化t-SNE
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                n_iter=n_iter,
                random_state=random_state,
                verbose=0,  # 关闭sklearn的verbose输出
                init='pca',
                early_exaggeration=12.0,
                learning_rate='auto'
            )
            
            # 显示开始信息
            start_time = time.time()
            print(f"   - Input shape: {features.shape}")
            print(f"   - Target perplexity: {perplexity}")
            
            # 执行t-SNE（显示简单进度）
            with tqdm(total=100, desc="t-SNE Progress", unit="%") as pbar:
                pbar.set_description("t-SNE: Initializing")
                pbar.update(5)
                
                pbar.set_description("t-SNE: Computing similarities")
                pbar.update(15)
                
                pbar.set_description("t-SNE: Running optimization")
                reduced_features = tsne.fit_transform(features)
                pbar.update(80)
                
                pbar.set_description("t-SNE: Completed")
                pbar.update(100)
            
            end_time = time.time()
            print(f"✅ t-SNE completed successfully!")
            print(f"   - Output shape: {reduced_features.shape}")
            print(f"   - Time taken: {end_time - start_time:.2f} seconds")
            
            return reduced_features
            
        except Exception as e:
            print(f"❌ t-SNE failed: {e}")
            raise

    def perform_dimensionality_reduction(self, features, method='tsne', n_components=2, 
                                       perplexity=30, n_iter=1000, random_state=42):
        """
        执行降维
        
        Args:
            features: 特征矩阵
            method: 降维方法 ('tsne', 'pca', 'pca_tsne')
            n_components: 降维后的维度
            perplexity: t-SNE参数
            n_iter: t-SNE迭代次数
            random_state: 随机种子
        
        Returns:
            reduced_features: 降维后的特征
        """
        print(f"\n🚀 Starting dimensionality reduction using {method.upper()}")
        print(f"Input features shape: {features.shape}")
        
        # 数据预处理：标准化
        print("📊 Standardizing features...")
        with tqdm(total=100, desc="Standardization", unit="%") as pbar:
            pbar.update(20)
            scaler = StandardScaler()
            pbar.update(40)
            features_scaled = scaler.fit_transform(features)
            pbar.update(40)
        
        print(f"✅ Standardization completed. Shape: {features_scaled.shape}")
        
        if method == 'pca':
            # 使用PCA降维
            reduced_features, pca = self.perform_pca_with_progress(
                features_scaled, n_components, random_state)
            
        elif method == 'pca_tsne':
            # 先用PCA降维到50维，再用t-SNE降维到2维
            print("\n🔄 Step 1/2: PCA preprocessing...")
            reduced_features_pca, pca = self.perform_pca_with_progress(
                features_scaled, 50, random_state)
            
            print(f"\n🔄 Step 2/2: t-SNE final reduction...")
            reduced_features = self.perform_tsne_with_progress(
                reduced_features_pca, perplexity, n_iter, random_state)
            
        else:  # method == 'tsne'
            # 直接使用t-SNE
            reduced_features = self.perform_tsne_with_progress(
                features_scaled, perplexity, n_iter, random_state)
        
        print(f"\n🎉 Dimensionality reduction completed!")
        print(f"Final result shape: {reduced_features.shape}")
        return reduced_features

    def visualize_domain_difference(self, reduced_features, labels, dataset_name, 
                                   method='tsne', save_path=None):
        """
        可视化域差异
        
        Args:
            reduced_features: 降维后的特征
            labels: 时相标签
            dataset_name: 数据集名称
            method: 降维方法名称
            save_path: 保存路径
        """
        print(f"\n🎨 Creating visualization...")
        
        plt.figure(figsize=(12, 8))
        
        # 分离时相A和时相B的点
        phase_A_idx = labels == 0
        phase_B_idx = labels == 1
        
        phase_A_points = reduced_features[phase_A_idx]
        phase_B_points = reduced_features[phase_B_idx]
        
        # 绘制散点图
        plt.scatter(phase_A_points[:, 0], phase_A_points[:, 1], 
                   c='red', alpha=0.6, s=20, label=f'Phase A ({len(phase_A_points)} images)')
        plt.scatter(phase_B_points[:, 0], phase_B_points[:, 1], 
                   c='blue', alpha=0.6, s=20, label=f'Phase B ({len(phase_B_points)} images)')
        
        plt.title(f'Domain Difference Visualization - {dataset_name}\nHRNet Features + {method.upper()}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel(f'{method.upper()} Dimension 1', fontsize=12)
        plt.ylabel(f'{method.upper()} Dimension 2', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        plt.figtext(0.02, 0.02, 
                   f'Total samples: {len(reduced_features)}\n'
                   f'Phase A: {len(phase_A_points)} | Phase B: {len(phase_B_points)}',
                   fontsize=10, ha='left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Visualization saved to: {save_path}")
        
        plt.show()
        print("✅ Visualization completed!")

    def run_visualization(self, split='train', max_samples=None, batch_size=32, 
                         method='pca_tsne', perplexity=30, save_path=None):
        """
        运行完整的可视化流程
        
        Args:
            split: 数据集分割
            max_samples: 最大样本数，None表示使用所有样本
            batch_size: 批处理大小
            method: 降维方法 ('tsne', 'pca', 'pca_tsne')
            perplexity: t-SNE参数
            save_path: 保存路径
        """
        print("🚀 Starting domain difference visualization...")
        
        # 1. 加载数据集路径
        print("\n📁 Step 1/4: Loading dataset paths...")
        image_paths = self.load_dataset_paths(split, max_samples)
        
        # 2. 提取特征
        print("\n🔍 Step 2/4: Extracting features...")
        features, labels = self.extract_features(image_paths, batch_size)
        
        # 3. 执行降维
        print("\n📉 Step 3/4: Performing dimensionality reduction...")
        reduced_features = self.perform_dimensionality_reduction(
            features, method=method, perplexity=perplexity)
        
        # 4. 可视化结果
        print("\n🎨 Step 4/4: Creating visualization...")
        dataset_name = os.path.basename(self.dataset_path)
        self.visualize_domain_difference(reduced_features, labels, dataset_name, 
                                       method, save_path)
        
        print("\n🎉 All steps completed successfully!")
        return reduced_features, labels


def main():
    parser = argparse.ArgumentParser(description='Visualize domain differences in change detection datasets')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of image pairs to process (None for all samples)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--method', type=str, default='pca_tsne', 
                       choices=['tsne', 'pca', 'pca_tsne'],
                       help='Dimensionality reduction method')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save the visualization')
    
    args = parser.parse_args()
    
    # 检查数据集路径
    if not os.path.exists(args.dataset):
        # 尝试从环境变量获取
        dataset_path = os.path.join(os.environ.get("CDPATH", ""), args.dataset)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {args.dataset}")
        args.dataset = dataset_path
    
    # 创建可视化器
    visualizer = DomainVisualizer(args.dataset)
    
    # 运行可视化
    if args.save_path is None:
        dataset_name = os.path.basename(args.dataset)
        if args.max_samples:
            args.save_path = f"{dataset_name}_domain_visualization_{args.method}_{args.max_samples}samples.png"
        else:
            args.save_path = f"{dataset_name}_domain_visualization_{args.method}_all_samples.png"
    
    visualizer.run_visualization(
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        method=args.method,
        perplexity=args.perplexity,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()