import argparse, os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import json
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm

## import user lib
from base.data_eeg import load_eeg_data
from base.utils import update_config, ClipLoss, instantiate_from_config, get_device

device = get_device('auto')


def load_joint_model(config, img_checkpoint_path, text_checkpoint_path):
    """加载两个预训练模型并冻结参数"""
    model_img = instantiate_from_config(config['models']['brain_img'])
    model_text = instantiate_from_config(config['models']['brain_text'])

    print(f"\n{'='*60}")
    print("Loading Pre-trained Backbones (Frozen)")
    print('='*60)

    # Load Image Model
    img_checkpoint = torch.load(img_checkpoint_path, map_location='cpu', weights_only=False)
    img_state_dict = img_checkpoint['state_dict'] if 'state_dict' in img_checkpoint else img_checkpoint
    img_state_dict_clean = {k.replace('brain.', ''): v for k, v in img_state_dict.items() if k.startswith('brain.')}
    model_img.load_state_dict(img_state_dict_clean, strict=True)

    # Load Text Model
    text_checkpoint = torch.load(text_checkpoint_path, map_location='cpu', weights_only=False)
    text_state_dict = text_checkpoint['state_dict'] if 'state_dict' in text_checkpoint else text_checkpoint
    text_state_dict_clean = {k.replace('brain.', ''): v for k, v in text_state_dict.items() if k.startswith('brain.')}
    model_text.load_state_dict(text_state_dict_clean, strict=False)

    # Freeze Backbones
    for param in model_img.parameters():
        param.requires_grad = False
    for param in model_text.parameters():
        param.requires_grad = False
    model_img.eval()
    model_text.eval()

    return {'brain_img': model_img, 'brain_text': model_text}


class ScoreGatingNetwork(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim * 2, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, 1)
        )

    def forward(self, feat_img, feat_text):
        x = torch.cat([feat_img, feat_text], dim=-1)
        return torch.sigmoid(self.mlp(x))


class JointTestModel(pl.LightningModule):
    def __init__(self, model, config, gate_type='learnable_scalar', init_alpha=0.5):
        super().__init__()
        self.config = config
        self.brain_img = model['brain_img']
        self.brain_text = model['brain_text']
        self.z_dim = self.config['z_dim']
        self.gate_type = gate_type
        
        # Gating Parameters
        if gate_type == 'learnable_scalar':
            self.alpha_param = nn.Parameter(torch.tensor(float(init_alpha)))
            self.gate_net = None
        elif gate_type == 'dynamic_score':
            self.gate_net = ScoreGatingNetwork(self.z_dim)
            self.alpha_param = None
        elif gate_type == 'fixed':
            self.register_buffer('fixed_alpha', torch.tensor(float(init_alpha)))
            self.gate_net = None
            self.alpha_param = None
            
        # Channel Config
        self.channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                        'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
                        'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                        'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
                        'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                        'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                        'O1', 'Oz', 'O2']
        
        self.selected_ch_img = config['data']['selected_ch_img']
        self.selected_ch_text = config['data']['selected_ch_text']
        if self.selected_ch_img == "None": self.selected_ch_img = self.channels
        if self.selected_ch_text == "None": self.selected_ch_text = self.channels
        
        self.selected_idx_img = [self.channels.index(ch) for ch in self.selected_ch_img]
        self.selected_idx_text = [self.channels.index(ch) for ch in self.selected_ch_text]
        
        # Metrics Storage
        self.all_predicted_classes_img = []
        self.all_predicted_classes_text = []
        self.all_predicted_classes_joint = []
        self.all_true_labels = []
        self.mAP_total_img = 0
        self.mAP_total_text = 0
        self.mAP_total_joint = 0
        self.alpha_stats = []
        
        # 新增：保存每个样本的详细结果
        self.all_sample_info = []

    def _get_features_and_sims(self, batch):
        eeg_imgs = batch['eeg_img']
        eeg_texts = batch['eeg_text']
        img_z = batch['img_features']
        text_z = batch['text_features']
        
        eeg_img_ch = eeg_imgs[:, self.selected_idx_img, :]
        eeg_text_ch = eeg_texts[:, self.selected_idx_text, :]
        
        with torch.no_grad():
            feat_img = self.brain_img(eeg_img_ch)
            feat_text = self.brain_text(eeg_text_ch)
        
        feat_img = F.normalize(feat_img, p=2, dim=-1)
        feat_text = F.normalize(feat_text, p=2, dim=-1)
        img_z = F.normalize(img_z, p=2, dim=-1)
        text_z = F.normalize(text_z, p=2, dim=-1)
        
        logit_scale = getattr(self.brain_img, 'logit_scale', torch.tensor(1.0, device=feat_img.device)).exp()
        sim_img = (feat_img @ img_z.T) * logit_scale
        sim_text = (feat_text @ text_z.T) * logit_scale
        
        # Compute Alpha based on current mode
        if self.gate_type == 'learnable_scalar':
            alpha = torch.sigmoid(self.alpha_param)
            alpha_batch = alpha.expand(feat_img.shape[0], 1)
        elif self.gate_type == 'dynamic_score':
            alpha_batch = self.gate_net(feat_img, feat_text)
        elif self.gate_type == 'fixed':
            alpha_batch = self.fixed_alpha.expand(feat_img.shape[0], 1)
            
        sim_joint = alpha_batch * sim_img + (1 - alpha_batch) * sim_text
        
        return feat_img, feat_text, img_z, text_z, sim_img, sim_text, sim_joint, alpha_batch

    def test_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        feat_img, feat_text, img_z, text_z, sim_img, sim_text, sim_joint, alpha_batch = self._get_features_and_sims(batch)
        
        self.alpha_stats.append(alpha_batch.detach().cpu().numpy())
        
        # 获取 top-5 索引（注意处理 batch_size 可能小于 5 的情况）
        k = min(5, sim_img.size(1))
        _, top_k_indices_img = sim_img.topk(k, dim=-1)
        _, top_k_indices_text = sim_text.topk(k, dim=-1)
        _, top_k_indices_joint = sim_joint.topk(k, dim=-1)
        
        self.all_predicted_classes_img.append(top_k_indices_img.cpu().numpy())
        self.all_predicted_classes_text.append(top_k_indices_text.cpu().numpy())
        self.all_predicted_classes_joint.append(top_k_indices_joint.cpu().numpy())
        
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())
        
        for i in range(batch_size):
            true_idx = i
            rank_img = (torch.argsort(-sim_img[i]) == true_idx).nonzero()[0][0] + 1
            self.mAP_total_img += 1.0 / rank_img
            rank_text = (torch.argsort(-sim_text[i]) == true_idx).nonzero()[0][0] + 1
            self.mAP_total_text += 1.0 / rank_text
            rank_joint = (torch.argsort(-sim_joint[i]) == true_idx).nonzero()[0][0] + 1
            self.mAP_total_joint += 1.0 / rank_joint

        # ========== 新增：保存每个样本的详细结果（将索引转换为实际内容） ==========
        # 获取 batch 中的图片路径和文本描述，转换为 Python 列表便于索引
        img_paths = batch['img_path']
        texts = batch['text']
        if torch.is_tensor(img_paths):
            img_paths = img_paths.cpu().tolist()
        if torch.is_tensor(texts):
            texts = texts.cpu().tolist()

        # 确保 top_k_indices 是 CPU 张量并转换为列表
        top_k_img = top_k_indices_img.cpu()
        top_k_text = top_k_indices_text.cpu()
        top_k_joint = top_k_indices_joint.cpu()

        for i in range(batch_size):
            true_label_idx = i  # 真实标签的索引

            # 获取真实内容
            true_img_path = img_paths[true_label_idx]
            true_text = texts[true_label_idx]

            # 预测索引列表
            img_pred_indices = top_k_img[i].tolist()
            text_pred_indices = top_k_text[i].tolist()
            joint_pred_indices = top_k_joint[i].tolist()

            # 转换为实际内容
            img_pred_top1_path = img_paths[img_pred_indices[0]]
            img_pred_top5_paths = [img_paths[idx] for idx in img_pred_indices]

            text_pred_top1_text = texts[text_pred_indices[0]]
            text_pred_top5_texts = [texts[idx] for idx in text_pred_indices]
            
            text_pred_top1_path = img_paths[text_pred_indices[0]]
            text_pred_top5_paths = [img_paths[idx] for idx in text_pred_indices]

            joint_pred_top1_path = img_paths[joint_pred_indices[0]]
            joint_pred_top5_paths = [img_paths[idx] for idx in joint_pred_indices]

            # 正确性判断（基于索引）
            img_top1_correct = (img_pred_indices[0] == true_label_idx)
            img_top5_correct = (true_label_idx in img_pred_indices)
            text_top1_correct = (text_pred_indices[0] == true_label_idx)
            text_top5_correct = (true_label_idx in text_pred_indices)
            joint_top1_correct = (joint_pred_indices[0] == true_label_idx)
            joint_top5_correct = (true_label_idx in joint_pred_indices)

            sample_info = {
                'idx': batch['idx'][i].item() if torch.is_tensor(batch['idx'][i]) else batch['idx'][i],
                'true_label_idx': true_label_idx,
                'true_img_path': true_img_path,
                'true_text': true_text,
                'img_top1': img_pred_top1_path,
                'img_top5': img_pred_top5_paths,
                'img_top1_correct': img_top1_correct,
                'img_top5_correct': img_top5_correct,
                'text_top1_text': text_pred_top1_text,
                'text_top5_text': text_pred_top5_texts,
                'text_top1_correct': text_top1_correct,
                'text_top5_correct': text_top5_correct,
                'text_top1': text_pred_top1_path,
                'text_top5': text_pred_top5_paths,
                'joint_top1': joint_pred_top1_path,
                'joint_top5': joint_pred_top5_paths,
                'joint_top1_correct': joint_top1_correct,
                'joint_top5_correct': joint_top5_correct,
            }
            self.all_sample_info.append(sample_info)
        # ================================================

        return None

    def on_test_epoch_end(self):
        if not self.all_true_labels: return
        
        pred_img = np.concatenate(self.all_predicted_classes_img, axis=0)
        pred_text = np.concatenate(self.all_predicted_classes_text, axis=0)
        pred_joint = np.concatenate(self.all_predicted_classes_joint, axis=0)
        labels = np.array(self.all_true_labels)
        n_samples = len(labels)
        
        def calc_acc(preds, lbls):
            t1 = np.mean(preds[:, 0] == lbls)
            t5 = np.mean((preds == lbls[:, np.newaxis]).any(axis=1))
            return t1, t5
            
        t1_img, t5_img = calc_acc(pred_img, labels)
        t1_text, t5_text = calc_acc(pred_text, labels)
        t1_joint, t5_joint = calc_acc(pred_joint, labels)
        
        results = {
            'test_top1_acc_img': t1_img, 'test_top5_acc_img': t5_img, 'mAP_img': self.mAP_total_img / n_samples,
            'test_top1_acc_text': t1_text, 'test_top5_acc_text': t5_text, 'mAP_text': self.mAP_total_text / n_samples,
            'test_top1_acc_joint': t1_joint, 'test_top5_acc_joint': t5_joint, 'mAP_joint': self.mAP_total_joint / n_samples
        }
        
        if self.alpha_stats:
            alphas = np.concatenate(self.alpha_stats, axis=0)
            results['gate_alpha_mean'] = float(np.mean(alphas))
            results['gate_alpha_std'] = float(np.std(alphas))
            print(f"\n[Gating Stats] Mean Alpha: {results['gate_alpha_mean']:.4f} (Std: {results['gate_alpha_std']:.4f})")
            if self.gate_type == 'learnable_scalar':
                val = torch.sigmoid(self.alpha_param).item()
                print(f"  -> Used Global Weight (Image): {val:.4f}")

        for k, v in results.items():
            self.log(k, v, sync_dist=True, batch_size=n_samples)
        
        print("\n" + "="*60)
        print("Final Test Results")
        print("="*60)
        print(f"Image       : Top1 {t1_img:.4f} | Top5 {t5_img:.4f} | mAP {results['mAP_img']:.4f}")
        print(f"Text        : Top1 {t1_text:.4f} | Top5 {t5_text:.4f} | mAP {results['mAP_text']:.4f}")
        print(f"Joint (Best): Top1 {t1_joint:.4f} | Top5 {t5_joint:.4f} | mAP {results['mAP_joint']:.4f}")
        print("="*60)
        
        # 保存每个样本的详细结果到 JSON 文件
        if self.all_sample_info and hasattr(self, 'logger') and self.logger is not None:
            sample_file = os.path.join(self.logger.log_dir, 'test_samples_results.json')
            with open(sample_file, 'w') as f:
                json.dump(self.all_sample_info, f, indent=4)
            print(f"Saved per-sample results to {sample_file}")
        
        # Reset
        self.all_predicted_classes_img = []
        self.all_predicted_classes_text = []
        self.all_predicted_classes_joint = []
        self.all_true_labels = []
        self.mAP_total_img = 0; self.mAP_total_text = 0; self.mAP_total_joint = 0
        self.alpha_stats = []
        self.all_sample_info = []
        return results

    def configure_optimizers(self):
        return None


def grid_search_top1(model, val_loader, device, gate_type):
    """
    在验证集上执行高精度网格搜索，返回最佳alpha和所有结果排序后的前50个
    """
    print("\n" + "=" * 60)
    print("Phase 1: High-Precision Grid Search (Step=0.001)")
    print("Scanning alpha from 0.000 to 1.000 (1001 points)")
    print("=" * 60)

    model.eval()
    model.to(device)

    alphas = np.linspace(0.0, 1.0, 1001)

    best_alpha = 0.5
    best_top1 = -1.0
    best_top5 = -1.0
    best_map = -1.0
    all_results = []  # 存储所有alpha的结果

    with torch.no_grad():
        for a in tqdm(alphas, desc="Searching (0.001 step)"):
            if gate_type == 'learnable_scalar':
                a_safe = max(min(a, 1.0 - 1e-7), 1e-7)
                logit = np.log(a_safe / (1.0 - a_safe))
                model.alpha_param.data = torch.tensor([logit], device=device)

            elif gate_type == 'fixed':
                model.fixed_alpha.data = torch.tensor([a], device=device)

            elif gate_type == 'dynamic_score':
                continue

            total_correct_1 = 0
            total_correct_5 = 0
            total_samples = 0
            sum_rank = 0.0

            for batch in val_loader:
                batch_dev = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_dev[k] = v.to(device, non_blocking=True)
                    else:
                        batch_dev[k] = v

                try:
                    _, _, _, _, sim_img, sim_text, sim_joint, _ = model._get_features_and_sims(batch_dev)
                except RuntimeError as e:
                    print(f"\nError at alpha={a}: {e}")
                    raise e

                bs = sim_joint.shape[0]
                labels = torch.arange(bs, device=device)

                # Top-1
                _, top1_idx = sim_joint.topk(1, dim=-1)
                total_correct_1 += (top1_idx.squeeze() == labels).sum().item()

                # Top-5
                _, top5_idx = sim_joint.topk(min(5, sim_joint.size(1)), dim=-1)
                total_correct_5 += (top5_idx == labels.unsqueeze(1)).any(dim=1).sum().item()

                # mAP
                for i in range(bs):
                    rank_tensor = torch.argsort(-sim_joint[i])
                    pos = (rank_tensor == i).nonzero(as_tuple=True)[0]
                    if pos.numel() > 0:
                        rank = pos[0].item() + 1
                        sum_rank += 1.0 / rank

                total_samples += bs

            curr_top1 = total_correct_1 / total_samples
            curr_top5 = total_correct_5 / total_samples
            curr_map = sum_rank / total_samples if total_samples > 0 else 0

            # 记录所有结果
            all_results.append((a, curr_top1, curr_top5, curr_map))

            if curr_top1 > best_top1:
                best_top1 = curr_top1
                best_alpha = a
                best_top5 = curr_top5
                best_map = curr_map
            elif abs(curr_top1 - best_top1) < 1e-6 and curr_map > best_map:
                best_alpha = a
                best_top5 = curr_top5
                best_map = curr_map

    # 排序取前1000
    all_results.sort(key=lambda x: (x[1], x[3]), reverse=True)  # 按top1降序，相同则按mAP降序
    best_results = all_results[:1000]

    print(f"\n>>> SEARCH COMPLETE <<<")
    print(f"Best Alpha Found: {best_alpha:.4f}")
    print(f"Validation Top-1: {best_top1:.4f} | Top-5: {best_top5:.4f} | mAP: {best_map:.4f}")
    print("\nTop 5 Alphas:")
    for i, (a, t1, t5, mp) in enumerate(best_results[:5]):
        print(f"  {i+1}. alpha={a:.4f}, top1={t1:.4f}, top5={t5:.4f}, mAP={mp:.4f}")

    if gate_type == 'learnable_scalar':
        a_safe = max(min(best_alpha, 1.0 - 1e-7), 1e-7)
        best_logit = np.log(a_safe / (1.0 - a_safe))
        model.alpha_param.data = torch.tensor([best_logit], device=device)
    elif gate_type == 'fixed':
        model.fixed_alpha.data = torch.tensor([best_alpha], device=device)

    return best_alpha, best_top1, best_results   # 返回最佳alpha、最佳top1和前1000结果列表


def main():
    parser = argparse.ArgumentParser(description="Joint test with High-Precision Grid Search")
    parser.add_argument("--config", type=str, default="configs/eeg/joint_test.yaml")
    parser.add_argument("--img_checkpoint", type=str, required=True)
    parser.add_argument("--text_checkpoint", type=str, required=True)

    parser.add_argument("--gate_type", type=str, default='learnable_scalar', choices=['learnable_scalar', 'fixed'])
    parser.add_argument("--init_alpha", type=float, default=0.5)
    parser.add_argument("--search_top1", action='store_true')

    parser.add_argument("--dataset", type=str, default="eeg", choices=["eeg", "meg"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subjects", type=str, default='sub-08')
    parser.add_argument("--exp_setting", type=str, default='intra-subject')
    parser.add_argument("--vision_backbone", type=str, default='ViT-H-14')
    parser.add_argument("--text_backbone", type=str, default='ViT-H-14')
    parser.add_argument("--c", type=int, default=6)

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    config = update_config(opt, config)
    config['data']['subjects'] = [opt.subjects]

    pretrain_map = {
        'RN50': 1024, 'RN101': 512,
        'ViT-B-16': 512, 'ViT-B-32': 512,
        'ViT-L-14': 768, 'ViT-H-14': 1024,
        'ViT-g-14': 1024, 'ViT-bigG-14': 1280
    }
    config['z_dim'] = pretrain_map.get(opt.vision_backbone, 1024)
    config['vision_backbone'] = opt.vision_backbone
    config['text_backbone'] = opt.text_backbone

    os.makedirs(config['save_dir'], exist_ok=True)

    # 日志目录结构：name="gate_test", version=opt.subjects
    logger_name = "gate_test"
    logger_version = opt.subjects
    logger = TensorBoardLogger(
        config['save_dir'],
        name=logger_name,
        version=logger_version
    )
    os.makedirs(logger.log_dir, exist_ok=True)
    shutil.copy(opt.config, os.path.join(logger.log_dir, os.path.basename(opt.config)))

    train_loader_orig, val_loader_orig, test_loader = load_eeg_data(config)

    print(f"Data Loaded:")
    print(f"  Val Set Size: {len(val_loader_orig.dataset)}")
    print(f"  Test Set Size: {len(test_loader.dataset)}")
    print(f"Results will be saved to: {logger.log_dir}")

    backbones = load_joint_model(config, opt.img_checkpoint, opt.text_checkpoint)
    pl_model = JointTestModel(backbones, config, gate_type=opt.gate_type, init_alpha=opt.init_alpha)

    pl_model.to(device)

    final_alpha = opt.init_alpha

    if opt.search_top1:
        # 接收三个返回值：最佳alpha、最佳top1、前1000结果列表
        final_alpha, best_val_top1, best_results = grid_search_top1(pl_model, val_loader_orig, device, opt.gate_type)
        print(f"\nOptimization Done. Using Alpha = {final_alpha:.4f} for Testing.")

        # 保存前1000个最佳alpha及其指标
        top1000_file = os.path.join(logger.log_dir, 'top1000_alphas.json')
        with open(top1000_file, 'w') as f:
            json.dump([
                {'alpha': a, 'top1': t1, 'top5': t5, 'mAP': mp}
                for a, t1, t5, mp in best_results
            ], f, indent=4)
        print(f"Top 1000 alphas saved to: {top1000_file}")
    else:
        print(f"\nSkipping Search. Using Initial/Fixed Alpha = {opt.init_alpha:.4f}")
        if opt.gate_type == 'learnable_scalar':
            a_safe = max(min(opt.init_alpha, 1.0 - 1e-7), 1e-7)
            pl_model.alpha_param.data = torch.tensor([np.log(a_safe / (1.0 - a_safe))], device=device)
        elif opt.gate_type == 'fixed':
            pl_model.fixed_alpha.data = torch.tensor([opt.init_alpha], device=device)

    print("\n" + "=" * 60)
    print("Phase 2: Final Testing on Test Set")
    print("=" * 60)

    trainer = Trainer(
        devices=[device],
        accelerator='cuda',
        enable_checkpointing=False,
        logger=logger,
        enable_progress_bar=True
    )

    results = trainer.test(pl_model, dataloaders=test_loader)

    output_file = os.path.join(logger.log_dir, 'results.json')
    with open(output_file, 'w') as f:
        final_res = results[0] if isinstance(results, list) else results
        final_res['search_info'] = {
            'method': 'grid_search_top1_high_precision' if opt.search_top1 else 'none',
            'step_size': 0.001 if opt.search_top1 else None,
            'best_alpha': float(final_alpha),
            'subject': opt.subjects,
            'gate_type': opt.gate_type,
            'seed': opt.seed
        }
        json.dump(final_res, f, indent=4)

    print(f"\nAll results saved to: {logger.log_dir}")
    print(f"Best Alpha Used: {final_alpha:.4f}")


if __name__ == "__main__":
    main()