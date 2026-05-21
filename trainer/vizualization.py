

class VIBE_DeepVisualizer:
    def __init__(self, log_dir):
        self.save_dir = Path(log_dir) / "deep_visualizations"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        self.colors = {'Pos': '#2E86C1', 'Neg': '#E74C3C', 'Neu': '#95A5A6'} 
        self.history = {'epoch': [], 'ebs': [], 'rc': [], 'acc': [], 'gamma_mean': []}

    def update_history(self, epoch, ebs, rc, acc, gamma_mean=0.0):
        self.history['epoch'].append(epoch)
        self.history['ebs'].append(ebs)
        self.history['rc'].append(rc)
        self.history['acc'].append(acc)
        self.history['gamma_mean'].append(gamma_mean)

    def plot_3d_manifold(self, h_final, labels, epoch):
        if len(labels) > 800:
            idx = np.random.choice(len(labels), 800, replace=False)
            h_final, labels = h_final[idx], labels[idx]
        tsne = TSNE(n_components=3, perplexity=30, init='pca', learning_rate='auto')
        emb = tsne.fit_transform(h_final)
        class_names = ['Pos', 'Neg', 'Neu']
        txt_labels = [class_names[i] for i in labels]
        colors = [self.colors[l] for l in txt_labels]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=colors, s=50, alpha=0.7, edgecolors='w')
        
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors[k], label=k, markersize=10) for k in class_names]
        ax.legend(handles=legend_elements, loc='upper right', title="Emotion")
        ax.set_title(f"3D Manifold Structure - Epoch {epoch}", fontsize=16, fontweight='bold')
        
        ax.view_init(elev=20, azim=45)
        plt.savefig(self.save_dir / f"manifold_3d_ep{epoch}_v1.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_disentanglement_proof(self, z_aff, z_env, labels, epoch):
        if len(labels) > 600:
            idx = np.random.choice(len(labels), 600, replace=False)
            z_aff, z_env, labels = z_aff[idx], z_env[idx], labels[idx]
        combined = np.concatenate([z_aff, z_env], axis=0)
        proj = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto').fit_transform(combined)
        p_aff, p_env = proj[:len(z_aff)], proj[len(z_aff):]
        class_names = ['Pos', 'Neg', 'Neu']
        txt_labels = [class_names[i] for i in labels]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        sns.scatterplot(x=p_aff[:,0], y=p_aff[:,1], hue=txt_labels, palette=self.colors, style=txt_labels, s=60, alpha=0.8, ax=axes[0])
        axes[0].set_title(f"Affect Latents (Clusters)", fontsize=14, fontweight='bold')
        sns.scatterplot(x=p_env[:,0], y=p_env[:,1], hue=txt_labels, palette=self.colors, style=txt_labels, s=60, alpha=0.4, ax=axes[1], legend=False)
        axes[1].set_title(f"Environment Latents (Random)", fontsize=14, fontweight='bold')
        plt.savefig(self.save_dir / f"disentangle_ep{epoch}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_gamma_dynamics(self, gammas, labels, epoch):
        class_names = ['Pos', 'Neg', 'Neu']
        txt_labels = [class_names[i] for i in labels]
        gammas = np.array(gammas).flatten()
        plt.figure(figsize=(8, 6))
        sns.violinplot(x=txt_labels, y=gammas, palette=self.colors, inner="box", linewidth=1.5, alpha=0.8)
        sns.stripplot(x=txt_labels, y=gammas, color="black", alpha=0.2, size=3, jitter=True)
        plt.title(f"Group Synchrony (Gamma) by Emotion", fontsize=14, fontweight='bold')
        plt.savefig(self.save_dir / f"gamma_dist_ep{epoch}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_cot_alignment(self, rationales, text_anchors, epoch):
        sims = F.cosine_similarity(torch.tensor(rationales), torch.tensor(text_anchors))
        plt.figure(figsize=(8, 5))
        sns.histplot(sims.numpy(), bins=25, kde=True, color="#8E44AD", fill=True, alpha=0.6)
        mean_val = sims.mean().item()
        plt.axvline(mean_val, color='#2C3E50', linestyle='--', label=f'Mean: {mean_val:.2f}')
        plt.title(f"Vision-Text Alignment", fontsize=14, fontweight='bold')
        plt.legend()
        plt.savefig(self.save_dir / f"cot_hist_ep{epoch}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_metric_trends(self):
        if len(self.history['epoch']) < 2: return
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
        color_bias, color_acc = '#E74C3C', '#2E86C1'
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('EBS (Lower Better)', color=color_bias, fontweight='bold')
        ax1.plot(self.history['epoch'], self.history['ebs'], color=color_bias, marker='o')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color=color_acc, fontweight='bold')
        ax2.plot(self.history['epoch'], self.history['acc'], color=color_acc, marker='s', linestyle='--')
        ax1.set_title("Efficacy", fontsize=14, fontweight='bold')
        ax3.plot(self.history['epoch'], self.history['rc'], color='#8E44AD', marker='^')
        ax3.set_title("Interpretability", fontsize=14, fontweight='bold')
        plt.savefig(self.save_dir / "training_trends.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, targets, preds, epoch):
        cm = confusion_matrix(targets, preds, normalize='true')
        classes = ['Pos', 'Neg', 'Neu']
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix', fontsize=14, fontweight='bold')
        plt.savefig(self.save_dir / f"conf_mat_ep{epoch}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_reasoning_logic(self, predicates, labels, epoch):
        """
        Plots a Radar-style 'Logic Fingerprint' for each class.
        predicates: [N, 4] tensor
        labels: [N] array
        """
        class_names = ['Pos', 'Neg', 'Neu']
        pred_names = ['Energy', 'Integration', 'Dominance', 'Congruence']
        
        # Convert to DataFrame for easy manipulation
        df = pd.DataFrame(predicates, columns=pred_names)
        df['Emotion'] = [class_names[i] for i in labels]
        
        # Calculate mean predicates per class
        logic_summary = df.groupby('Emotion').mean().reset_index()
        
        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw=dict(polar=True))
        
        for i, (idx, row) in enumerate(logic_summary.iterrows()):
            emotion = row['Emotion']
            values = row[pred_names].values.flatten().tolist()
            values += values[:1] # Repeat first value to close the circle
            angles = [n / float(len(pred_names)) * 2 * np.pi for n in range(len(pred_names))]
            angles += angles[:1]
            
            axes[i].set_theta_offset(np.pi / 2)
            axes[i].set_theta_direction(-1)
            plt.xticks(angles[:-1], pred_names)
            
            # Fill and outline
            axes[i].plot(angles, values, color=self.colors[emotion], linewidth=2, linestyle='solid')
            axes[i].fill(angles, values, color=self.colors[emotion], alpha=0.4)
            axes[i].set_title(f"Logic: {emotion}", size=16, color=self.colors[emotion], weight='bold', pad=20)
            axes[i].set_ylim(0, 1.0) # Normalized range

        plt.tight_layout()
        plt.savefig(self.save_dir / f"reasoning_fingerprint_ep{epoch}.png", dpi=300)
        plt.close()

