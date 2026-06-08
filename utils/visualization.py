import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from core.renderer import differentiable_render_alpha_rgb

def visualize_clean(model, dataset, device='cuda', num_samples=5, save_path=None):
    """
    Pobiera próbki z datasetu, przepuszcza przez model i wizualizuje 
    porównanie: [Oryginał | Predykcja].
    """
    model.eval()
    model.to(device)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
        
    with torch.no_grad():
        for i in range(num_samples):
            # Pobranie czystego obrazu (lub zaszumionego, jeśli to usg_dataset)
            img, _ = dataset[i]
            img_tensor = img.unsqueeze(0).to(device)
            
            # Predykcja
            pred_params = model(img_tensor)
            
            # Renderowanie wyniku
            pred_img = differentiable_render_alpha_rgb(pred_params, image_size=img.shape[1])
            
            # Konwersja do numpy [H, W, C]
            img_np = img.permute(1, 2, 0).cpu().numpy()
            pred_np = pred_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            axes[i][0].imshow(np.clip(img_np, 0, 1))
            axes[i][0].set_title("Oryginał / Wejście")
            axes[i][0].axis('off')
            
            axes[i][1].imshow(np.clip(pred_np, 0, 1))
            axes[i][1].set_title("Predykcja Modelu")
            axes[i][1].axis('off')
            
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Zapisano wizualizację jako {save_path}")
    else:
        plt.show()
    plt.close()

def plot_tsne_latent_space(model, dataloader, device='cuda', num_samples=2000, save_path="tsne_gru.pdf"):
    """
    Generuje wykres t-SNE dla przestrzeni ukrytej (wektor C) modelu GRU,
    kolorując punkty na podstawie rzeczywistej długości sekwencji.
    """
    model.eval()
    model.to(device)
    
    latent_vectors = []
    labels = []
    
    print("Ekstrakcja cech do analizy t-SNE...")
    with torch.no_grad():
        for imgs, target_params in dataloader:
            imgs = imgs.to(device)
            
            # Wektor kontekstu C [batch_size, 128]
            C = model.encoder(imgs)
            latent_vectors.append(C.cpu().numpy())
            
            # Obliczenie długości sekwencji na podstawie flagi EOS (indeks 8)
            lengths = (target_params[..., 8] == 0).sum(dim=1).cpu().numpy()
            labels.append(lengths)
            
            if sum(len(b) for b in latent_vectors) >= num_samples:
                break
                
    latent_vectors = np.concatenate(latent_vectors, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]
    
    print("Obliczanie t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=latent_2d[:, 0], y=latent_2d[:, 1],
        hue=labels, palette="viridis", legend="full", alpha=0.8, s=60
    )
    plt.title("Projekcja t-SNE przestrzeni ukrytej (Model GRU)", fontsize=14)
    plt.xlabel("Wymiar 1")
    plt.ylabel("Wymiar 2")
    plt.legend(title="Liczba obiektów")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Zapisano wykres t-SNE jako {save_path}")
    else:
        plt.show()
    plt.close()