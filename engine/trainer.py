import torch
import torch.nn as nn
import torch.optim as optim
from core.losses import OrthogonalImageLoss, MaskedSequenceLoss
from core.renderer import differentiable_render_alpha_rgb

# ==========================================
# 1. PĘTLA UCZĄCA DLA DEKODERA SIATKOWEGO
# ==========================================
def train_grid_model(model, train_loader, epochs=60, lr=2e-4, device='cuda'):
    """
    Standardowa pętla ucząca dla modelu GridNeurosymbolicAutoencoder.
    Wykorzystuje paradygmat Analysis-by-Synthesis (ortogonalna funkcja straty).
    """
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # λ_Dice=0.8 wymusza priorytet nauki topologii, λ_MSE=0.2 dopasowuje kolor
    criterion = OrthogonalImageLoss(dice_weight=0.8, mse_weight=0.2)
    
    print(f"Rozpoczynam trening dekodera siatkowego na urządzeniu: {device}")
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch_idx, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            
            # 1. Predykcja parametrów (Forward pass)
            pred_params = model(imgs)
            
            # 2. Renderowanie różniczkowalne
            pred_imgs = differentiable_render_alpha_rgb(pred_params, image_size=imgs.size(2))
            
            # 3. Obliczenie ortogonalnej funkcji straty
            loss, loss_topo, loss_color = criterion(pred_imgs, imgs)
            
            # 4. Wsteczna propagacja
            loss.backward()
            
            # 5. Przycinanie gradientów (Gradient Clipping)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 6. Krok optymalizatora
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Pobranie wyuczalnych mnożników siatki (jeśli istnieją)
        scale_rx = getattr(model, 'scale_rx', torch.tensor(1.0)).item()
        scale_ry = getattr(model, 'scale_ry', torch.tensor(1.0)).item()
        
        print(f"Epoka [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Mnożnik X: {abs(scale_rx):.2f} | Mnożnik Y: {abs(scale_ry):.2f}")

    return model


# ==========================================
# 2. TRÓJFAZOWA PĘTLA UCZĄCA DLA MODELU GRU
# ==========================================
def train_3phase(model, train_loader, epochs_per_phase=20, lr=1e-3, device='cuda'):
    """
    Pętla ucząca dla modelu sekwencyjnego (PhysicsAutoencoder).
    Harmonogram wymuszania nauczyciela (Teacher Forcing) zgodny z równaniem 5.1 pracy:

        ratio = max(0.1, 1.0 - epoch / 40.0)

    TF spada liniowo od 1.0 do wartości minimalnej 0.1, osiągając minimum
    około epoki 36. Od tej chwili model pracuje prawie w pełni samodzielnie.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = MaskedSequenceLoss(bce_weight=1.0, mse_weight=10.0)
    
    total_epochs = epochs_per_phase * 3
    print(f"Rozpoczynam trening modelu GRU ({total_epochs} epok, harmonogram TF wg równania 5.1)")

    for epoch in range(total_epochs):
        model.train()
        total_loss = 0.0
        
        # Harmonogram wygaszania Teacher Forcing – równanie 5.1 pracy
        tf_ratio = max(0.1, 1.0 - epoch / 40.0)
            
        for batch_idx, (imgs, target_params) in enumerate(train_loader):
            imgs = imgs.to(device)
            target_params = target_params.to(device)
            
            # Forward pass z wymuszaniem nauczyciela
            pred_params = model(imgs, target_seq=target_params, teacher_forcing_ratio=tf_ratio)
            
            # Obliczenie maskowanej straty sekwencji
            loss, loss_eos, loss_geom = criterion(pred_params, target_params)
            
            # Wsteczna propagacja i optymalizacja
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoka [{epoch+1}/{total_epochs}] | TF Ratio: {tf_ratio:.2f} | Loss: {avg_loss:.4f}")

    return model