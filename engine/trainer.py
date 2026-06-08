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
    criterion = OrthogonalImageLoss(dice_weight=1.0, mse_weight=1.0)
    
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
    Trójfazowa pętla ucząca dla modelu sekwencyjnego (PhysicsAutoencoder).
    Faza 1: Teacher Forcing = 1.0 (Pełne wsparcie)
    Faza 2: Liniowe wygaszanie Teacher Forcing (1.0 -> 0.0)
    Faza 3: Teacher Forcing = 0.0 (Samodzielna predykcja)
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = MaskedSequenceLoss(bce_weight=1.0, mse_weight=10.0)
    
    total_epochs = epochs_per_phase * 3
    print(f"Rozpoczynam trening 3-fazowy modelu GRU ({total_epochs} epok)")

    for epoch in range(total_epochs):
        model.train()
        total_loss = 0.0
        
        # Obliczanie obecnego poziomu Teacher Forcing
        if epoch < epochs_per_phase:
            tf_ratio = 1.0  # Faza 1
        elif epoch < 2 * epochs_per_phase:
            # Faza 2: Wygaszanie
            decay_step = epoch - epochs_per_phase
            tf_ratio = 1.0 - (decay_step / epochs_per_phase)
        else:
            tf_ratio = 0.0  # Faza 3
            
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
        
        # Wyświetlanie informacji o fazie
        phase = 1 if epoch < epochs_per_phase else (2 if epoch < 2*epochs_per_phase else 3)
        print(f"Epoka [{epoch+1}/{total_epochs}] | Faza: {phase} | TF Ratio: {tf_ratio:.2f} | Loss: {avg_loss:.4f}")

    return model