import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. BŁĄD TOPOLOGII (DICE LOSS)
# ==========================================
def dice_loss(pred, target, smooth=1e-5):
    """
    Funkcja straty oparta na współczynniku Dice'a (Sørensena-Dice’a).
    Idealna do optymalizacji topologii i zwalczania problemu niezbalansowanych klas 
    (np. czarne tło vs cienkie struktury wężowe).
    
    Args:
        pred: Tensor predykcji [B, C, H, W] (wartości po aktywacji Sigmoid, w przedziale [0,1]).
        target: Tensor referencyjny [B, C, H, W].
        smooth: Współczynnik wygładzający zapobiegający dzieleniu przez zero.
    """
    # Spłaszczenie wymiarów przestrzennych
    pred_flat = pred.contiguous().view(pred.size(0), -1)
    target_flat = target.contiguous().view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    
    # Zwracamy błąd (1 - Dice), uśredniony dla całego batcha
    return 1.0 - dice_score.mean()


# ==========================================
# 2. ORTOGONALNA STRATA REKONSTRUKCJI OBRAZU
# ==========================================
class OrthogonalImageLoss(nn.Module):
    """
    Kluczowa innowacja z pracy magisterskiej: 
    Rozdziela karę za błędną topologię (Dice) od kary za błędny kolor (MSE).
    Zapobiega zjawisku Zero-Collapse (model generujący czarne tło).
    """
    def __init__(self, dice_weight=1.0, mse_weight=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.mse_weight = mse_weight
        self.mse_criterion = nn.MSELoss()

    def forward(self, pred_img, target_img):
        # 1. Ekstrakcja "kształtu" (zamiana RGB na maskę jasności/grayscale)
        # Służy do analizy czysto topologicznej
        pred_mask = pred_img.mean(dim=1, keepdim=True)
        target_mask = target_img.mean(dim=1, keepdim=True)
        
        # 2. Błąd topologii (Dice) - wymusza poprawne kształty niezależnie od koloru
        loss_topo = dice_loss(pred_mask, target_mask)
        
        # 3. Błąd fotometrii (MSE) - dopasowuje kolory RGB
        loss_color = self.mse_criterion(pred_img, target_img)
        
        # Całkowity błąd (Analysis-by-Synthesis Loss)
        total_loss = (self.dice_weight * loss_topo) + (self.mse_weight * loss_color)
        
        return total_loss, loss_topo, loss_color


# ==========================================
# 3. MASKOWANA STRATA DLA PARAMETRÓW (GRU)
# ==========================================
class MaskedSequenceLoss(nn.Module):
    """
    Funkcja straty dedykowana modelowi sekwencyjnemu (GRU).
    Oblicza błąd tylko dla rzeczywistych elementów węża, ignorując padding (maskowanie).
    """
    def __init__(self, bce_weight=1.0, mse_weight=10.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight

    def forward(self, pred_params, target_params):
        """
        Args:
            pred_params: [B, seq_len, 9] (0-7: geometria, 8: flaga P/EOS)
            target_params: [B, seq_len, 9] (0: aktywny, 1: padding/EOS w indeksie 8)
        """
        # Indeks 8 to flaga EOS (Zakończenie sekwencji)
        pred_eos = pred_params[..., 8]
        target_eos = target_params[..., 8]
        
        # 1. Błąd predykcji długości węża (BCE Loss na fladze EOS)
        # Tu uczymy model zjawiska problemu stopu
        loss_eos = F.binary_cross_entropy(pred_eos, target_eos)
        
        # 2. Maskowanie paddingu (Interesują nas tylko aktywne obiekty)
        # target_eos == 0 oznacza, że obiekt jest prawdziwy
        mask = (target_eos == 0).float() 
        
        # 3. Błąd regresji parametrów geometrycznych i RGB (MSE)
        pred_geom = pred_params[..., :8]
        target_geom = target_params[..., :8]
        
        # Obliczamy MSE bazowe bez redukcji
        loss_geom_raw = F.mse_loss(pred_geom, target_geom, reduction='none')
        
        # Aplikujemy maskę i uśredniamy tylko po aktywnych elementach
        loss_geom_masked = (loss_geom_raw.mean(dim=-1) * mask).sum() / (mask.sum() + 1e-8)
        
        total_loss = (self.bce_weight * loss_eos) + (self.mse_weight * loss_geom_masked)
        
        return total_loss, loss_eos, loss_geom_masked