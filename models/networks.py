import torch
import torch.nn as nn
import random

# ==========================================
# 1. WSPÓLNY MODUŁ PERCEPCYJNY (ENKODER CNN)
# ==========================================
class CNNEncoder(nn.Module):
    """
    Splotowy moduł percepcyjny zgodny z Tabelą 4.1.
    Kompresuje obraz wejściowy 128x128x3 do wektora kontekstu C (128 wymiarów).
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
            nn.Tanh() # Normalizacja wektora ukrytego do [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2. MODEL SEKWENCYJNY (GRU)
# ==========================================
class PhysicsAutoencoder(nn.Module):
    """
    Rekurencyjny modeler generujący sekwencyjne struktury 2D (węże).
    Wykorzystuje 2-warstwowe GRU oraz mechanizm integracji kontekstu.
    """
    def __init__(self, latent_dim=128, hidden_dim=256, max_seq_len=10):
        super().__init__()
        self.encoder = CNNEncoder(latent_dim)
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Warstwa fuzji kontekstu (Konkatenacja y_{t-1} oraz C)
        self.fusion = nn.Sequential(
            nn.Linear(9 + latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim)
        )
        
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # Dedykowane głowice wyjściowe (Ruch i Parametry geometryczne)
        self.head_move = nn.Linear(hidden_dim, 2)
        self.head_params = nn.Linear(hidden_dim, 7)

    def forward(self, x, target_seq=None, teacher_forcing_ratio=0.0):
        batch_size = x.size(0)
        device = x.device
        
        # 1. Ekstrakcja wektora ukrytego C
        C = self.encoder(x)
        
        # 2. Inicjalizacja stanu ukrytego GRU
        h = torch.zeros(2, batch_size, self.hidden_dim, device=device)
        
        # 3. Inicjalizacja pustego wektora wejściowego dla pierwszego kroku
        current_y = torch.zeros(batch_size, 9, device=device)
        
        outputs = []
        for t in range(self.max_seq_len):
            # Mechanizm wymuszania nauczyciela (Teacher Forcing)
            if target_seq is not None and t > 0 and random.random() < teacher_forcing_ratio:
                current_y = target_seq[:, t-1, :]
                
            # Fuzja kontekstu
            fused_input = self.fusion(torch.cat([current_y, C], dim=-1)).unsqueeze(1)
            
            # Krok GRU
            out, h = self.gru(fused_input, h)
            out = out.squeeze(1)
            
            # Głowice wyjściowe
            move = torch.tanh(self.head_move(out)) * 0.5   # Ograniczenie kroku dx, dy
            params = torch.sigmoid(self.head_params(out))  # Promienie, rotacja, RGB, EOS w [0, 1]
            
            current_y = torch.cat([move, params], dim=-1)
            outputs.append(current_y.unsqueeze(1))
            
        return torch.cat(outputs, dim=1)


# ==========================================
# 3. DEKODER SIATKOWY (GRID / YOLO-style)
# ==========================================
class GridNeurosymbolicAutoencoder16RGB(nn.Module):
    """
    Przestrzenny dekoder siatkowy (16x16) rozwiązujący problem topologii złożonych.
    Zwraca równoległą predykcję 256 elementów. Zawiera wyuczalne parametry skalowania.
    """
    def __init__(self, image_size=128, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        
        # Enkoder przestrzenny (zachowuje wymiary dwuwymiarowe, brak spłaszczania)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Ostatnia warstwa dostosowująca kanały do 256, utrzymująca siatkę 16x16
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Warstwa dekodująca (1x1 Conv) - mapowanie 256 kanałów na 9 atrybutów
        self.decoder = nn.Conv2d(256, 9, kernel_size=1)
        
        # Globalne, wyuczalne parametry skalowania (S_rx, S_ry) ułatwiające domykanie topologii
        self.scale_rx = nn.Parameter(torch.tensor(1.0))
        self.scale_ry = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # 1. Ekstrakcja cech przestrzennych
        features = self.encoder(x) # [B, 256, 16, 16]
        
        # Bezpiecznik wymiarów dla różnych wejść obrazu
        if features.shape[-1] != self.grid_size:
            features = torch.nn.functional.interpolate(
                features, size=(self.grid_size, self.grid_size), mode='bilinear', align_corners=False
            )
            
        # 2. Dekodowanie parametrów w każdej komórce
        out = self.decoder(features)     # [B, 9, 16, 16]
        out = out.permute(0, 2, 3, 1)    # [B, 16, 16, 9] - Format przestrzenny
        
        # 3. Aplikacja funkcji aktywacji Sigmoid do wszystkich wyjść
        # Indeksy: 0:P, 1:dx, 2:dy, 3:rx, 4:ry, 5:theta, 6:R, 7:G, 8:B
        p = torch.sigmoid(out[..., 0:1])
        dx_dy = torch.sigmoid(out[..., 1:3])
        rx_ry = torch.sigmoid(out[..., 3:5])
        theta_rgb = torch.sigmoid(out[..., 5:9])
        
        return torch.cat([p, dx_dy, rx_ry, theta_rgb], dim=-1)