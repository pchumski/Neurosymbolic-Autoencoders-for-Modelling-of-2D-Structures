import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import random
import math

class GeometricDataset(Dataset):
    """
    Podstawowy dataset generujący proste sekwencyjne struktury 2D (węże) 
    z wykorzystaniem błądzenia losowego (Random Walk).
    """
    def __init__(self, num_samples=10000, img_size=128, min_n=3, max_n=10, 
                 r_min=0.03, r_max=0.08, color_mode='fade'):
        self.num_samples = num_samples
        self.img_size = img_size
        self.min_n = min_n
        self.max_n = max_n
        self.r_min = r_min
        self.r_max = r_max
        self.color_mode = color_mode

    def __len__(self):
        return self.num_samples

    def _generate_path(self):
        N = random.randint(self.min_n, self.max_n)
        params = np.zeros((self.max_n, 9), dtype=np.float32)
        
        # Inicjalizacja w centrum z lekkim przesunięciem
        x, y = 0.5 + random.uniform(-0.1, 0.1), 0.5 + random.uniform(-0.1, 0.1)
        angle = random.uniform(0, 2 * math.pi)
        
        base_r = random.uniform(0.1, 0.9)
        base_g = random.uniform(0.1, 0.9)
        base_b = random.uniform(0.1, 0.9)

        for i in range(N):
            r_maj = random.uniform(self.r_min, self.r_max)
            r_min = r_maj * random.uniform(0.8, 1.0)
            
            if self.color_mode == 'fade':
                # Symulacja gradientu (fade)
                base_r = np.clip(base_r + random.uniform(-0.1, 0.1), 0, 1)
                base_g = np.clip(base_g + random.uniform(-0.1, 0.1), 0, 1)
                base_b = np.clip(base_b + random.uniform(-0.1, 0.1), 0, 1)
            
            # EOS flag: 0 oznacza aktywny obiekt, 1 oznacza padding/koniec
            eos = 0.0 if i < N - 1 else 1.0
            
            params[i] = [x, y, r_maj, r_min, angle / (2*math.pi), base_r, base_g, base_b, eos]
            
            # Krok i momentum (krętość)
            step_size = r_maj * 1.2
            angle += random.uniform(-0.5, 0.5)
            x += math.cos(angle) * step_size
            y += math.sin(angle) * step_size
            
            # Odbicie od krawędzi (Collision avoidance z krawędziami)
            if x < 0.1 or x > 0.9 or y < 0.1 or y > 0.9:
                angle += math.pi
                x += math.cos(angle) * step_size * 2
                y += math.sin(angle) * step_size * 2

        # Padding (resztę wypełniamy flagą EOS = 1.0)
        for i in range(N, self.max_n):
            params[i, 8] = 1.0
            
        return params

    def _render(self, params):
        img = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
        draw = ImageDraw.Draw(img, 'RGBA')
        
        for p in params:
            if p[8] == 1.0 and p[2] == 0: # Jeśli to czysty padding
                continue
            
            cx, cy = p[0] * self.img_size, p[1] * self.img_size
            rx, ry = p[2] * self.img_size, p[3] * self.img_size
            
            r, g, b = int(p[5]*255), int(p[6]*255), int(p[7]*255)
            
            draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=(r, g, b, 255))
            
        return np.array(img).transpose(2, 0, 1) / 255.0

    def __getitem__(self, idx):
        params = self._generate_path()
        img = self._render(params)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(params, dtype=torch.float32)


class BifurcationDataset12(GeometricDataset):
    """
    Zaawansowany dataset generujący rozgałęzienia, bifurkacje typu T/Y 
    oraz skrzyżowania typu X (Pączkująca mrówka).
    Wykorzystuje mechanizm Rejection Sampling.
    """
    def __init__(self, branch_prob=0.3, max_branches=3, **kwargs):
        super().__init__(**kwargs)
        self.branch_prob = branch_prob
        self.max_branches = max_branches

    def _generate_path(self):
        # Inicjalizacja z Rejection Samplingiem (Odrzucanie kolizji)
        params = np.zeros((self.max_n, 9), dtype=np.float32)
        active_branches = [(0.5, 0.5, random.uniform(0, 2*math.pi))]
        
        idx = 0
        while idx < self.max_n and active_branches:
            x, y, angle = active_branches.pop(0)
            
            r_maj = random.uniform(self.r_min, self.r_max)
            eos = 0.0 if idx < self.max_n - 1 else 1.0
            
            # Przypisz parametry
            params[idx] = [x, y, r_maj, r_maj*0.9, angle/(2*math.pi), 
                           random.uniform(0,1), random.uniform(0,1), random.uniform(0,1), eos]
            idx += 1
            
            # Bifurkacja
            if random.random() < self.branch_prob and len(active_branches) < self.max_branches:
                # Tworzymy nową gałąź
                new_angle = angle + random.uniform(math.pi/4, math.pi/2)
                active_branches.append((x + math.cos(new_angle)*r_maj*2, 
                                        y + math.sin(new_angle)*r_maj*2, new_angle))
            
            # Kontynuacja obecnej gałęzi
            if idx < self.max_n:
                angle += random.uniform(-0.3, 0.3)
                active_branches.append((x + math.cos(angle)*r_maj*1.5, 
                                        y + math.sin(angle)*r_maj*1.5, angle))
                
        # Wypełnij resztę paddingiem
        for i in range(idx, self.max_n):
            params[i, 8] = 1.0
            
        return params


class HardcoreUSGDataset(Dataset):
    """
    Dataset symulujący warunki medyczne (USG / Angiografia).
    Nakłada ekstremalne zakłócenia: Gaussian Noise, Gaussian Blur, Dropout.
    Wspiera parametr severity (0.0 - 1.0) do programowanego uczenia (Curriculum Learning).
    """
    def __init__(self, base_dataset, severity=0.7):
        self.base_dataset = base_dataset
        self.severity = severity

    def __len__(self):
        return len(self.base_dataset)

    def _apply_degradations(self, img_tensor):
        img_np = (img_tensor.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        
        if self.severity > 0:
            # 1. Rozmycie optyczne (Gaussian Blur)
            blur_kernel = max(1, int(7 * self.severity))
            if blur_kernel % 2 == 0: blur_kernel += 1
            img_np = cv2.GaussianBlur(img_np, (blur_kernel, blur_kernel), 0)
            
            # 2. Szum sensorowy (Gaussian Noise)
            noise_sigma = 50 * self.severity
            noise = np.random.normal(0, noise_sigma, img_np.shape).astype(np.float32)
            img_np = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            # 3. Utrata pikseli (Dropout / Salt & Pepper)
            dropout_prob = 0.15 * self.severity
            mask = np.random.rand(*img_np.shape[:2]) < dropout_prob
            img_np[mask] = [0, 0, 0] # Zastąpienie czarnymi pikselami
            
        return torch.tensor(img_np.transpose(2, 0, 1) / 255.0, dtype=torch.float32)

    def __getitem__(self, idx):
        clean_img, params = self.base_dataset[idx]
        noisy_img = self._apply_degradations(clean_img)
        return noisy_img, params