import torch
import math

def differentiable_render_alpha_rgb(params, image_size=128, sharpness=300.0):
    """
    Różniczkowalny renderer 2D pracujący w paradygmacie Analysis-by-Synthesis.
    Przekształca parametry geometryczne (wektory) w wyrenderowany obraz RGB.
    
    Args:
        params: Tensor o kształcie [B, N, 9] (dla modelu sekwencyjnego) 
                lub [B, H, W, 9] (dla dekodera siatkowego).
                Format parametrów na ostatnim wymiarze: 
                [x, y, r_x, r_y, theta, R, G, B, alpha_prob]
        image_size: Rozmiar docelowego obrazka wyjściowego (H=W).
        sharpness: Skalar kontrolujący stromość funkcji Sigmoid (ostrość krawędzi elips).
    
    Returns:
        Tensor reprezentujący wyrenderowany obraz o kształcie [B, 3, image_size, image_size].
    """
    batch_size = params.size(0)
    device = params.device
    
    # Automatyczna adaptacja do modelu siatkowego (spłaszczanie [B, 16, 16, 9] -> [B, 256, 9])
    if params.dim() == 4:
        params = params.view(batch_size, -1, params.size(-1))
        
    N = params.size(1) # Liczba prymitywów w sekwencji/siatce
    
    # 1. Inicjalizacja siatki pikseli obrazu (Coordinate Grid)
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(0, 1, image_size, device=device),
        torch.linspace(0, 1, image_size, device=device),
        indexing='ij'
    )
    # Rozszerzenie wymiarów do broadacastingu: [1, 1, image_size, image_size]
    x_grid = x_grid.unsqueeze(0).unsqueeze(0)
    y_grid = y_grid.unsqueeze(0).unsqueeze(0)
    
    # 2. Ekstrakcja parametrów prymitywów do broadacastingu: [B, N, 1, 1]
    cx = params[..., 0:1].unsqueeze(-1).unsqueeze(-1)
    cy = params[..., 1:2].unsqueeze(-1).unsqueeze(-1)
    rx = params[..., 2:3].unsqueeze(-1).unsqueeze(-1)
    ry = params[..., 3:4].unsqueeze(-1).unsqueeze(-1)
    theta = params[..., 4:5].unsqueeze(-1).unsqueeze(-1) * math.pi
    
    color_r = params[..., 5:6].unsqueeze(-1).unsqueeze(-1)
    color_g = params[..., 6:7].unsqueeze(-1).unsqueeze(-1)
    color_b = params[..., 7:8].unsqueeze(-1).unsqueeze(-1)
    
    # Flaga prawdopodobieństwa istnienia obiektu (Alpha / zaufanie P)
    alpha_prob = params[..., 8:9].unsqueeze(-1).unsqueeze(-1)
    
    # Zabezpieczenie przed niestabilnością numeryczną (dzieleniem przez zero w równaniu elipsy)
    rx = torch.clamp(rx, min=1e-4)
    ry = torch.clamp(ry, min=1e-4)
    
    # 3. Transformacja do lokalnego układu współrzędnych rotowanej elipsy
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    dx = x_grid - cx
    dy = y_grid - cy
    
    # Rotacja 2D
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t
    
    # 4. Równanie dystansu od środka elipsy
    dist = (x_rot / rx)**2 + (y_rot / ry)**2
    
    # 5. Różniczkowalny antyaliasing (Rasteryzacja za pomocą Sigmoidy)
    # Wartości wewnątrz elipsy dążą do 1, na zewnątrz dążą do 0
    shape_alpha = torch.sigmoid(sharpness * (1.0 - dist))
    
    # Skalowanie maski kształtu przez prawdopodobieństwo istnienia obiektu (bramkowanie)
    final_alpha = shape_alpha * alpha_prob
    
    # 6. Kompozycja kolorów (Differentiable Alpha Blending)
    # Inicjalizacja pustego (czarnego) płótna o wymiarach [B, image_size, image_size]
    canvas_r = torch.zeros((batch_size, image_size, image_size), device=device)
    canvas_g = torch.zeros((batch_size, image_size, image_size), device=device)
    canvas_b = torch.zeros((batch_size, image_size, image_size), device=device)
    
    # Iteracyjna kompozycja obiektów (od tyłu do przodu dla poprawnego over-blendingu)
    for i in range(N):
        a_src = final_alpha[:, i, 0] # Maska i-tego prymitywu: [B, H, W]
        r_src = color_r[:, i, 0]
        g_src = color_g[:, i, 0]
        b_src = color_b[:, i, 0]
        
        # Wzór na Over Operator: C_out = C_src * A_src + C_dst * (1 - A_src)
        canvas_r = r_src * a_src + canvas_r * (1.0 - a_src)
        canvas_g = g_src * a_src + canvas_g * (1.0 - a_src)
        canvas_b = b_src * a_src + canvas_b * (1.0 - a_src)
        
    # Sklejenie kanałów do finalnego formatu obrazu NCHW
    rendered_img = torch.stack([canvas_r, canvas_g, canvas_b], dim=1)
    
    return rendered_img