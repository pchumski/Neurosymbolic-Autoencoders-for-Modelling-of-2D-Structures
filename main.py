import argparse
import torch
from torch.utils.data import DataLoader

# Importy autorskich modułów
from utils.dataset import GeometricDataset, BifurcationDataset12, HardcoreUSGDataset
from models.networks import PhysicsAutoencoder, GridNeurosymbolicAutoencoder16RGB
from engine.trainer import train_grid_model, train_3phase
from utils.visualization import visualize_clean

def parse_args():
    parser = argparse.ArgumentParser(description="Neurosymbolic Autoencoder dla struktur sekwencyjnych 2D")
    
    parser.add_argument('--model-type', type=str, choices=['gru', 'grid'], required=True,
                        help='Wybór wariantu architektonicznego dekodera.')
    parser.add_argument('--dataset-type', type=str, choices=['geometric', 'bifurcation'], default='geometric',
                        help='Wybór generatora danych (proste węże vs złożone rozgałęzienia).')
    parser.add_argument('--grid-size', type=int, default=16, 
                        help='Rozmiar siatki przestrzennej (domyślnie 16).')
    parser.add_argument('--noise-severity', type=float, default=0.0, 
                        help='Poziom degradacji obrazu wejściowego (0.0 - 1.0).')
    parser.add_argument('--epochs', type=int, default=60, 
                        help='Liczba epok treningowych.')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Rozmiar paczki danych (batch size).')
    parser.add_argument('--lr', type=float, default=2e-4, 
                        help='Współczynnik uczenia (learning rate).')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', 
                        help='Wybór akceleratora obliczeniowego.')
    parser.add_argument('--num-samples', type=int, default=10000, 
                        help='Liczba wygenerowanych w locie próbek treningowych.')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Konfiguracja sprzętowa
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"--- Inicjalizacja środowiska na: {device.type.upper()} ---")
    
    # 2. Generowanie zbioru danych (w locie)
    print(f"Generowanie {args.num_samples} próbek treningowych ({args.dataset_type})...")
    
    if args.dataset_type == 'geometric':
        base_dataset = GeometricDataset(num_samples=args.num_samples, img_size=128)
    elif args.dataset_type == 'bifurcation':
        base_dataset = BifurcationDataset12(num_samples=args.num_samples, img_size=128)
    
    if args.noise_severity > 0.0:
        print(f"Nakładanie degradacji środowiskowych (Severity: {args.noise_severity})")
        dataset = HardcoreUSGDataset(base_dataset, severity=args.noise_severity)
    else:
        dataset = base_dataset
        
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # 3. Wybór architektury i trening
    if args.model_type == 'grid':
        print(f"Uruchamianie: Przestrzenny Dekoder Siatkowy ({args.grid_size}x{args.grid_size})")
        model = GridNeurosymbolicAutoencoder16RGB(image_size=128, grid_size=args.grid_size)
        
        trained_model = train_grid_model(
            model, train_loader, epochs=args.epochs, lr=args.lr, device=device
        )
        
    elif args.model_type == 'gru':
        print("Uruchamianie: Model Sekwencyjny (GRU)")
        model = PhysicsAutoencoder(latent_dim=128, hidden_dim=256, max_seq_len=10)
        
        # Podział epok na 3 równe fazy dla harmonogramu Teacher Forcing
        epochs_per_phase = max(1, args.epochs // 3)
        trained_model = train_3phase(
            model, train_loader, epochs_per_phase=epochs_per_phase, lr=args.lr, device=device
        )

    # 4. Zapisanie wag modelu do pliku .pth
    save_path = f"neurosymbolic_{args.model_type}_model.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"✅ Trening zakończony sukcesem. Wagi zapisano w: {save_path}")
    
    # 5. Wygenerowanie podglądu wyników (Test)
    print("Generowanie wizualizacji predykcji...")
    visualize_clean(
        trained_model, dataset, device=device, num_samples=3, 
        save_path=f"test_results_{args.model_type}.png"
    )

if __name__ == '__main__':
    main()