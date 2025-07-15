import torch
import time

def test_gpu():
    # Verifica se è disponibile almeno una GPU
    if not torch.cuda.is_available():
        print("GPU non trovata. Il test verrà eseguito su CPU.")
        device = torch.device("cpu")
    else:
        print(f"GPU trovata: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda:0")

    # Crea due matrici di grandi dimensioni
    size = 5000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Forza la sincronizzazione prima della misurazione
    torch.cuda.synchronize() if device.type == "cuda" else None

    # Misura il tempo di moltiplicazione
    start = time.time()
    c = torch.mm(a, b)
    # Sincronizza di nuovo per attendere la fine dell’operazione GPU
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.time()

    print(f"Tempo impiegato per mm ({size}×{size}) su {device}: {end - start:.4f} secondi")

if __name__ == "__main__":
    test_gpu()
