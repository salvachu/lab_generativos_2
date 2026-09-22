# Quick start

Requisitos: Windows, PowerShell y Python 3.11 o 3.12.

## 1. Clone

```powershell
git clone https://github.com/salvachu/lab_generativos_2.git
cd lab_generativos_2
```

## 2. Setup

El entorno reproducido usa PyTorch 2.7.1 con CUDA 12.6:

```powershell
.\scripts\setup.ps1 -Python python
```

En CPU, instala la distribución de PyTorch apropiada y después:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

`train.pkl`, `val.pkl` y la calibración geométrica ya están incluidos.

## 3. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 4. Train

Ejemplo con C; cambia `C` por A/B/D/E y usa su config correspondiente:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --config configs\C.json --models C --output runs\comparison_C
```

## 5. Evaluate

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate runs\comparison_C\C\best.pt --output runs\evaluation_C --count 8 --samples 3 --seed 2026
```

## 6. Random generation

```powershell
.\.venv\Scripts\python.exe -m scripts.generate runs\comparison_C\C\best.pt --output runs\random_C --candidates 20 --top-k 6
```

## 7. Completion

```powershell
.\.venv\Scripts\python.exe -m scripts.generate runs\comparison_C\C\best.pt --prefix configs\prefix_example.json --output runs\completion_C --candidates 20 --top-k 6
```

## 8. UI

```powershell
.\scripts\run_demo.ps1
```

Abre <http://127.0.0.1:8000>. La interfaz descubre checkpoints `best.pt`, `last.pt` o `checkpoint.pt` dentro de `runs/`. Azul es el prefix; naranja es la continuación.
