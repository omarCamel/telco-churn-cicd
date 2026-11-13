# Práctica 8 — CI/CD con GitHub Actions

Proyecto ejecutable para Windows + Visual Studio Code. El modelo incluido fue entrenado con **datos sintéticos**: sirve para aprender el flujo técnico, no para decisiones reales sobre clientes.

## 1. Ejecutar localmente (CMD de VS Code)

El modelo incluido requiere scikit-learn 1.8.0 y Python 3.11. Cree el entorno con Python 3.11 para evitar incompatibilidades al cargarlo.

```bat
py -3.11 -m venv .venv311
.venv311\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m pytest -v
python -m uvicorn app.main:app --reload
```

Abrir `http://127.0.0.1:8000/`, `/docs` y `/health`.

## 2. Probar Docker

```bat
docker build -t telco-api:v1 .
docker run --rm -p 8000:8000 --name telco-api telco-api:v1
```

Abrir `http://127.0.0.1:8000/docs`. Detener con Ctrl+C.

## 3. Publicar en GitHub

Crear un repositorio vacío llamado `telco-churn-cicd`. Sustituir `TU_USUARIO`:

```bat
git init -b main
git add .
git commit -m "Proyecto inicial CI CD"
git remote add origin https://github.com/TU_USUARIO/telco-churn-cicd.git
git push -u origin main
```

En **Actions**, observar los jobs `test` y `docker`. La publicación ocurre solo en push a `main` después de pasar las pruebas. En Pull Request se ejecutan pruebas, pero no se publica la imagen.

## 4. Provocar y corregir un fallo

En `tests/test_health.py`, cambiar temporalmente el esperado `200` por `404`. Hacer commit y push. Comprobar que `test` falla y `docker` no se ejecuta. Restaurar `200`, hacer otro commit y push; comprobar que ambos jobs terminan correctamente.

## 5. Descargar desde GHCR

La imagen publicada usa el formato `ghcr.io/tu_usuario/telco-churn-cicd:latest`, en minúsculas. Si el paquete es privado, autenticar Docker con credenciales autorizadas.

```bat
docker pull ghcr.io/tu_usuario/telco-churn-cicd:latest
docker run --rm -p 8000:8000 ghcr.io/tu_usuario/telco-churn-cicd:latest
```

## Notas

- `models/champion.joblib` contiene un pipeline completo de scikit-learn con preprocesamiento y regresión logística.
- Variables: `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`.
- El workflow implementa integración continua y publicación de imagen (Continuous Delivery), no despliegue automático a producción.
- Para producción, fijar versiones exactas de dependencias, verificar el artefacto y usar datos/modelos validados.
