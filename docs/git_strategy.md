# Estrategia de Git

Este documento describe cómo se administra el control de versiones del proyecto.

## Ramas

- **`main`**: rama de producción. Contiene únicamente versiones estables del
  proyecto. Cada entrega se materializa aquí vía pull request y se marca con un
  *tag* (ej. `v1.0.0`).
- **`development`**: rama de integración. Todo el trabajo de desarrollo se
  integra aquí a través de pull requests desde ramas de funcionalidad.
- **`feature/*`**: ramas temporales creadas a partir de `development` para cada
  tarea o fase (ej. `feature/setup`, `feature/eda`, `feature/mlflow`). Se eliminan
  tras fusionar.

## Flujo de trabajo

Modelo basado en GitFlow simplificado (rama `main` + `development` + ramas
`feature/*`), adaptado de GitHub Flow:

1. Crear rama `feature/*` desde `development`.
2. Desarrollar con commits atómicos y mensajes descriptivos.
3. Abrir una **pull request** hacia `development`, revisarla y cerrarla con merge.
4. Cuando una fase está lista, abrir PR de `development` hacia `main` y revisar.
5. Antes de cada entrega, crear un **release** en GitHub con su tag y notas.

## Convención de commits

Mensajes descriptivos en español, siguiendo el estilo convencional:

- `feat: ...` (nueva funcionalidad)
- `fix: ...` (corrección)
- `docs: ...` (documentación)
- `refactor: ...` (refactorización)
- `chore: ...` (tareas de mantenimiento)

Ejemplo: `feat: agrega módulo de construcción de features`.

## Pull requests

Toda PR debe describir:
- Qué cambio se hizo y por qué.
- Resultados/evidencia cuando aplique (ej. métricas, capturas de MLflow).
- Vínculo con la fase del proyecto (ver `PROYECTO.md`).

## Releases

- Los releases se crean desde la rama `main` con tag semver (`v1.0.0`).
- Las notas de release resumen los cambios y evidencias de cada fase.
