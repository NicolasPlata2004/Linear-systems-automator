#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resolver_sistemas_detallado.py
Resuelve Ax=b (3x3..10x10) por:
  - Gauss-Jordan (pivoteo parcial)
  - Gauss-Seidel (iterativo)
  - Regla de Cramer

Incluye "modo detallado" para imprimir los PASOS del método.

Entradas permitidas para A y b:
  (1) Pegado rápido (bloque)  [acepta estilo MATLAB: [ ... ; ... ]]
  (2) Paso a paso (fila por fila)
  (3) Generar ejemplo aleatorio (A diagonalmente dominante, b coherente)

Requiere: NumPy
"""
from __future__ import annotations
import re
from typing import Tuple, Optional, List
import numpy as np

# ------------------ Utilidades de I/O y formateo ------------------

def menu(titulo: str, opciones: List[str]) -> int:
    print("\n" + "="*70)
    print(titulo)
    print("="*70)
    for i, op in enumerate(opciones, 1):
        print(f"{i}) {op}")
    while True:
        s = input("Elige una opción (número): ").strip()
        if s.isdigit():
            i = int(s)
            if 1 <= i <= len(opciones):
                return i
        print("Entrada inválida. Intenta de nuevo.")

def leer_entero_en_rango(prompt: str, vmin: int, vmax: int) -> int:
    while True:
        s = input(prompt).strip()
        if s.isdigit():
            v = int(s)
            if vmin <= v <= vmax:
                return v
        print(f"Por favor ingresa un entero entre {vmin} y {vmax}.")

def _split_numbers(line: str) -> List[float]:
    parts = re.split(r"[,\s]+", line.strip())
    parts = [p for p in parts if p != ""]
    return [float(p) for p in parts]

def parse_block_matrix(text: str, n: int, m: int) -> np.ndarray:
    # Limpia estilo MATLAB: quita corchetes y usa ';' como salto de línea
    text = re.sub(r"[\[\]]", "", text)
    text = text.replace(";", "\n")
    lines = [ln for ln in text.strip().splitlines() if ln.strip() != ""]
    if len(lines) != n:
        raise ValueError(f"Se esperaban {n} filas, se obtuvieron {len(lines)}.")
    rows = []
    for li, ln in enumerate(lines, 1):
        nums = _split_numbers(ln)
        if len(nums) != m:
            raise ValueError(f"Fila {li}: se esperaban {m} números y hay {len(nums)}.")
        rows.append(nums)
    return np.array(rows, dtype=float)

def leer_matriz_pegado(n: int, m: int, nombre: str) -> np.ndarray:
    print("\nPega la matriz/vector en bloque.")
    print(f"- {nombre} debe ser de tamaño {n}x{m}.")
    print("- Una fila por línea (espacios o comas). Puedes usar ';' para separar filas.")
    print("- También puedes pegar formato MATLAB con corchetes: [a11 a12; a21 a22].")
    print("Finaliza con una línea vacía (ENTER dos veces).")
    buf = []
    while True:
        ln = input()
        if ln.strip() == "" and buf:
            break
        buf.append(ln)
    text = "\n".join(buf)
    return parse_block_matrix(text, n, m)

def leer_matriz_paso_a_paso(n: int, m: int, nombre: str) -> np.ndarray:
    print(f"\nIngresar {nombre} fila por fila ({n} filas, {m} columnas).")
    print("Escribe los valores separados por espacios o comas.\n")
    M = np.zeros((n, m), dtype=float)
    for i in range(n):
        while True:
            linea = input(f"Fila {i+1}: ")
            try:
                vals = _split_numbers(linea)
                if len(vals) != m:
                    print(f"Se requieren {m} números. Intentar de nuevo.")
                    continue
                M[i, :] = vals
                mostrar_matriz(M[:i+1, :], f"{nombre} (parcial)")
                break
            except Exception as e:
                print("Error:", e)
    return M

def mostrar_matriz(M: np.ndarray, titulo: str):
    print("\n---", titulo, "---")
    with np.printoptions(precision=6, suppress=True):
        print(M)
    print("--- fin", titulo, "---\n")

def mostrar_aumentada(A: np.ndarray, b: np.ndarray, titulo: str):
    Ab = np.hstack([A, b.reshape(-1,1)])
    mostrar_matriz(Ab, f"[A | b] {titulo}")

def generar_ejemplo(n: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    A = rng.uniform(-5, 5, size=(n, n))
    # Forzar dominancia diagonal
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + rng.uniform(1, 3)
    x_true = rng.uniform(-3, 3, size=n)
    b = A @ x_true
    return A, b

def elegir_modo_entrada(n: int, m: int, nombre: str, permite_ejemplo: bool = True) -> np.ndarray:
    # SOLO 1) Pegado, 2) Paso a paso, 3) Generar ejemplo
    opciones = ["Pegado rápido (bloque de texto)", "Paso a paso (fila por fila)"]
    if permite_ejemplo:
        opciones.append("Generar ejemplo aleatorio")
    idx = menu(f"¿Cómo quieres introducir {nombre}?", opciones)
    if idx == 1:
        return leer_matriz_pegado(n, m, nombre)
    elif idx == 2:
        return leer_matriz_paso_a_paso(n, m, nombre)
    elif permite_ejemplo and idx == 3:
        if m == n:
            M, _b_dummy = generar_ejemplo(n)
            print(f"{nombre} aleatoria diagonalmente dominante generada.")
            mostrar_matriz(M, nombre)
            return M
        else:
            v = np.random.default_rng().uniform(-5,5,size=(n,))
            print(f"{nombre} aleatorio generado.")
            mostrar_matriz(v.reshape(-1,1), nombre)
            return v
    else:
        raise RuntimeError("Selección inválida.")

# ------------------ Métodos con modo detallado ------------------

def gauss_jordan(A: np.ndarray, b: np.ndarray, pivoteo: bool = True, detallado: bool = False):
    A = A.astype(float).copy()
    b = b.astype(float).copy().reshape(-1,)
    n = A.shape[0]
    Ab = np.hstack([A, b.reshape(-1,1)])
    info = {"success": False, "message": "", "iter": None, "advice": ""}

    if detallado:
        mostrar_aumentada(A, b, "inicial")

    tol = 1e-14
    for k in range(n):
        if detallado:
            print(f"Paso {k+1}/{n}: pivote en columna {k+1}")
        # Pivoteo parcial
        if pivoteo:
            p = k + np.argmax(np.abs(Ab[k:, k]))
            if abs(Ab[p, k]) < tol:
                info["message"] = "Matriz singular o casi singular (pivote≈0)."
                return np.full(n, np.nan), info
            if p != k:
                if detallado:
                    print(f"  Intercambio de filas: F{k+1} <-> F{p+1}")
                Ab[[k, p], :] = Ab[[p, k], :]
        else:
            if abs(Ab[k, k]) < tol:
                info["message"] = "Pivote nulo sin pivoteo; abortando."
                return np.full(n, np.nan), info

        # Normalizar fila k
        piv = Ab[k, k]
        Ab[k, :] = Ab[k, :] / piv
        if detallado:
            print(f"  Normalizar F{k+1} dividida por {piv:.6g}")
            mostrar_matriz(Ab, f"tras normalizar F{k+1}")

        # Eliminar columna k en otras filas
        for i in range(n):
            if i != k:
                factor = Ab[i, k]
                if abs(factor) > 0:
                    Ab[i, :] -= factor * Ab[k, :]
                    if detallado:
                        print(f"  F{i+1} <- F{i+1} - ({factor:.6g})*F{k+1}")
        if detallado:
            mostrar_matriz(Ab, f"columna {k+1} anulada")

    x = Ab[:, -1]
    info["success"] = True
    if detallado:
        print("Matriz reducida [I | x] alcanzada.")
        mostrar_matriz(Ab, "final")
    return x, info

def gauss_seidel(A: np.ndarray, b: np.ndarray, tol: float = 1e-8, maxit: int = 1000,
                 x0: Optional[np.ndarray] = None, detallado: bool = False):
    A = A.astype(float)
    b = b.astype(float).reshape(-1,)
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float).reshape(-1,)
    info = {"success": False, "message": "", "iter": None, "advice": ""}

    diag = np.abs(np.diag(A))
    if np.any(diag < np.finfo(float).eps):
        info["advice"] += "Hay elementos ~0 en la diagonal; puede fallar.\n"
    dd = np.all(diag >= np.sum(np.abs(A), axis=1) - diag)
    if not dd:
        info["advice"] += "La matriz NO es diagonalmente dominante; podría no converger.\n"

    if detallado:
        print("\n== Gauss-Seidel: inicio ==")
        mostrar_matriz(A, "A")
        mostrar_matriz(b.reshape(-1,1), "b")
        mostrar_matriz(x.reshape(-1,1), "x^(0)")

    for k in range(1, maxit+1):
        x_old = x.copy()
        if detallado:
            print(f"\nIteración {k}:")
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x_old[i+1:])
            if A[i, i] == 0:
                info["message"] = f"A[{i},{i}]=0; imposible actualizar."
                if detallado:
                    print(info["message"])
                return x, info
            nuevo = (b[i] - s1 - s2) / A[i, i]
            if detallado:
                # expresar la fórmula
                print(f"  x_{i+1} = (b[{i+1}] - sum(A[{i+1},<i]*x_nuevos) - sum(A[{i+1},>i]*x_prev)) / A[{i+1},{i+1}]")
                print(f"       = ({b[i]:.6g} - {s1:.6g} - {s2:.6g}) / {A[i,i]:.6g} = {nuevo:.9g}")
            x[i] = nuevo

        err = np.linalg.norm(x - x_old, ord=np.inf)
        if detallado:
            mostrar_matriz(x.reshape(-1,1), f"x^({k})")
            print(f"  ||x^(k) - x^(k-1)||_inf = {err:.6e}")

        if err <= tol * (1 + np.linalg.norm(x, ord=np.inf)):
            info["success"] = True
            info["iter"] = k
            if detallado:
                print(f"Convergencia alcanzada con tol={tol:g}.")
            return x, info

    info["message"] = f"No convergió en {maxit} iteraciones (tol={tol})."
    if detallado:
        print(info["message"])
    return x, info

def cramer(A: np.ndarray, b: np.ndarray, detallado: bool = False):
    A = A.astype(float)
    b = b.astype(float).reshape(-1,)
    n = A.shape[0]
    info = {"success": False, "message": "", "iter": None, "advice": ""}

    try:
        detA = float(np.linalg.det(A))
        if detallado:
            mostrar_matriz(A, "A (para Cramer)")
            print(f"det(A) = {detA:.9g}")
        if abs(detA) < 1e-15 or not np.isfinite(detA):
            info["message"] = "det(A)≈0: no hay solución única; Cramer no aplica."
            return np.full(n, np.nan), info

        x = np.zeros(n)
        for i in range(n):
            Ai = A.copy()
            Ai[:, i] = b
            detAi = float(np.linalg.det(Ai))
            if detallado and n <= 6:
                mostrar_matriz(Ai, f"A_{i+1} (columna {i+1} reemplazada por b)")
                print(f"det(A_{i+1}) = {detAi:.9g}")
            elif detallado and n > 6:
                print(f"det(A_{i+1}) = {detAi:.9g} (no se imprime A_{i+1} para n>6)")

            x[i] = detAi / detA
            if detallado:
                print(f"x_{i+1} = det(A_{i+1}) / det(A) = {detAi:.9g} / {detA:.9g} = {x[i]:.9g}")

        info["success"] = True
        if detallado:
            mostrar_matriz(x.reshape(-1,1), "Solución x (Cramer)")
        # Aviso de condición
        cond = np.linalg.cond(A)
        if cond > 1e12:
            info["advice"] = "Advertencia: A está mal condicionada; Cramer puede ser numéricamente inestable."
        return x, info

    except np.linalg.LinAlgError as e:
        info["message"] = f"Falla numérica: {e}"
        return np.full(n, np.nan), info

def residuo(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(A @ x - b, 2))

def imprimir_resultado(metodo: str, x: np.ndarray, info: dict, A: np.ndarray, b: np.ndarray):
    print("\n" + "="*70)
    print(f"RESULTADO - Método: {metodo}")
    print("="*70)
    if info.get("success", False) and np.all(np.isfinite(x)):
        with np.printoptions(precision=8, suppress=True):
            print("x =\n", x.reshape(-1,))
        print(f"||Ax - b||_2 = {residuo(A, x, b):.3e}")
        if info.get("iter") is not None:
            print(f"Iteraciones: {info['iter']}")
        if info.get("advice"):
            print(info["advice"].strip())
    else:
        print("No se obtuvo solución.\nMotivo:", info.get("message","(desconocido)"))
        if "advice" in info and info["advice"]:
            print(info["advice"].strip())
    print("="*70 + "\n")

# ------------------ Programa principal ------------------

def main():
    print("Resolver sistemas lineales Ax=b (3x3 a 10x10)")
    metodo_idx = menu("Selecciona el método:", ["Gauss-Jordan", "Gauss-Seidel", "Cramer"])
    metodo = ["Gauss-Jordan", "Gauss-Seidel", "Cramer"][metodo_idx-1]

    print("\nElige el orden del sistema (3..10).")
    n = leer_entero_en_rango("n = ", 3, 10)

    # ¿Mostrar pasos?
    det_idx = menu("¿Quieres ver los PASOS del método?", ["Sí, modo detallado", "No, solo resultado"])
    detallado = (det_idx == 1)

    # A y b
    A = elegir_modo_entrada(n, n, "A", permite_ejemplo=True)
    b = elegir_modo_entrada(n, 1, "b", permite_ejemplo=True).reshape(-1,)

    # Vista previa del sistema completo
    mostrar_aumentada(A, b, "antes de resolver")

    # Resolver
    if metodo == "Gauss-Jordan":
        x, info = gauss_jordan(A, b, pivoteo=True, detallado=detallado)
    elif metodo == "Gauss-Seidel":
        if detallado:
            # Si hay pasos, también preguntamos parámetros
            try:
                tol = float(input("Tolerancia (p.ej. 1e-8) [1e-8]: ").strip() or "1e-8")
            except:
                tol = 1e-8
            try:
                maxit = int(input("Máximo de iteraciones [1000]: ").strip() or "1000")
            except:
                maxit = 1000
            x0 = None
            resp = (input("¿Deseas ingresar x0? (s/n) [n]: ").strip().lower() or "n")
            if resp.startswith("s"):
                x0 = leer_matriz_pegado(n, 1, "x0").reshape(-1,)
        else:
            tol, maxit, x0 = 1e-8, 1000, None
        x, info = gauss_seidel(A, b, tol=tol, maxit=maxit, x0=x0, detallado=detallado)
    else:  # Cramer
        x, info = cramer(A, b, detallado=detallado)

    imprimir_resultado(metodo, x, info, A, b)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelado por el usuario.")
